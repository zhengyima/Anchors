import os, sys, json, collections, shelve, copy
from argparse import ArgumentParser
sys.path.append('./')
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
from random import random, shuffle, choice, sample
from multiprocessing import Value, Lock
from tempfile import TemporaryDirectory
from pathlib import Path
from utils.utils import *

parser = ArgumentParser()
parser.add_argument('--train_corpus', type=str, required=True)
parser.add_argument("--do_lower_case", action="store_true")
parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
parser.add_argument("--reduce_memory", action="store_true",
						help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")
parser.add_argument("--epochs_to_generate", type=int, default=1,
						help="Number of epochs of data to pregenerate")
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--mlm", action="store_true")
parser.add_argument("--masked_lm_prob", type=float, default=0.15,
						help="Probability of masking each token for the LM task")
parser.add_argument("--max_predictions_per_seq", type=int, default=60,
						help="Maximum number of tokens to mask in each sequence")
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--passage_cls_weight_file', type=str, required=True)
parser.add_argument('--stop_words_file', type=str, required=True)
parser.add_argument('--passage_file', type=str, required=True)
parser.add_argument('--sentence_file', type=str, required=True)
parser.add_argument('--anchor_file', type=str, required=True)
parser.add_argument('--sentence_anchor_weight_file', type=str, required=True)
args = parser.parse_args()

def construct_pairwise_examples(examples, chunk_indexs, max_seq_len, mlm, bert_tokenizer, masked_lm_prob, 
			max_predictions_per_seq, bert_vocab_list, epoch_filename, word_weights, 
			anchors_dic, passages, sentence_anchor_weights):
	num_examples = len(examples)
	print("num_examples", num_examples)
	num_instance = 0
	niv = 0
	st_ls3 = 0 
	st_aid = 0
	st_pid = 0
	total_trunc = 0
	err_trunc = 0
	err_len = 0
	total_len = 0
	for doc_idx in tqdm(chunk_indexs):
		print(doc_idx)
		if doc_idx % 100 == 0:
			print(f"doc_idx {doc_idx}, niv {niv}, err_trunc {err_trunc}, total_trunc {total_trunc}, err_len {err_len}, total_len {total_len}")
		example = examples[doc_idx]
		sentence = example['sentence']
		sid = example['sid']
		sentence_tokenized = bert_tokenizer.tokenize(sentence)
		if len(sentence_tokenized) < 3:
			st_ls3 += 1
			continue
		anchors = example['anchors']
		instances = []
		for i, anchor in enumerate(anchors):
			text = anchor['text']
			idx_start, idx_end = anchor['pos']
			anchor_passage = anchor['passage']
			aid = anchor['aid']
			pid = anchor['pid']
			if int(aid) not in anchors_dic:
				st_aid += 1
				continue
			if int(pid) not in passages:
				st_pid += 1
				continue 
			anchor_passage_tokenized = bert_tokenizer.tokenize(anchor_passage)
			if pid not in word_weights:
				continue
			passage_word_weights = word_weights[pid] 
			if f"{sid}@{pid}@{aid}" not in sentence_anchor_weights:
				continue
			sentence_anchor_weights_this = sentence_anchor_weights[f"{sid}@{pid}@{aid}"]
			pos_query = sample_query_from_sentence(sid, aid, pid, sentence_anchor_weights_this)
			pos_query += bert_tokenizer.tokenize(anchor['text'])
			


			if len(pos_query) < 1:
				continue
			neg_query = sample_neg_from_passage(anchor_passage_tokenized, passage_word_weights)
			if len(neg_query) <= 1:
				continue
			if len(pos_query) <= 1:
				continue
			if len(anchor_passage_tokenized) <= 5:
				continue
			pair_wise_instances = []
			for j, query in enumerate([pos_query, neg_query]):
				if j == 0:
					label = 1
				else:
					label = 0
				anchor_passage_tokenized_copied = copy.deepcopy(anchor_passage_tokenized)
				# 不然会truncate两次，就不对了。
				total_trunc += 1
				try:
					truncate_seq_pair(query, anchor_passage_tokenized_copied, max_seq_len - 3)
				except:
					err_trunc += 1
					break
				tokens = ["[CLS]"] + query + ["[SEP]"] + anchor_passage_tokenized_copied + ["[SEP]"]
				segment_ids = [0 for _ in range(len(query) + 2)] + [1 for _ in range(len(anchor_passage_tokenized_copied) + 1)]
				if mlm:
				    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
				        tokens, masked_lm_prob, max_predictions_per_seq, True, bert_vocab_list)
				else:
				    masked_lm_positions, masked_lm_labels = [], []
				tokens_idx = bert_tokenizer.convert_tokens_to_ids(tokens)
				tokens_idx_labels = bert_tokenizer.convert_tokens_to_ids(masked_lm_labels)
				instance = {
				    "tokens_idx":tokens_idx,
					"tokens":tokens,
				    "segment_ids": segment_ids,
				    "label": label,
				    "masked_lm_positions": masked_lm_positions,
					"masked_lm_labels_idxs": tokens_idx_labels,
				    }

				# 确保每次是pair-wise的！
				pair_wise_instances.append(instance)
			total_len += 1
			if len(pair_wise_instances) == 2:
				instances += [{"pos": pair_wise_instances[0], "neg": pair_wise_instances[1]}]
			else:
				err_len += 1

		doc_instances = [json.dumps(instance, ensure_ascii=False) for instance in instances]
		with open(epoch_filename,'a+') as epoch_file:
			for i, instance in enumerate(doc_instances):
				epoch_file.write(instance + '\n')
				niv += 1
				if niv % 100 == 0:
					print("niv", niv)
	return niv

def read_data():
	stopwords = set()
	with open(args.stop_words_file, 'r') as sf:
		for line in tqdm(sf):
			stopwords.add(line.strip().lower())
	
	word_weights = {}
	with open(args.passage_cls_weight_file, 'r') as pf:
		for line in tqdm(pf):
			data = json.loads(line.strip())
			pid = data['pid']
			att_texts = data['att_text']
			weight_dic = {}
			for word, weight in att_texts:
				if word in stopwords:
					continue
				if word in weight_dic:
					weight_dic[word] += weight
				else:
					weight_dic[word] = weight
			word_weights[pid] = weight_dic
	
	sentences = {}	
	with open(args.sentence_file) as sf:
		for line in tqdm(sf):
			data = json.loads(line.strip())
			sentences[int(data['id'])] = 1
	
	passages = {}
	with open(args.passage_file) as af:
		for line in tqdm(af):
			data = json.loads(line.strip())
			passages[int(data['id'])] = 1

	anchors = {}
	with open(args.anchor_file) as af:
		for line in tqdm(af):
			data = json.loads(line.strip())
			anchors[int(data['id'])] = 1
	
	sentence_anchor_weights = {}
	with open(args.sentence_anchor_weight_file) as sawf:
		for line in tqdm(sawf):
			data = json.loads(line.strip())
			sid = data['sid']
			aid = data['aid']
			pid = data['pid']
			att_texts = data['att_text']
			weight_dic = {}
			for word, weight in att_texts:
				if word in stopwords:
					continue
				if word in weight_dic:
					weight_dic[word] += weight
				else:
					weight_dic[word] = weight
			sentence_anchor_weights[f"{sid}@{pid}@{aid}"] = weight_dic
	return stopwords, word_weights, sentences, passages, anchors, sentence_anchor_weights

if __name__ == '__main__':
	bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	bert_vocab_list = list(bert_tokenizer.vocab.keys())
	stopwords, word_weights, sentences, passages, anchors, sentence_anchor_weights = read_data()
	examples = []
	with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
		with open(args.train_corpus) as f:
			for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
				example = json.loads(line)
				docs.add_document(example)
		print('Reading file is done! Total doc num:{}'.format(len(docs)))
		rqp_filename =  f"{args.output_dir}/task_rqp.json"
		if os.path.exists(rqp_filename):
			with open(rqp_filename, "w") as ef:
				print(f"start generating {rqp_filename}")
		num_processors = 1
		cand_idxs = list(range(0, len(docs)))
		chunk_size = int(len(cand_idxs) / num_processors)
		chunk_indexs = cand_idxs[0*chunk_size:(0+1)*chunk_size]
		print("len of docs:", len(docs), chunk_indexs)
		niv = construct_pairwise_examples(docs, chunk_indexs, args.max_seq_len, args.mlm, bert_tokenizer, args.masked_lm_prob, \
			args.max_predictions_per_seq, bert_vocab_list, rqp_filename, word_weights, anchors, passages, sentence_anchor_weights)




