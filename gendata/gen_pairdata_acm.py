import os, sys, json, copy, traceback
from argparse import ArgumentParser
sys.path.append('./')
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from utils.utils import *
parser = ArgumentParser()
parser.add_argument("--do_lower_case", action="store_true")
parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
parser.add_argument("--reduce_memory", action="store_true",
						help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")
	# parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--mlm", action="store_true")
parser.add_argument("--masked_lm_prob", type=float, default=0.15,
						help="Probability of masking each token for the LM task")
parser.add_argument("--max_predictions_per_seq", type=int, default=60,
						help="Maximum number of tokens to mask in each sequence")
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--sentence_file_path', type=str, required=True)
parser.add_argument('--stop_words_file', type=str, required=True)
parser.add_argument('--anchor_file', type=str, required=True)
parser.add_argument('--passage_file', type=str, required=True)
parser.add_argument('--passage_cls_weight_file', type=str, required=True)
args = parser.parse_args()

def construct_pairwise_examples(examples, chunk_indexs, max_seq_len, mlm, bert_tokenizer, masked_lm_prob, 
			max_predictions_per_seq, bert_vocab_list, epoch_filename, anchors, passages, word_weights):
	# print(examples)
	num_examples = len(examples)
	print("num_examples", num_examples)
	num_instance = 0
	pno_list = list(passages.keys())
	num_instances_value = 0
	for doc_idx in tqdm(chunk_indexs):
		# print(doc_idx)
		if doc_idx % 100 == 0:
			print(doc_idx)
		example = examples[doc_idx]
		instances = [] 
		sentence_tokens = example['sentence_tokens']
		sid = example['sid']
		pnos = example['pno']
		anos = example['ano']
		num_anchors = len(anos)
		assert len(pnos) == len(anos)
		pairs = []
		for i in range(num_anchors):
			for j in range(i+1, num_anchors):
				p1 = pnos[i]
				p2 = pnos[j]
				a1 = anos[i]
				a2 = anos[j]
				pneg = select_neg_from_allpassages(pno_list, p1, p2)
				pairs += [(a1, a2, p1, p2, pneg)]

		for i, (a1, a2, p1, p2, pno_neg) in enumerate(pairs):
			pair_wise_instances = []
			for j, (pp1, pp2) in enumerate([(p1, p2), (p1, pno_neg)]):
				if pp1 not in word_weights:
					continue
				passage_word_weights = word_weights[pp1] 
				if j == 0:
					label = 1
				else:
					label = 0
				anchor_p1_tokenized_copied = sample_neg_from_passage(passages[pp1]['tokens'], passage_word_weights)
				anchor_p1_tokenized_copied += anchors[int(a1)]['tokens']
				anchor_p2_tokenized_copied = copy.deepcopy(passages[pp2]['tokens'])
				anchor_p1_tokenized_copied = anchor_p1_tokenized_copied[:250]
				anchor_p2_tokenized_copied = anchor_p2_tokenized_copied[:250]
				try:
					truncate_seq_pair(anchor_p1_tokenized_copied, anchor_p2_tokenized_copied, max_seq_len - 3)
				except:
					break
				tokens = ["[CLS]"] + anchor_p1_tokenized_copied + ["[SEP]"] + anchor_p2_tokenized_copied + ["[SEP]"]
				segment_ids = [0 for _ in range(len(anchor_p1_tokenized_copied) + 2)] + [1 for _ in range(len(anchor_p2_tokenized_copied) + 1)]
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
			if len(pair_wise_instances) == 2:
				# instances += pair_wise_instances
				instances += [{"pos": pair_wise_instances[0], "neg": pair_wise_instances[1]}]
		doc_instances = [json.dumps(instance, ensure_ascii=False) for instance in instances]
		# lock.acquire()
		with open(epoch_filename,'a+') as epoch_file:
			for i, instance in enumerate(doc_instances):
				epoch_file.write(instance + '\n')
				num_instances_value += 1
	return num_instances_value

def read_data():
	stopwords = set()
	with open(args.stop_words_file, 'r') as sf:
		for line in sf:
			stopwords.add(line.strip().lower())
	
	passages = {}
	with open(args.passage_file) as af:
		for line in af:
			data = json.loads(line.strip())
			passages[data['id']] = data

	anchors = {}
	with open(args.anchor_file) as af:
		for line in af:
			data = json.loads(line.strip())
			anchors[data['id']] = data
	
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
	return stopwords, passages, anchors, word_weights
	
if __name__ == '__main__':
	bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	bert_vocab_list = list(bert_tokenizer.vocab.keys())
	word2df = None
	stopwords, passages, anchors, word_weights = read_data()
	examples = []
	with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
		with open(args.sentence_file_path) as f:
			for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
				example = json.loads(line.strip())
				# examples.append(example)
				docs.add_document(example)
		print('Reading file is done! Total doc num:{}'.format(len(docs)))
		acm_filename =  f"{args.output_dir}/task_acm.json"
		if os.path.exists(acm_filename):
			with open(acm_filename, "w") as ef:
				print(f"start generating {acm_filename}")
		cand_idxs = list(range(0, len(docs)))
		chunk_size = int(len(cand_idxs) / 1)
		chunk_indexs = cand_idxs[0*chunk_size:(0+1)*chunk_size]
		num_instances_value = construct_pairwise_examples(docs, chunk_indexs, args.max_seq_len, args.mlm, bert_tokenizer, args.masked_lm_prob, \
			args.max_predictions_per_seq, bert_vocab_list, acm_filename, anchors, passages, word_weights)
		







			
			


