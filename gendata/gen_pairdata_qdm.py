from argparse import ArgumentParser
import sys, os, json, copy, collections
sys.path.append('./')
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
										  ["index", "label"])
sys.path.append('./')
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from utils.utils import *

parser = ArgumentParser()
parser.add_argument('--passage_file', type=str, required=True)
parser.add_argument('--sentence_file', type=str, required=True)
parser.add_argument('--anchor_file', type=str, required=True)
parser.add_argument('--sap_file', type=str, required=True)
parser.add_argument('--max_pair_perquery', type=int, required=True)
parser.add_argument('--max_seq_len', type=int, required=True)
parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
parser.add_argument("--output_dir", type=str, default='bert-base-uncased')
parser.add_argument("--mlm", action="store_true")
parser.add_argument("--masked_lm_prob", type=float, default=0.15,
						help="Probability of masking each token for the LM task")
parser.add_argument("--max_predictions_per_seq", type=int, default=60,
						help="Maximum number of tokens to mask in each sequence")
parser.add_argument("--do_lower_case", action="store_true")
parser.add_argument('--sentence_anchor_weight_file', type=str, required=True)
parser.add_argument('--stop_words_file', type=str, required=True)
args = parser.parse_args()

def gen_data(sentences, passages, anchors, sap_triples, sap_pnos, args, epoch_filename, bert_vocab_list, sentence_anchor_weights, stopwords):
	num_instance_value = 0
	for ano in tqdm(sap_triples, desc="anchors"):
		if num_instance_value % 1000 == 0:
			print(num_instance_value)
		instances = []
		pnos_list = sap_pnos[ano]
		if len(pnos_list) <= 1:
			continue
		len_senthasq = len(sap_triples[ano])
		max_iter = min(len_senthasq, args.max_pair_perquery)
		perm = np.random.permutation(len_senthasq)
		for i in range(max_iter):
			idx = perm[i]
			sno, pno = sap_triples[ano][i]
			if f"{sno}@{pno}@{ano}" not in sentence_anchor_weights:
				continue
			sentence_anchor_weights_this = sentence_anchor_weights[f"{sno}@{pno}@{ano}"]
			query = sample_query_from_sentence(sno, pno, ano, sentence_anchor_weights_this)
			query += anchors[ano]['tokens']
			pos_passage = passages[pno]['tokens']
			neg_pnos_list = []
			for p in pnos_list:
				if p != pno:
					neg_pnos_list += [p]
			a1 = np.random.choice(a=len(neg_pnos_list), size=1)
			neg_pno = neg_pnos_list[a1[0]]
			neg_passage = passages[neg_pno]['tokens']
			pair_wise_instances = []
			for j, pas in enumerate([pos_passage, neg_passage]):
				if j == 0:
					label = 1
				else:
					label = 0
				anchor_passage_tokenized_copied = copy.deepcopy(pas)
				try:
					truncate_seq_pair(query, anchor_passage_tokenized_copied, args.max_seq_len - 3)
				except:
					break
				tokens = ["[CLS]"] + query + ["[SEP]"] + anchor_passage_tokenized_copied + ["[SEP]"]
				segment_ids = [0 for _ in range(len(query) + 2)] + [1 for _ in range(len(anchor_passage_tokenized_copied) + 1)]
				if args.mlm:
				    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
				        tokens, args.masked_lm_prob, args.max_predictions_per_seq, True, bert_vocab_list)
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
				# print("test")
				instances += [{"pos": pair_wise_instances[0], "neg": pair_wise_instances[1]}]
		doc_instances = [json.dumps(instance, ensure_ascii=False) for instance in instances]
		# lock.acquire()
		with open(epoch_filename,'a+') as epoch_file:
			for i, instance in enumerate(doc_instances):
				epoch_file.write(instance + '\n')
				num_instance_value += 1
	return num_instance_value
		
def read_data():
	stopwords = set()
	with open(args.stop_words_file, 'r') as sf:
		for line in tqdm(sf):
			stopwords.add(line.strip().lower())
	sentences = {}
	with open(args.sentence_file) as sf:
		for line in tqdm(sf):
			data = json.loads(line.strip())
			sentences[data['id']] = data
	passages = {}
	with open(args.passage_file) as af:
		for line in tqdm(af):
			data = json.loads(line.strip())
			passages[data['id']] = data
	anchors = {}
	with open(args.anchor_file) as af:
		for line in tqdm(af):
			data = json.loads(line.strip())
			anchors[data['id']] = data
	sap_triples = {}
	sap_pnos = {}
	with open(args.sap_file) as sf:
		for line in tqdm(sf):
			data = json.loads(line.strip())
			ano = data['ano']
			pno = data['pno']
			sno = data['sno']
			if ano not in sap_triples:
				sap_triples[ano] = [[sno, pno]]
			else:
				sap_triples[ano].append([sno, pno])
			
			if ano not in sap_pnos:
				sap_pnos[ano] = []
			if pno not in sap_pnos[ano]:
				sap_pnos[ano] += [pno]
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
	return stopwords, sentences, passages, anchors, sap_triples, sap_pnos, sentence_anchor_weights

if __name__ == '__main__':
	bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	bert_vocab_list = list(bert_tokenizer.vocab.keys())
	stopwords, sentences, passages, anchors, sap_triples, sap_pnos, sentence_anchor_weights = read_data()
	qdm_filename =  f"{args.output_dir}/task_qdm.json"
	num_instance_value = gen_data(sentences, passages, anchors, sap_triples, sap_pnos, args, qdm_filename, bert_vocab_list, sentence_anchor_weights, stopwords)

