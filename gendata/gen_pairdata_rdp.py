import os, sys, json, copy
from argparse import ArgumentParser
sys.path.append('./')
from tqdm import tqdm
from transformers import BertTokenizer
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from utils.utils import *

parser = ArgumentParser()
parser.add_argument("--do_lower_case", action="store_true")
parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
parser.add_argument("--reduce_memory", action="store_true",
						help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--mlm", action="store_true")
parser.add_argument("--masked_lm_prob", type=float, default=0.15,
						help="Probability of masking each token for the LM task")
parser.add_argument("--max_predictions_per_seq", type=int, default=60,
						help="Maximum number of tokens to mask in each sequence")
parser.add_argument("--pairnum_per_doc", type=int, default=2,
						help="How many samples for each document")
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--sentence_cls_weight_file', type=str, required=True)
parser.add_argument('--stop_words_file', type=str, required=True)
parser.add_argument('--anchor_file', type=str, required=True)
parser.add_argument('--passage_file', type=str, required=True)
args = parser.parse_args()

def construct_pairwise_examples(examples, chunk_indexs, max_seq_len, mlm, bert_tokenizer, masked_lm_prob, 
			max_predictions_per_seq, bert_vocab_list, epoch_filename, anchors, passages):
	# print(examples)
	num_examples = len(examples)
	print("num_examples", num_examples)
	num_instance = 0
	num_instances_value = 0
	ct_anchor_None = 0
	total_anchor_None = 0
	for doc_idx in tqdm(chunk_indexs):
		example = examples[doc_idx]
		if doc_idx % 100 == 0:
			print(f"doc_idx {doc_idx} num_instances_value {num_instances_value} total_anchor_none {total_anchor_None} ct_anchor_None {ct_anchor_None}")
		instances = [] 
		sentence_tokens = example['sentence_tokens']
		sid = example['sid']
		pnos = example['pno']
		anos = example['ano']
		num_anchors = len(anos)
		att_text = example['att_text']
		anchor_weights = []
		assert len(pnos) == len(anos)
		anchor_none = False
		for i, (ano, pno) in enumerate(zip(anos, pnos)):
			anchor_tokens = anchors[ano]['tokens']
			passage_tokens = passages[pno]['tokens']
			anchor_weight = cal_weight_of_anchor(anchor_tokens, att_text, 'avg')
			if anchor_weight == None:
				anchor_none = True
				break
			anchor_weights += [anchor_weight]
		total_anchor_None += 1
		if anchor_none:
			ct_anchor_None += 1
			continue
		query = sentence_tokens
		pairs = []
		for i in range(num_anchors):
			for j in range(i+1, num_anchors):
				p1 = pnos[i]
				p2 = pnos[j]
				a1_weight = anchor_weights[i]
				a2_weight = anchor_weights[j]
				if a1_weight < a2_weight:
					p1 = pnos[j]
					p2 = pnos[i]
				pairs += [(p1, p2)]

		for i, (pno_pos, pno_neg) in enumerate(pairs):
			pair_wise_instances = []
			for j, pid in enumerate([pno_pos, pno_neg]):
				if j == 0:
					label = 1
				else:
					label = 0
				anchor_passage_tokenized_copied = copy.deepcopy(passages[pid]['tokens'])
				# 不然会truncate两次，就不对了。
				try:
					truncate_seq_pair(query, anchor_passage_tokenized_copied, max_seq_len - 3)
				except:
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
			if len(pair_wise_instances) == 2:
				# instances += pair_wise_instances
				instances += [{"pos": pair_wise_instances[0], "neg": pair_wise_instances[1]}]
		doc_instances = [json.dumps(instance, ensure_ascii=False) for instance in instances]
		# lock.acquire()
		with open(epoch_filename,'a+') as epoch_file:
			for i, instance in enumerate(doc_instances):
				num_instances_value += 1
				epoch_file.write(instance + '\n')
				if num_instances_value % 100 == 0:
					print(num_instances_value)
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
	return stopwords, passages, anchors

if __name__ == '__main__':

	stopwords, passages, anchors = read_data()
	bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	bert_vocab_list = list(bert_tokenizer.vocab.keys())

	examples = []
	with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
		with open(args.sentence_cls_weight_file) as f:
			for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
				example = json.loads(line.strip())
				# examples.append(example)
				docs.add_document(example)
		print('Reading file is done! Total doc num:{}'.format(len(docs)))

		rdp_filename =  f"{args.output_dir}/task_rdp.json"
		if os.path.exists(rdp_filename):
			with open(rdp_filename, "w") as ef:
				print(f"start generating {rdp_filename}")
		num_processors = 1
		cand_idxs = list(range(0, len(docs)))
		chunk_size = int(len(cand_idxs) / num_processors)
		chunk_indexs = cand_idxs[0*chunk_size:(0+1)*chunk_size]
		num_instances_value = construct_pairwise_examples(docs, chunk_indexs, args.max_seq_len, args.mlm, bert_tokenizer, args.masked_lm_prob, \
				args.max_predictions_per_seq, bert_vocab_list, rdp_filename, anchors, passages)






			
			


