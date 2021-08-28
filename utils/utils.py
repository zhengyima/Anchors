import os
from argparse import ArgumentParser
import sys
sys.path.append('./')
from tqdm import tqdm
import json
from transformers import BertTokenizer
import random
from random import shuffle, choice, sample
import collections
import traceback
from multiprocessing import Pool, Value, Lock
from tempfile import TemporaryDirectory
from pathlib import Path
import shelve
import numpy as np
import torch
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",["index", "label"])
lock = Lock()
num_instances = Value('i', 0)
num_docs = Value('i', 0)
num_words = Value('i', 0)
TEMP_DIR = './'
TEMP_DIR = './'

class DocumentDatabase:
	def __init__(self, reduce_memory=False):
		if reduce_memory:
			self.temp_dir = TemporaryDirectory(dir=TEMP_DIR)
			self.working_dir = Path(self.temp_dir.name)
			self.document_shelf_filepath = self.working_dir / 'shelf.db'
			self.document_shelf = shelve.open(str(self.document_shelf_filepath),
											  flag='n', protocol=-1)
			self.documents = None
		else:
			self.documents = []
			self.document_shelf = None
			self.document_shelf_filepath = None
			self.temp_dir = None
		self.doc_lengths = []
		self.doc_cumsum = None
		self.cumsum_max = None
		self.reduce_memory = reduce_memory

	def add_document(self, document):
		if not document:
			return
		if self.reduce_memory:
			current_idx = len(self.doc_lengths)
			self.document_shelf[str(current_idx)] = document
		else:
			self.documents.append(document)
		self.doc_lengths.append(len(document))

	def __len__(self):
		return len(self.doc_lengths)

	def __getitem__(self, item):
		if self.reduce_memory:
			return self.document_shelf[str(item)]
		else:
			return self.documents[item]

	def __contains__(self, item):
		if str(item) in self.document_shelf:
			return True
		else:
			return False

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, traceback):
		if self.document_shelf is not None:
			self.document_shelf.close()
		if self.temp_dir is not None:
			self.temp_dir.cleanup()

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
	"""Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_num_tokens:
			break
		# truncate from the doc side
		tokens_b.pop()

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
	"""Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
	with several refactors to clean it up and remove a lot of unnecessary variables."""
	cand_indices = []
	# [MASK] word from DOC, not the query
	START_DOC = False
	for (i, token) in enumerate(tokens):
		if token == "[SEP]":
			START_DOC = True
			continue
		if token == "[CLS]":
			continue
		if not START_DOC:
			continue
		# Whole Word Masking means that if we mask all of the wordpieces
		# corresponding to an original word. When a word has been split into
		# WordPieces, the first token does not have any marker and any subsequence
		# tokens are prefixed with ##. So whenever we see the ## token, we
		# append it to the previous set of word indexes.
		#
		# Note that Whole Word Masking does *not* change the training code
		# at all -- we still predict each WordPiece independently, softmaxed
		# over the entire vocabulary.
		if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
			cand_indices[-1].append(i)
		else:
			cand_indices.append([i])

	num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(cand_indices) * masked_lm_prob))))
	shuffle(cand_indices)
	# print(tokens)
	# print(cand_indices, num_to_mask)
	mask_indices = sorted(sample(cand_indices, num_to_mask))
	masked_lms = []
	covered_indexes = set()
	for index_set in cand_indices:
		if len(masked_lms) >= num_to_mask:
			break
		# If adding a whole-word mask would exceed the maximum number of
		# predictions, then just skip this candidate.
		if len(masked_lms) + len(index_set) > num_to_mask:
			continue
		is_any_index_covered = False
		for index in index_set:
			if index in covered_indexes:
				is_any_index_covered = True
				break
		if is_any_index_covered:
			continue
		for index in index_set:
			covered_indexes.add(index)

			masked_token = None
			# 80% of the time, replace with [MASK]
			if random.random() < 0.8:
				masked_token = "[MASK]"
			else:
				# 10% of the time, keep original
				if random.random() < 0.5:
					masked_token = tokens[index]
				# 10% of the time, replace with random word
				else:
					masked_token = choice(vocab_list)
			masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
			tokens[index] = masked_token

	assert len(masked_lms) <= num_to_mask
	masked_lms = sorted(masked_lms, key=lambda x: x.index)
	mask_indices = [p.index for p in masked_lms]
	masked_token_labels = [p.label for p in masked_lms]

	return tokens, mask_indices, masked_token_labels

def error_callback(e):
	print('error')
	print(dir(e), "\n")
	traceback.print_exception(type(e), e, e.__traceback__)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 0, keepdims = True)
    s = x_exp / x_sum    
    return s

def sample_query_from_sentence(sid, aid, pid, sentence_anchor_weights):
	len_query = int(np.random.normal(loc=3.0, scale=1.0, size=None))
	tq = 0
	while tq < 100 and len_query < 1:
		len_query = int(np.random.normal(loc=3.0, scale=1.0, size=None))
		tq += 1
	if tq == 100:
		len_query = 3
	words = list(sentence_anchor_weights.keys())
	if len(words) == 0:
		return []
	word_weights = [sentence_anchor_weights[w] for w in words]
	normalized_word_weights = softmax(np.array(word_weights))
	samples = np.random.choice(a=len(words), size=len_query, replace=True, p=normalized_word_weights)
	word_set = [words[s] for s in samples]
	return word_set


def sample_neg_from_passage(anchor_passage_tokenized, p_word_weights):
	len_query = int(np.random.normal(loc=3.0, scale=1.0, size=None))
	tq = 0
	while tq < 100 and len_query < 1:
		len_query = int(np.random.normal(loc=3.0, scale=1.0, size=None))
		tq += 1
	if tq == 100:
		len_query = 3
	len_pos_query = len_query
	len_anchor_passage = len(anchor_passage_tokenized)
	i = 0
	if len_anchor_passage == 0:
		return []
	words = list(p_word_weights.keys())
	if len(words) == 0:
		return []
	word_weights = [p_word_weights[w] for w in words]
	normalized_word_weights = softmax(np.array(word_weights))
	samples = np.random.choice(a=len(words), size=len_pos_query, replace=True, p=normalized_word_weights)
	word_set = [words[s] for s in samples]
	return word_set


def cal_weight_of_anchor(anchor_tokens, att_text, mode='avg'):

	anchor_tokens_str = ' '.join(anchor_tokens)
	sent_tokens_str = ' '.join([w for w, ww in att_text])
	# print(anchor_tokens_str in sent_tokens_str)
	anchor_len = len(anchor_tokens)
	sent_len = len(att_text)
	start = -1
	for i in range(sent_len):
		cnt = 0
		for j in range(anchor_len):
			if i+j < sent_len and anchor_tokens[j] == att_text[i+j][0]:
				cnt += 1
			else:
				break
		if cnt == anchor_len:
			start = i
			break
	# print(start)

	sum_weights = -1
	avg_weights = -1
	max_weights = -1
	if start != -1:
		anchor_weights = [att_text[start+i][1] for i in range(anchor_len)]
		sum_weights = sum(anchor_weights)
		avg_weights = sum(anchor_weights) / anchor_len
		max_weights = max(anchor_weights)
	else:
		# print(f"[warning] anchor {anchor_tokens_str} not found in the sentence {sent_tokens_str}!")
		return None
	
	if mode == 'avg':
		return avg_weights
	elif mode == "sum":
		return sum_weights
	elif mode == "max":
		return max_weights
	else:
		return None


def select_neg_from_allpassages(pno_list, p1, p2):
	cc = 0
	while True and cc < 100:
		pneg = random.choice(pno_list)
		if pneg != p1 and pneg != p2:
			break
		cc += 1
	return pneg
