import os,sys,json
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import BertTokenizer
from utils.utils import *

def gen_data(docs, output_dir, bert_tokenizer):
	passage2id = {}
	sentence2id = {}
	anchor2id = {}
	
	sentences = []
	passages = []
	anchor_texts = []
	
	id_sentence = 1
	id_passage = 1
	id_anchor = 1
	
	fw = open(os.path.join(output_dir, "anchor_corpus.addid.txt"), "w")
	
	sap_triplet_list = []
	sap_dict = {}
	for i in tqdm(range(len(docs))):
		example = docs[i]
		newitem = {}

		sentence = example['sentence'].lower()
		anchors = example['anchors']
		newitem['sentence'] = sentence

		sent_id = None
		if sentence not in sentence2id:
			sentence2id[sentence] = id_sentence
			sentences.append(sentence)
			sent_id = id_sentence
			id_sentence += 1
		else:
			sent_id = sentence2id[sentence]
		newitem['sid'] = sent_id
		
		newanchors = []
		for j, anchor in enumerate(anchors):
			anchor = anchor[0]
			anchor_text = anchor['text'].lower()
			anchor_pos = anchor['pos']
			anchor_passage = anchor['passage'].lower()

			anchor_id = None
			if anchor_text not in anchor2id:
				anchor2id[anchor_text] = id_anchor
				anchor_texts.append(anchor_text)
				anchor_id = id_anchor
				id_anchor += 1
			else:
				anchor_id = anchor2id[anchor_text]

			anchor_passage_id = None
			if anchor_passage not in passage2id:
				passage2id[anchor_passage] = id_passage
				passages.append(anchor_passage)
				anchor_passage_id = id_passage
				id_passage += 1
			else:
				anchor_passage_id = passage2id[anchor_passage]
			
			sap_str = f"{sent_id}#{anchor_id}#{anchor_passage_id}"
			if sap_str not in sap_dict:
				sap_triplet_list += [(sent_id, anchor_id, anchor_passage_id)]
				sap_dict[sap_str] = 1
				
			newanchors.append({"text":anchor_text, "aid": anchor_id, "pos":anchor_pos, "passage": anchor_passage, "pid":anchor_passage_id})
		newitem['anchors'] = newanchors

		fw.write(json.dumps(newitem,ensure_ascii=False)+"\n")
	fw.close()

	valid_snos = {}
	valid_pnos = {}
	valid_anos = {}

	anchor_dic = {}
	sentence_dic = {}
	passage_dic = {}

	fw = open(os.path.join(output_dir, "anchor_sentences.txt"), "w")
	for i, sent in enumerate(sentences):
		sent_tokenized = bert_tokenizer.tokenize(sent)
		if len(sent_tokenized) >= 3:
			fw.write(json.dumps({"id":i+1,"text":sent,"tokens":sent_tokenized}, ensure_ascii=False)+"\n")
			sentence_dic[i+1] = {"id":i+1,"text":sent,"tokens":sent_tokenized}
			valid_snos[i+1] = 1
	fw.close()
	
	fw = open(os.path.join(output_dir, "anchor_passages.txt"), "w")
	for i, pas in enumerate(passages):
		pas_tokenized = bert_tokenizer.tokenize(pas)
		if len(pas_tokenized) >= 5:
			fw.write(json.dumps({"id":i+1,"text":pas,"tokens":pas_tokenized}, ensure_ascii=False)+"\n")
			passage_dic[i+1] = {"id":i+1,"text":pas,"tokens":pas_tokenized}
			valid_pnos[i+1] = 1
	fw.close()

	fw = open(os.path.join(output_dir, "anchor_anchors.txt"), "w")
	for i, anc in enumerate(anchor_texts):
		anchor_tokenized = bert_tokenizer.tokenize(anc)
		if len(anchor_tokenized) >= 1:
			fw.write(json.dumps({"id":i+1,"text":anc,"tokens":anchor_tokenized}, ensure_ascii=False)+"\n")
			anchor_dic[i+1] = {"id":i+1,"text":anc,"tokens":anchor_tokenized}
			valid_anos[i+1] = 1
	fw.close()

	saps = []
	fw = open(os.path.join(output_dir, "anchor_sap_triples.txt"), "w")
	for i, sap in enumerate(sap_triplet_list):
		sno, ano, pno = sap
		if sno in valid_snos and ano in valid_anos and pno in valid_pnos:
			fw.write(json.dumps({"sno":sno,"ano":ano,"pno":pno},ensure_ascii=False)+"\n")
			saps += [{"sno":sno,"ano":ano,"pno":pno}]
	fw.close()

	fw = open(os.path.join(output_dir, "sap_sentences.txt"),'w')
	for sap in saps:
		sno = sap['sno']
		ano = sap['ano']
		pno = sap['pno']

		anchor_tokens = anchor_dic[ano]['tokens']
		anchor_token = anchor_tokens[0]
		sentence_tokens = sentence_dic[sno]['tokens']

		try:
			anchor_token_idx = sentence_tokens.index(anchor_token)
		except:
			continue
		
		wdata = {}
		wdata['sid'] = sno
		wdata['aid'] = ano
		wdata['pid'] = pno
		wdata['tokens'] = sentence_tokens

		wdata['aidx'] = anchor_token_idx
		fw.write(json.dumps(wdata,ensure_ascii=False)+"\n")
	
	fw.close()



if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--train_corpus', type=str, required=True)
	parser.add_argument("--do_lower_case", action="store_true")
	parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
	parser.add_argument("--reduce_memory", action="store_true",
						help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")
	parser.add_argument('--output_dir', type=str, required=True)
	args = parser.parse_args()

	bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
		with open(args.train_corpus) as f:
			for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
				example = json.loads(line)
				docs.add_document(example)
		print('Reading file is done! Total doc num:{}'.format(len(docs)))
		gen_data(docs, args.output_dir, bert_tokenizer)


	
	

