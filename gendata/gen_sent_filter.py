from argparse import ArgumentParser
import sys
sys.path.append('./')
import json
def gen_data(sentences, passages, anchors, sap_triples, args):
	cnt = 0
	fw = open(args.output_file, 'w')
	for sno in sap_triples:
		if len(sap_triples[sno]) >= 2:
			print(sno)
			outdata = {
				"sid": sno,
				"sentence_tokens": sentences[sno]['tokens'],
				"ano": [],
				"pno": [] 
			}
			for ano, pno in sap_triples[sno]:
				outdata['ano'] += [ano]
				outdata['pno'] += [pno]
			cnt += 1
			fw.write(json.dumps(outdata,ensure_ascii=False)+"\n")
	fw.close()
	return cnt

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--passage_file', type=str, required=True)
	parser.add_argument('--sentence_file', type=str, required=True)
	parser.add_argument('--anchor_file', type=str, required=True)
	parser.add_argument('--sap_file', type=str, required=True)

	parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
	parser.add_argument("--output_file", type=str, default='bert-base-uncased')

	args = parser.parse_args()
	
	sentences = {}
	with open(args.sentence_file) as sf:
		for line in sf:
			data = json.loads(line.strip())
			sentences[data['id']] = data
	
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
	
	sap_triples = {}
	with open(args.sap_file) as sf:
		for line in sf:
			data = json.loads(line.strip())
			ano = data['ano']
			pno = data['pno']
			sno = data['sno']
			if sno not in sap_triples:
				sap_triples[sno] = [[ano, pno]]
			else:
				sap_triples[sno].append([ano, pno])
	
	cnt = gen_data(sentences, passages, anchors, sap_triples, args)
	print(f"gen {cnt} sentence+passage datas")
