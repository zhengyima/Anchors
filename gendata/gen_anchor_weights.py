import sys, os, argparse
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from models import BertForAnchorWeight
from mydatasets import SentenceGenAnchorWeightDataset
import os
from tqdm import tqdm
import json
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--per_gpu_test_batch_size", default=64, type=int, help="The batch size.")
parser.add_argument("--model_path", default="/home/yutao_zhu/BertModel/", type=str, help="The path to save log.")
parser.add_argument("--passage_file_path", default="/home/yutao_zhu/BertModel/", type=str, help="The path to save log.")
parser.add_argument("--dataset_script_dir", default="/home/yutao_zhu/BertModel/", type=str, help="The path to save log.")
parser.add_argument("--dataset_cache_dir", default="/home/yutao_zhu/BertModel/", type=str, help="The path to save log.")
parser.add_argument("--output_file_path", default="/home/yutao_zhu/BertModel/", type=str, help="The path to save log.")
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
if args.cpu:
	device = torch.device("cpu")
	args.test_batch_size = args.per_gpu_test_batch_size * 1
else:
	device = torch.device("cuda:0")
	args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
print(args)
predict_data = args.passage_file_path
tokenizer = BertTokenizer.from_pretrained(args.model_path)

def predict(model, X_test):
	model.eval()
	test_dataset = SentenceGenAnchorWeightDataset(X_test, 512, tokenizer, args.dataset_script_dir, args.dataset_cache_dir)
	test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
	y_pred = []
	y_label = []
	fw = open(args.output_file_path, 'w')
	start = 0
	with torch.no_grad():
		epoch_iterator = tqdm(test_dataloader)
		for i, test_data in enumerate(epoch_iterator):
			with torch.no_grad():
				for key in test_data.keys():
					test_data[key] = test_data[key].to(device)
			attentions = model(test_data)
			output_attention_weights =  attentions.data.cpu().numpy().tolist()
			attention_weights = output_attention_weights
			for j in range(start, start + len(attention_weights)):
				pid = test_dataset.passage_dataset[j]['pid']
				sid = test_dataset.passage_dataset[j]['sid']
				aid = test_dataset.passage_dataset[j]['aid']
				aidx = test_dataset.passage_dataset[j]['aidx']
				ptext = test_dataset.passage_dataset[j]['tokens']
				plen = len(ptext)
				cls_attention = attention_weights[j - start][aidx + 1][1:plen+1]
				if len(cls_attention) != plen:
					ptext = ptext[:500]
					cls_attention =  attention_weights[j - start][1:500+1]
				att_text = []
				for pt, pa in zip(ptext, cls_attention):
					att_text += [[pt, pa]]
				att_text = sorted(att_text, key=lambda x: x[1], reverse=True)
				fw.write(json.dumps({"att_text": att_text, "pid":pid, "sid":sid, "aid":aid, "aidx":aidx}, ensure_ascii=False)+"\n")
			start += len(attention_weights)
		fw.close()
	return y_pred, y_label

if __name__ == '__main__':
	set_seed()
	bert_model = BertModel.from_pretrained(args.model_path,output_attentions=True)
	model = BertForAnchorWeight(bert_model)
	model = model.to(device)
	model = torch.nn.DataParallel(model)
	predict(model, predict_data)
