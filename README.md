# Anchors
Source code of CIKM2021 Long Paper: 

[Pre-training for Ad-hoc Retrieval: Hyperlink is Also You Need](https://arxiv.org/abs/2108.09346),

including the following two parts: 
- Pre-training on corpus based on hyperlinks ✅
- Fine-tuning on MS MARCO Document Ranking Datasets 🌀

## Preinstallation
First, prepare a **Python3** environment, and run the following commands:
```
  git clone https://github.com/zhengyima/anchors.git anchors
  cd anchors
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Besides, you should download the BERT model checkpoint in format of huggingface transformers, and save them in a directory ```BERT_MODEL_PATH```. In our paper, we use the version of ```bert-base-uncased```. you can download it from the huggingface official [model zoo](https://huggingface.co/bert-base-uncased/tree/main), or [Tsinghua mirror](https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/).

## Pre-training on Raw Corpus

### Prepare the Corpus Data

The corpus data should have one passage (in JSON format) per line, with the anchor texts saved in an array. e.g.
```
{
	"sentence": one sentence s in source page,
	"anchors": [
		{
			"text": anchor text of anchor a1,
			"pos": [the start index of a1 in s, the end index of a1 in s],
			"passage": the destination page of a1
		},
		{
			"text": anchor text of anchor a2,
			"pos": [the start index of a2 in s, the end index of a2 in s],
			"passage": the destination page of a2
		},
		...
	] 
}
```

For your convenience, we provide the demo corpus file ```data/corpus/demo_data.txt```. You can refer to the demo data to generate the pre-trained corpus, such as from [Wikipedia dump](https://dumps.wikimedia.org/enwiki/).


### Generate Pre-training Samples from the Corpus 

The process of generating the pre-training samples are complex, which has a long pipeline including four pre-training tasks. Thus, we build a shell ```shells/gendata.sh``` to complete the whole process. If you are interested in the detailed process, you can refer to the shell. If you just want to run the code, you can run the following:
```
 export CORPUS_DATA=./data/corpus/demo_data.txt
 export DATA_PATH=./data/
 export BERT_MODEL_PATH=/path/to/bert_model
 bash shells/gendata.sh
```

After running ```gendata.sh``` success, you will get the pre-training data stored in ```DATA_PATH/merged/```.

### Running Pre-training
```
 export PERTRAIN_OUTPUT_DIR=/path/to/output_path
 bash shells/pretrain.sh
```

## Fine-tuning on MS MARCO

The process of fine-tuning is more complex than pre-training 💤

Thus, the author decides to pack and clean the fine-tuning part when he is free, such as the next weekend.

**Notes**: Since the pre-training of our model is completed in the standard manner of huggingface. So, you can apply the output checkpoints of pre-training into any down-stream method, just like using ```bert-base-uncased```. 


## Citations
If you use the code and datasets, please cite the following paper:  

```
@article{DBLP:journals/corr/abs-2108-09346,
  author    = {Zhengyi Ma and
               Zhicheng Dou and
               Wei Xu and
               Xinyu Zhang and
               Hao Jiang and
               Zhao Cao and
               Ji{-}Rong Wen},
  title     = {Pre-training for Ad-hoc Retrieval: Hyperlink is Also You Need},
  booktitle = {{CIKM} '21: The 30th {ACM} International Conference on Information
               and Knowledge Management, Virtual Event, QLD, Australia, November 1-5, 2021},
  publisher = {{ACM}},
  year      = {2021}
}
```

## Links
- [Wikipedia dump](https://dumps.wikimedia.org/enwiki/)
- [WikiExtractor](https://github.com/attardi/wikiextractor)
- [MS MARCO Document Ranking](https://github.com/microsoft/MSMARCO-Document-Ranking)
- [TREC 2019 Deep Learning](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html)
- [Pytorch](https://pytorch.org)
- [Huggingface Transformers](https://huggingface.co/)


