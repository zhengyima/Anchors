# preprocess the data
export DATA_CORPUS_DIR=$DATA_PATH/corpus
export TASK_DATA_DIR=$DATA_PATH/tasks
export MERGED_DATA_DIR=$DATA_PATH/merged

mkdir -p  $DATA_CORPUS_DIR
python preprocess/gen_data_addpid.py --train_corpus $CORPUS_DATA \
  --bert_model $BERT_MODEL_PATH --do_lower_case \
  --output_dir $DATA_CORPUS_DIR --reduce_memory

python gendata/gen_passage_weights.py --per_gpu_test_batch_size 2 \
  --model_path $BERT_MODEL_PATH \
  --passage_file_path $DATA_CORPUS_DIR/anchor_passages.txt \
  --dataset_script_dir ./data_scripts --dataset_cache_dir /tmp/anchor_cache \
  --output_file_path $DATA_CORPUS_DIR/passage_cls_weight.txt

python gendata/gen_anchor_weights.py --per_gpu_test_batch_size 2 \
 --model_path $BERT_MODEL_PATH  \
 --passage_file_path $DATA_CORPUS_DIR/sap_sentences.txt \
 --dataset_script_dir  ./data_scripts --dataset_cache_dir /tmp/anchor_cache \
 --output_file_path $DATA_CORPUS_DIR/sentence_anchor_weight.txt

python gendata/gen_sent_filter.py --passage_file $DATA_CORPUS_DIR/anchor_passages.txt \
		--sentence_file $DATA_CORPUS_DIR/anchor_sentences.txt --sap_file $DATA_CORPUS_DIR/anchor_sap_triples.txt \
		--anchor_file $DATA_CORPUS_DIR/anchor_anchors.txt --bert_model $BERT_MODEL_PATH \
		--output_file $DATA_CORPUS_DIR/sentence_passages.txt
python gendata/gen_sent_weights.py --per_gpu_test_batch_size 128 \
		--model_path $BERT_MODEL_PATH \
		--sentence_file_path $DATA_CORPUS_DIR/sentence_passages.txt \
		--dataset_script_dir ./data_scripts \
		--dataset_cache_dir /tmp/mycache \
		--output_file_path $DATA_CORPUS_DIR/sentence_cls_weight.txt
mkdir -p $TASK_DATA_DIR
# gen data for task RQP
python gendata/gen_pairdata_rqp.py \
		--train_corpus $DATA_CORPUS_DIR/anchor_corpus.addid.txt \
		--bert_model  $BERT_MODEL_PATH \
		--stop_words_file $DATA_CORPUS_DIR/stopwords.txt  \
		--passage_cls_weight_file  $DATA_CORPUS_DIR/passage_cls_weight.txt \
		--passage_file $DATA_CORPUS_DIR/anchor_passages.txt \
		--sentence_file $DATA_CORPUS_DIR/anchor_sentences.txt \
		--anchor_file $DATA_CORPUS_DIR/anchor_anchors.txt \
		--sentence_anchor_weight_file $DATA_CORPUS_DIR/sentence_anchor_weight.txt \
		--output_dir $TASK_DATA_DIR \
		--do_lower_case --mlm --reduce_memory
# gen data for task QDM
python gendata/gen_pairdata_qdm.py \
		--passage_file $DATA_CORPUS_DIR/anchor_passages.txt \
		--sentence_file $DATA_CORPUS_DIR/anchor_sentences.txt \
		--anchor_file $DATA_CORPUS_DIR/anchor_anchors.txt \
		--sap_file  $DATA_CORPUS_DIR/anchor_sap_triples.txt \
		--bert_model $BERT_MODEL_PATH \
		--max_pair_perquery 10 --max_seq_len 512 --output_dir $TASK_DATA_DIR   --mlm \
		--sentence_anchor_weight_file $DATA_CORPUS_DIR/sentence_anchor_weight.txt \
		--stop_words_file $DATA_CORPUS_DIR/stopwords.txt
# gen data for task RDP
python gendata/gen_pairdata_rdp.py --bert_model  $BERT_MODEL_PATH \
    --do_lower_case  --output_dir $TASK_DATA_DIR \
    --mlm  --reduce_memory --stop_words_file $DATA_CORPUS_DIR/stopwords.txt \
    --anchor_file $DATA_CORPUS_DIR/anchor_anchors.txt --passage_file $DATA_CORPUS_DIR/anchor_passages.txt \
    --sentence_cls_weight_file $DATA_CORPUS_DIR/sentence_cls_weight.txt
# gen data for task ACM
python gendata/gen_pairdata_acm.py --bert_model  $BERT_MODEL_PATH \
	--do_lower_case  --output_dir $TASK_DATA_DIR  \
	--mlm  --reduce_memory  --stop_words_file $DATA_CORPUS_DIR/stopwords.txt \
	--anchor_file $DATA_CORPUS_DIR/anchor_anchors.txt --passage_file $DATA_CORPUS_DIR/anchor_passages.txt \
	--sentence_file_path  $DATA_CORPUS_DIR/sentence_passages.txt \
	--passage_cls_weight_file $DATA_CORPUS_DIR/passage_cls_weight.txt
# merge the data of four tasks, and shuffle.
mkdir -p $MERGED_DATA_DIR
python pretrain/shuffle_tasks.py --input_dir $TASK_DATA_DIR \
    --output_dir $MERGED_DATA_DIR --shuffle
