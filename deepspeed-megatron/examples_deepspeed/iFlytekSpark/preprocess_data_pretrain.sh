python ./preprocess_data_iFlytekSpark.py \
--raw_data_path "/data/dataset/raw_pretrain_data/*.txt" \
--output_filepath "/data/dataset/pretrain/" \
--tokenizer "/data/tokenizer/tokenizer" \
--dataset-impl mmap \
--append-eod
