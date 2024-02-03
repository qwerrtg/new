python ./preprocess_sft_data_iFlytekSpark.py \
--raw_data_path "/data/dataset/raw_sft_data/*.json" \
--output_filepath "/data/dataset/sft/seq_length_32768_" \
--tokenizer "/data/tokenizer/tokenizer" \
--seq_length 32768
--dataset-impl mmap \
--append-eod
