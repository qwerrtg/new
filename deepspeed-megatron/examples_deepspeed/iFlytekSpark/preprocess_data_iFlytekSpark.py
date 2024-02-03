"""Processing data for pretraining iFlytekSpark."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import glob
import time
import argparse
import multiprocessing

import torch

from megatron.data import indexed_dataset
from megatron.tokenizer.iFlytekSpark_tokenization import iFlytekSparkSPTokenizer


class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = iFlytekSparkSPTokenizer(self.args.tokenizer)

    def encode(self, iterator):
        key = self.args.json_keys[0]
        len_paras = 0
        ids = {}
        doc_ids = []
        
        encode_start_time = time.time()
        for file_path in iterator:
            print(file_path)
            each_start_time = time.time()
            json_line = open(file_path, 'r', encoding='utf-8')
            strr = json_line.read()
            lista = strr.split('\n\n')
            len_paras += len(lista)
            for para in lista:
                if para:
                    contenta = self.tokenizer.tokenize(para)
                    para_ids = self.tokenizer.convert_tokens_to_ids(contenta)
                    if len(para_ids) > 0:
                        doc_ids.append(para_ids)
                        if self.args.append_eod:
                            for i in range(self.args.eod_num):
                                doc_ids[-1].append(self.tokenizer.eod_id)
            each_end_time = time.time()
            print("encode this file using {}s".format(each_end_time - each_start_time))
        ids[key] = doc_ids
        encode_end_time = time.time()
        print("FINISHING ENCODING, USING {}s".format(encode_end_time - encode_start_time))
        
        return ids, len_paras


def package_file(it, n):
    """ package multiple files"""
    stop = False
    while not stop:
        batch = []
        for _ in range(n):
            try:
                batch.append(next(it))
            except StopIteration:
                stop = True
        if not batch:
            break
        yield batch


def get_args():
    parser = argparse.ArgumentParser()
    
    # Input Data Config
    group = parser.add_argument_group(title='input data')
    group.add_argument('--raw_data_path',
        type=str,
        default='/raid/gpt3-train-data/data-v1/new2016zh/txt-data/train/0000*.txt',
        help='Path to input txt'
    )
    group.add_argument('--json-keys',
        nargs='+',
        default=['text'],
        help='space separate listed of keys to extract from json'
    )
    group.add_argument('--split-sentences',
        action='store_true',
        help='Split documents into sentences.'
    )

    # Tokenizer Config
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer',
        type=str,
        default='megatron/tokenizer/bpe_4w_pcl/vocab',
        help='Path to the tokenizer file'
    )
    group.add_argument('--append-eod',
        action='store_true',
        help='Append an <eod> token to the end of a document.'
    )
    group.add_argument('--eod-num',
        type=int,
        default=1,
        help='eot number.'
    )

    # Output Config
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output_filepath',
        type=str,
        required=True,
        help='Path to binary output file without suffix'
    )
    group.add_argument('--dataset-impl',
        type=str,
        default='mmap',
        choices=['lazy', 'cached', 'mmap']
    )

    # Runtime Config
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers',
        type=int,
        default=200,
        help='Number of worker processes to launch'
    )
    group.add_argument('--log-interval',
        type=int,
        default=1,
        help='Interval between progress updates'
    )

    args = parser.parse_args()
    args.keep_empty = False
    
    return args


def main():
    
    args = get_args()
    startup_start = time.time()

    os.makedirs(args.output_filepath, exist_ok=True)

    print("Opening", args.raw_data_path)
    file_iter = glob.iglob(args.raw_data_path)
    
    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers)
    encoded_docs = pool.imap(encoder.encode, package_file(file_iter, 128))
    print('encoded_docs', encoded_docs)

    print(f"Vocab size: {encoder.tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_filepath}")
    level = "document"
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}{}_{}.bin".format(args.output_filepath, key, level)
        output_idx_files[key] = "{}{}_{}.idx".format(args.output_filepath, key, level)
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=encoder.tokenizer.vocab_size
        )

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
        
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i} documents", f"({i/elapsed} docs/s, {mbs} MB/s).", file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])
    
    end_time = time.time()
    print('Preprocess data using {}s'.format(end_time - startup_end))


if __name__ == '__main__': 
    main()
