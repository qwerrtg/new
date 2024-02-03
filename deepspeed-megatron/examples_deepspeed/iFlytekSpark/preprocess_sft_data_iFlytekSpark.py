"""Processing data for pretraining iFlytekSpark."""
from typing import List, Sequence, Optional, Callable
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import re
import json
import glob
import time
import random
import argparse
import multiprocessing
from itertools import takewhile, repeat

import torch
import numpy as np
import sentencepiece as spm

from megatron.data import indexed_dataset


class Tokenizer(object):
    def __init__(self, vocab_file: str, split: bool=True):
        self.version: str = ''
        self.vocab_file: str = vocab_file
        self.split: bool = split
        self.encoder = dict()
        self.decoder = dict()
        self.sp: spm.SentencePieceProcessor = None
        self._init()

        self.sep_id: int = self.encoder['<s>']
        self.eod_id: int = self.encoder['<end>']
        self.pad_id: int = self.encoder['<pad>']
        self.unk_id: int = self.encoder["<unk>"]
        self.preprocess_processor: Optional[Callable] = None

    def register_preprocess_processor(self, processor: Callable) -> None:
        self.preprocess_processor = processor

    def _init(self):
        if os.path.isfile(self.vocab_file+".model"):
            init_fn = self._init_from_src
        else:
            raise ValueError(f'illegal vocab path: {self.vocab_file}')
        try:
            init_fn()
        except Exception as e:
            print(f"deserialize vocab failed: {e}")
            raise e

    def _init_from_src(self) -> None:
        model_file = self.vocab_file + ".model"
        vocab_file = self.vocab_file + ".vocab"
        assert os.path.exists(vocab_file), \
                f"vocab file path ({vocab_file}) is not exist"
        assert os.path.exists(model_file), \
                f"vocab model path ({model_file}) is not exist"

        with open(vocab_file, 'r') as fp:
            lines = fp.readlines()

        for token, line in enumerate(lines):
            text = line.split('\t')[0]
            self.encoder[text] = token
            self.decoder[token] = text

        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def encode(self, text: str) -> Sequence[int]:
        res = self.tokenize(text)
        return res

    def decode(self, tokens: Sequence[int]) -> str:
        text = self.sp.decode(tokens)
        return text

    def tokenize(self, text: str, split_tokenize: Optional[bool]=None) -> Sequence[int]:
        """ Tokenize a string. """
        if self.preprocess_processor is not None:
            text = self.preprocess_processor(text)
        if split_tokenize is None:
            split_tokenize = self.split
        if split_tokenize:
            return self.split_tokenize(text)
        return self.sp.encode(text)

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return self.decode(token_ids)

    def convert_tokens_to_ids(self, tokens: Sequence[int]) -> Sequence[int]:
        return tokens

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> Sequence[int]:
        return self.decode(ids)

    def split_tokenize(self, text: str) -> Sequence[int]:
        text_list = re.split(r'(<ret>|<end>|<s>)', text)
        tokens = []
        for each in text_list:
            if each == '<ret>':
                tokens.append(self.encoder['<ret>'])
            elif each == '<end>':
                tokens.append(self.encoder['<end>'])
            elif each == '<s>':
                tokens.append(self.encoder['<s>'])
            else:
                tokens += self.sp.encode(each)
        return tokens


class Sample(object):
    def __init__(self, text: str, tokenizer: Tokenizer) -> None:
        self.text: str = text
        self.tokenizer: Tokenizer = tokenizer
        self.vocab_size: int = int(self.tokenizer.vocab_size)
        
    def _preprocess(self) -> None:
        item = json.loads(self.text.strip())
        assert 'input' in item, f'miss key=input in {item}'
        assert 'target' in item, f'miss key=target in {item}'
        query = '<User> ' + item['input'].replace('<end>', '<end><User> ').replace('<s>', '<end><Bot> ') + '<end><Bot> '
        respo = item['target'] + '<end>'
        if query.endswith('\n'):
            query = query.strip('\n')
        query = query.replace('\\r\\n', '<ret>').replace('\\r\n', '<ret>').replace('\\n', '<ret>').replace('\n', '<ret>')
        respo = respo.lstrip('?\n\n').lstrip('?\n').strip()
        respo = respo.replace('\\r\\n', '<ret>').replace('\\r\n', '<ret>').replace('\\n', '<ret>').replace('\n', '<ret>')
        self.query_text = query
        self.respo_text = respo

    def _query_tokenize(self, text: str) -> List[int]:
        text_list = re.split(r'(<end><User> |<end><Bot> )', text)
        tokenizer_list = []
        add_vocab_size = True
        for each in text_list:
            if each == '<end><User> ':
                token_new = self.tokenizer.sp.encode(each)
                token_new = [token + self.vocab_size for token in token_new]
                token_new[0] -= self.vocab_size
                tokenizer_list += token_new
                add_vocab_size = True
            elif each == '<end><Bot> ':
                token_new = self.tokenizer.sp.encode(each)
                token_new = [token + self.vocab_size for token in token_new]
                tokenizer_list += token_new
                add_vocab_size = False
            elif add_vocab_size:
                token_new = self.tokenizer.sp.encode(each)
                tokenizer_list += [token + self.vocab_size for token in token_new]
            else:
                tokenizer_list += self.tokenizer.sp.encode(each)
        return tokenizer_list

    def _respo_tokenize(self, text: str) -> List[int]:
        text_list = re.split(r'(<ret>|<end>|<s>)', text)
        tokenizer_list = []
        for each in text_list:
            if each == '<ret>':
                tokenizer_list.append(self.tokenizer.encoder['<ret>'])
            elif each == '<end>':
                tokenizer_list.append(self.tokenizer.encoder['<end>'])
            elif each == '<s>':
                tokenizer_list.append(self.tokenizer.encoder['<s>'])
            else:
                tokenizer_list += self.tokenizer.sp.encode(each)
        return tokenizer_list
    
    def tokenize(self) -> List[int]:
        self._preprocess()
        query_tokens = self._query_tokenize(self.query_text)
        respo_tokens = self._respo_tokenize(self.respo_text)
        tokens = query_tokens + respo_tokens
        if random.random() < 0.7:
            tokens.append(self.vocab_size + self.tokenizer.encoder['</s>'])
        return np.array(tokens, dtype=np.int32)


class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.seq_len = self.args.seq_length
        self.tokenizer = Tokenizer(self.args.tokenizer)

    def encode(self, iterator):
        key = self.args.json_keys[0]
        len_paras = 0
        ids = {}
        doc_ids = []
        
        encode_start_time = time.time()
        for file_path in iterator:
            print(file_path)
            each_start_time = time.time()
            with open(file_path, "r") as fp:
                for line in takewhile(lambda x: x, (fp.readline() for _ in repeat(None))):
                    len_paras += 1
                    try:
                        tokens = Sample(line, self.tokenizer).tokenize()
                    except Exception as e:
                        print(f"Parse jsonl error: {e}, source line: {line}")
                        continue
                    if len(tokens) > self.seq_len + 1:
                        print(
                            f"Warning: the sample length exceeds the maximum length({self.seq_len}) and be ignored: {line}"
                        )
                        continue
                    doc_ids.append(tokens)
            
            print(f"Get num={len(doc_ids)} samples from {file_path}")
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
    group.add_argument('--seq_length',
        type=int,
        default=32768,
        help='sequence length.'
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
        # required=True,
        default="/data/dataset/sft-test/megatron/seq_length_2048_",
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
        default=1,
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


def main(args):
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
            vocab_size=encoder.tokenizer.vocab_size,
            dtype=np.int64
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
    args = get_args()
    main(args)
