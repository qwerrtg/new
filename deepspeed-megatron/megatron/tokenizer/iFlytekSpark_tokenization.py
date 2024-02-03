import os
import re
from io import open
import sentencepiece as spm


class iFlytekSparkSPTokenizer(object):
    def __init__(self, vocab_file):
        model_file = vocab_file + ".model"
        vocab_file = vocab_file + ".vocab"
        assert os.path.exists(vocab_file), \
                f"vocab file path ({vocab_file}) is not exist"
        assert os.path.exists(model_file), \
                f"sentencepiece model path ({model_file}) is not exist"
        f = open(vocab_file,'r', encoding='utf-8')
        lines = f.readlines()
        self.encoder = {}
        for line in enumerate(lines):
            key = line[1].split('\t')[0]
            self.encoder[key] = line[0]

        self.decoder = {v:k for k,v in self.encoder.items()}

        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.ret_tok= "<ret>"

        self.eod_id = self.encoder['<end>']
        self.pad_id = self.encoder['<pad>']
        self.unk_id = self.encoder["<unk>"]
        
    
    @property
    def vocab_size(self):
        return len(self.encoder)

    def __len__(self):
        return len(self.encoder)
    
    def add_space(self, text):
        text=re.sub("(，|。|！|？) *",r"\1 ",text)
        return text

    @property
    def eod(self):
        return self.eod_id

    def tokenize(self, text):
        """ Tokenize a string. """
        text= self.add_space(text)
        text= text.replace("\n","<ret>")
        text = text.replace("\t", " "*4)
        # text= text.translate(self.translator)
        return self.sp.encode(text)

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def detokenize(self, token_ids):
        return self.decode(token_ids)

    def encode(self, text):
        res = self.tokenize(text)
        return res

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        return text


class iFlytekSparkSpaceTokenizer(object):
    def __init__(self, vocab_file):
        vocab_file = vocab_file + ".vocab"
        self.id2word=[]
        with open(vocab_file, 'r') as f:
            for line in f:
                parts= line.strip().split()
                if len(parts) < 2:
                    raise RuntimeError(f'bad line in vocab file:{line}')
                self.id2word.append(parts[0])
        self.word2id={w:idx for idx, w in enumerate(self.id2word)}
        self.eod_id= self.word2id['</s>']
        self.pad_id= self.word2id['<pad>']
        self.unk_word = "<unk>"
        self.unk_id= self.word2id[self.unk_word]
    
    @property
    def vocab_size(self):
        return len(self.id2word)

    def __len__(self):
        return len(self.id2word) 

    @property
    def eod(self):
        return self.eod_id
    
    @property
    def unk(self):
        return self.unk_id
    
    def encode(self,text):
        words = text.split()
        wids= [self.word2id.get(w,self.unk_id) for w in words]
        return wids
    
    def decode(self, tokens):
        words= [self.id2word[idx] for idx in tokens]
        text= " ".join(words)
        return text
    
    def tokenize(self, text):
        """ Tokenize a string. """
        return self.encode(text)

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def detokenize(self, token_ids):
        return self.decode(token_ids)
