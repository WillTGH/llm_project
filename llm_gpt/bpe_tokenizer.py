from tokenizers import Tokenizer
from tokenizers.normalizers import (Sequence, Lowercase, NFD, StripAccents)
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder

from tokenizers.processors import TemplateProcessing

from tokenizers.trainers import BpeTrainer

import nltk
from nltk.data import find
from nltk.corpus import gutenberg

import regex as re

nltk.download('gutenberg', download_dir='V:/llm-project/datasets')
nltk.download('punkt')

file_path = 'V:/llm-project/datasets/corpora/gutenberg/'
try:
    find('V:/llm-project/datasets/corpora/')
    print('Corpora Gutenberg is There')
except LookupError:
    print('Corpora Gutenberg is Not There')

# vocab_size = 1000000

class BPETokenizer():
    def __init__(self, vocab_size, text = None):
        self.plays = [
    f'{file_path}austen-sense.txt',
    f'{file_path}blake-poems.txt',
    f'{file_path}austen-persuasion.txt',
    f'{file_path}austen-emma.txt',
    f'{file_path}bryant-stories.txt',
    f'{file_path}burgess-busterbrown.txt',
    f'{file_path}bismarck.txt',
    f'{file_path}carroll-alice.txt',
    f'{file_path}chesterton-ball.txt',
    f'{file_path}chesterton-brown.txt',
    f'{file_path}chesterton-thursday.txt',
    f'{file_path}corpus1.txt',
    f'{file_path}corpus2.txt',
    f'{file_path}corpus3.txt',
    f'{file_path}corpus4.txt',
    f'{file_path}corpus5.txt',
    f'{file_path}edgeworth-parents.txt',
    f'{file_path}melville-moby_dick.txt',
    f'{file_path}milton-paradise.txt',
    f'{file_path}shakespeare-macbeth.txt',
    f'{file_path}shakespeare-hamlet.txt',
    f'{file_path}shakespeare-caesar.txt',
    f'{file_path}whitman-leaves.txt'
    ]
        self.shakespeare = [" ".join(s) for ply in self.plays for s in gutenberg.sents(ply)]

        self.special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"]
        self.temp_proc = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.special_tokens.index("[CLS]")),
                ("[SEP]", self.special_tokens.index("[SEP]")),
            ],
        )
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([NFD(),Lowercase(),StripAccents()])
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.decoder = BPEDecoder()
        self.tokenizer.post_processor=self.temp_proc

        self.tokenizer_train()
        # print(len(self.shakespeare))
        # print(self.shakespeare[100])

    def tokenizer_train(self):
        trainer = BpeTrainer(vocab_size=self.vocab_size,special_tokens=self.special_tokens)
        self.tokenizer.train_from_iterator(self.shakespeare, trainer=trainer)

        print(f"Trained vocab size: {self.tokenizer.get_vocab_size()}")

    def tokenize(self, text):
        #text = "in the village churches the medals won at Waterloo were hung up by those of Grossbehren and Leipzig."
        regex = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        rgx = re.findall(regex, text) #regex get words and break things like 12d124d1d4g3g
        text = " " . join(rgx)
        sen_enc = self.encode(text)
        # print(f"token: {sen_enc.tokens}")
        return sen_enc.tokens
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, text):
        return self.tokenizer.decode(text)