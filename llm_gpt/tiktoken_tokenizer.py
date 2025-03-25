import tiktoken
import regex as re

class TTK_Tokenizer():
    def __init__(self):
        self.enc = tiktoken.get_encoding('cl100k_base')
        self.regex = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        print(f'Tokenizer ready')

    def tokenize(self, text):
        rgx = re.findall(self.regex, text) #regex get words and break things like 12d124d1d4g3g
        # data = ' '. join(rgx)
        # print(f'{rgx}')
        return rgx
    
    def decode(self, text):
        return self.enc.decode(text)
    
    def encode(self, text):
        return self.enc.encode(text)