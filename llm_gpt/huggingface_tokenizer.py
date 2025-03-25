from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, StripAccents

import regex as re

class tokenizer_huggingface():
    def __init__(self):
        file_path = 'C:/Code_Projects/llm/datasets/corpora/gutenberg/'
        self.files = [
            f'{file_path}austen-sense.txt',
            f'{file_path}blake-poems.txt',
            f'{file_path}austen-persuasion.txt',
            f'{file_path}austen-emma.txt',
            f'{file_path}bryant-stories.txt',
            f'{file_path}burgess-busterbrown.txt',
            f'{file_path}bismarck.txt',
            f'{file_path}carroll-alice.txt',
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
            f'{file_path}whitman-leaves.txt'
            ]
        self.replaces = {
            '<unk>':[''],
            '<filtered>':[
                "ass",
                "asshole",
                "bastard",
                "bitch",
                "bollocks",
                "bullshit",
                "crap",
                "cunt",
                "dick",
                "douche",
                "fag",
                "fuck",
                "hell",
                "motherfucker",
                "nigga",
                "nigger",
                "piss",
                "prick",
                "pussy",
                "shit",
                "slut",
                "twat",
                "whore",
                "anal",
                "anus",
                "bdsm",
                "bi",
                "bisexual",
                "blowjob",
                "boner",
                "boobs",
                "bra",
                "breasts",
                "clit",
                "clitoris",
                "cock",
                "condom",
                "cum",
                "dildo",
                "ejaculation",
                "erection",
                "faggot",
                "fellatio",
                "fetish",
                "gay",
                "handjob",
                "homo",
                "homosexual",
                "incest",
                "jerkoff",
                "labia",
                "lesbian",
                "masturbate",
                "masturbation",
                "orgasm",
                "penis",
                "porn",
                "porno",
                "pornography",
                "prostate",
                "queer",
                "rectum",
                "rimjob",
                "semen",
                "sex",
                "sexual",
                "sodomite",
                "sodomy",
                "spunk",
                "testicle",
                "threesome",
                "tit",
                "tits",
                "vagina",
                "vulva",
                "wank",
                "xxx"
                ]
            }
        
        # Define the tokenizer
        self.tokenizer = Tokenizer(models.BPE())
        self.train()

    def train(self):
        print("Training")
        # Normalizer: Normalize the text
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
        
        # Pre-tokenizer: Split text into words
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Trainer: Train the tokenizer
        trainer = trainers.BpeTrainer(vocab_size=100000, special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<user>", "<bot>", "<startoftext>", "<endoftext>", "<filtered>"])
        
        # Train the tokenizer on your corpus
        self.tokenizer.train(self.files, trainer)
        
        # Set the post-processor
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>")),
            ],
        )

        # Save the tokenizer to disk
        self.tokenizer.save("tokenizer.json")
        print("Training complete")
    
    def encode(self, text):
        # Load the tokenizer
        tokenizer = Tokenizer.from_file("tokenizer.json")

        # Encode text
        encoded = tokenizer.encode(text)
        # print("Encoded:", encoded.ids)
        return encoded.ids
    
    def decode(self, text):
        # Load the tokenizer
        tokenizer = Tokenizer.from_file("tokenizer.json")

        # Decode text
        decoded = tokenizer.decode(text)
        # print("Decoded:", decoded)
        return decoded
    
    def tokenize(self, text):
        text = self.decode(self.encode(text))
        
        pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in self.replaces.get('<filtered>')) + r')\b', re.IGNORECASE)
        text = pattern.sub('<filtered>', text)

        return list(text.split(' '))