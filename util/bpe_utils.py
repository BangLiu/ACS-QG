import json
import re
import ftfy
import spacy
from tqdm import tqdm


class BPEEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load(
            'en',
            disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def get_pairs(self, word):
        """
        Return set of symbol pairs in a word.
        word is represented as tuple of symbols
        (symbols being variable-length strings)
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def text_standardize(self, text):
        """
        fixes some issues the spacy tokenizer had on books corpus
        also does some whitespace standardization
        """
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = text.replace('―', '-')
        text = text.replace('…', '...')
        text = text.replace('´', "'")
        text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
        text = re.sub(r'\s*\n\s*', ' \n ', text)
        text = re.sub(r'[^\S\n]+', ' ', text)
        return text.strip()

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = self.get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and \
                        i < len(word) - 1 and \
                        word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts, verbose=True):
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                text = self.nlp(self.text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend(
                        [self.encoder.get(t, 0) for t in
                         self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                text = self.nlp(self.text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend(
                        [self.encoder.get(t, 0) for t in
                         self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        return texts_tokens


def get_bpe_encoder(bpe_dict_path, bpe_vocab_path,
                    specials=["<PAD>", "<OOV>", "<SOS>", "<EOS>"]):
    """
    Given BPE encoder file paths, get BPE encoder.
    """
    bpe_encoder = BPEEncoder(bpe_dict_path, bpe_vocab_path)
    for s in specials:
        bpe_encoder.encoder[s] = len(bpe_encoder.encoder)
    return bpe_encoder


def spacy_doc2bpe_id(spacy_doc, bpe_encoder):
    bpe_ids = []
    for token in spacy_doc:
        bpe_ids.append(bpe_encoder.encode([token.text])[0])
    return bpe_ids


if __name__ == "__main__":
    # BPE from openai transformer
    bpe_dict_path = '../../../../datasets/original/OpenAITransformer/encoder_bpe_40000.json'
    bpe_vocab_path = '../../../../datasets/original/OpenAITransformer/vocab_40000.bpe'
    bpe_encoder = get_bpe_encoder(bpe_dict_path, bpe_vocab_path)
    text = "Apple is (machine) 3 three apple.com bang3@ualberta.ca dollar"
    import spacy
    NLP = spacy.load("en")
    spacy_doc = NLP(text)
    bpe_ids = spacy_doc2bpe_id(spacy_doc, bpe_encoder)
    print(bpe_ids)

    # BPE from https://github.com/bheinzerling/bpemb#how-to-use-bpemb
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    bpe_model_path = "../../../../datasets/original/BPE/en.wiki.bpe.op50000.model"
    bpe_emb_path = "../../../../datasets/original/BPE/en.wiki.bpe.op50000.d100.w2v.txt"
    sp.Load(bpe_model_path)
    print(sp.EncodeAsPieces(text))
