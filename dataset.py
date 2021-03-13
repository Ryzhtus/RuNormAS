import re
from deprecated_code.reader_old import collect_sentences
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from nltk.stem.snowball import SnowballStemmer

class RuNormASDataset():
    def __init__(self, sentences, normalized_sentences, tokenizer):
        self.sentences = sentences
        self.normalized_sentences = normalized_sentences
        self.endings = []

        self.stemmer = SnowballStemmer('russian')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        sentence_normalized = self.normalized_sentences[item]

        tokens = []
        normalized_tokens = []

        # чистка слов от пунктуации
        for word in range(len(sentence)):
            sentence[word] = re.sub(r'[^\w\s]', '', sentence[word])

        for word in range(len(sentence_normalized)):
            sentence_normalized[word] = re.sub(r'[^\w\s]', '', sentence_normalized[word])

        for word in sentence:
            if word not in ('[CLS]', '[SEP]'):
                subtokens = self.tokenizer.tokenize(word)
                tokens.extend(subtokens)

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        for word in sentence_normalized:
            if word not in ('[CLS]', '[SEP]'):
                subtokens = self.tokenizer.tokenize(word)
                normalized_tokens.extend(subtokens)

        normalized_tokens = ['[CLS]'] + normalized_tokens + ['[SEP]']
        normalized_tokens_ids = self.tokenizer.convert_tokens_to_ids(normalized_tokens)

        return torch.LongTensor(tokens_ids), torch.LongTensor(normalized_tokens_ids)
    
    def paddings(self, batch):
        tokens, normalized_tokens = list(zip(*batch))
    
        tokens = pad_sequence(tokens, batch_first=True)
        normalized_tokens = pad_sequence(normalized_tokens, batch_first=True)
    
        return tokens, normalized_tokens


class RuNormASDatasetForTokenClassification():
    def __init__(self, sentences, sentences_endings, endings_set, tokenizer):
        self.sentences = sentences
        self.sentences_endings = sentences_endings
        self.endings = ['<PAD>'] + list(endings_set)

        self.tag2idx = {tag: idx for idx, tag in enumerate(self.endings)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.endings)}

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        words = self.sentences[item]
        words_endings = self.sentences_endings[item]

        word2ending = dict(zip(words, words_endings))

        tokens = []
        tokenized_endings = []

        for word in words:
            if word not in ('[CLS]', '[SEP]'):
                subtokens = self.tokenizer.tokenize(word)
                for i in range(len(subtokens)):
                    if i != len(subtokens) - 1:
                        tokenized_endings.append('<NO>')
                    else:
                        tokenized_endings.append(word2ending[word])
                tokens.extend(subtokens)

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tokenized_endings = ['<NO>'] + tokenized_endings + ['<NO>']
        endings_ids = [self.tag2idx[tag] for tag in tokenized_endings]

        return torch.LongTensor(tokens_ids), torch.LongTensor(endings_ids)

    def paddings(self, batch):
        tokens, endings = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True)
        endings = pad_sequence(endings, batch_first=True)

        return tokens, endings
