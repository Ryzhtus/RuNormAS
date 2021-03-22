import torch
from torch.nn.utils.rnn import pad_sequence

class RuNormASDatasetForTokenClassification():
    def __init__(self, entities, normalization_endings, endings_set, tokenizer):
        self.entities = entities
        self.normalization_endings = normalization_endings
        self.endings = ['<PAD>', '<UNK>'] + list(endings_set)

        self.tag2idx = {tag: idx for idx, tag in enumerate(self.endings)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.endings)}

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, item):
        words = self.entities[item]
        words_endings = self.normalization_endings[item]

        word2ending = dict(zip(words, words_endings))

        tokens = []
        tokenized_endings = []

        for word in words:
            if word not in ('[CLS]', '[SEP]'):
                subtokens = self.tokenizer.tokenize(word)
                for i in range(len(subtokens)):
                    if word2ending[word] in self.endings:
                        tokenized_endings.append(word2ending[word])
                    else:
                        tokenized_endings.append('<UNK>')
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