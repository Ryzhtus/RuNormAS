import re
from reader import collect_sentences
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

    def find_endings(self, word):
        word_stem = self.stemmer.stem(word)
        if word[0].isupper():
            word_stem = word_stem.capitalize()
        ending = word.replace(word_stem, '')
        print(word, word_stem, ending)

        if word != ending:
            self.endings.append(ending)

if __name__ == '__main__':
    TOKENIZER = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", do_lower_case=False)
    sentences, normalized_sentences = collect_sentences()
    dataset = RuNormASDataset(sentences, normalized_sentences, TOKENIZER)

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.paddings)
    print(next(iter(dataloader)))
    print(dataset[10])

    """
    for i in range(200, 500):
        print(i, dataset.sentences[i])
        print(i, dataset.normalized_sentences[i])
        print('-' * 75)
    """