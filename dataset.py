import re
from reader import collect_sentences
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

class RuNormASDataset():
    def __init__(self, sentences, entities, normalized_sentences, normalized_entities, tokenizer):
        self.sentences = sentences
        self.entities = entities
        self.normalized_sentences = normalized_sentences
        self.normalized_entities = normalized_entities
        self.endings = []

        self.tokenizer = tokenizer

        self.idx2tag = {0: '<PAD>', 1: 'ENTITY', 2: 'O'}
        self.tag2idx = {'<PAD>': 0, 'ENTITY': 1, 'O': 2}

    def __getitem__(self, item):
        sentence = self.sentences[item]
        sentence_entities = self.entities[item]
        sentence_normalized = self.normalized_sentences[item]
        sentence_normalized_entities = self.normalized_entities[item]

        tokens = []
        tags = []
        normalized_tokens = []
        normalized_tags = []

        # чистка слов от пунктуации
        for word in range(len(sentence)):
            sentence[word] = re.sub(r'[^\w\s]', '', sentence[word])

        for word in range(len(sentence_normalized)):
            sentence_normalized[word] = re.sub(r'[^\w\s]', '', sentence_normalized[word])

        word2tag = dict(zip(sentence, sentence_entities))
        word2tag_normalized = dict(zip(sentence_normalized, sentence_normalized_entities))

        # это фигня, которая считает окончания, ее лучше пока не трогать, так как работает не на 100%
        """  
        if len(sentence) == len(sentence_normalized):
            for word in range(len(sentence)):
                if (sentence_entities[word] == 'ENTITY') or (sentence_normalized_entities[word] == 'ENTITY'):
                    if sentence[word] != sentence_normalized[word]:
                        entity1, ending1, entity2, ending2 = self.find_endings(sentence[word], sentence_normalized[word])
                        self.endings.append(ending1)
                        self.endings.append(ending2)
                        print(entity1, '-> ', ending1, '|', entity2, '->', ending2)
        """

        for word in sentence:
            if word not in ('[CLS]', '[SEP]'):
                subtokens = self.tokenizer.tokenize(word)
                for i in range(len(subtokens)):
                    tags.append(word2tag[word])
                tokens.extend(subtokens)
    
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    
        tags = ['O'] + tags + ['O']
        tags_ids = [self.tag2idx[tag] for tag in tags]

        for word in sentence_normalized:
            if word not in ('[CLS]', '[SEP]'):
                subtokens = self.tokenizer.tokenize(word)
                for i in range(len(subtokens)):
                    normalized_tags.append(word2tag_normalized[word])
                normalized_tokens.extend(subtokens)

        normalized_tokens = ['[CLS]'] + tokens + ['[SEP]']
        normalized_tokens_ids = self.tokenizer.convert_tokens_to_ids(normalized_tokens)

        normalized_tags = ['O'] + normalized_tags + ['O']
        normalized_tags_ids = [self.tag2idx[tag] for tag in normalized_tags]

        return torch.LongTensor(tokens_ids), torch.LongTensor(tags_ids), torch.LongTensor(normalized_tokens_ids), torch.LongTensor(normalized_tags_ids)
    
    def paddings(self, batch):
        tokens, tags, normalized_tokens, normalized_tags = list(zip(*batch))
    
        tokens = pad_sequence(tokens, batch_first=True)
        tags = pad_sequence(tags, batch_first=True)
        normalized_tokens = pad_sequence(normalized_tokens, batch_first=True)
        normalized_tags = pad_sequence(normalized_tags, batch_first=True)
    
        return tokens, tags, normalized_tokens, normalized_tags

    def find_endings(self, string1, string2):
        for letter in range(len(string1)):
            try:
                # print(string1[letter])
                if string1[letter] != string2[letter]:
                    return string1[:letter], string1[letter:], string2[:letter], string2[letter:]
            except IndexError:
                return string1[:letter], string1[letter:], string2[:letter], string2[letter:]


if __name__ == '__main__':
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_basic_tokenize=False, do_lower_case=False)
    sentences, sentences_tags, normalized_sentences, normalized_sentences_tags = collect_sentences()
    dataset = RuNormASDataset(sentences, sentences_tags, normalized_sentences, normalized_sentences_tags, TOKENIZER)
