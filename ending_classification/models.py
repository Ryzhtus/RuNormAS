from transformers import BertModel
from transformers import XLMRobertaModel
import torch.nn as nn


class BertEndingClassificator(nn.Module):
    def __init__(self, num_classes, pretrained='bert-base-multilingual-cased'):
        super(BertEndingClassificator, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained(pretrained, output_attentions=True)
        self.linear = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, tokens):
        embeddings = self.bert(tokens)[0]
        predictions = self.linear(embeddings)

        return predictions


class RuBertEndingClassificator(nn.Module):
    def __init__(self, num_classes, pretrained='DeepPavlov/rubert-base-cased'):
        super(RuBertEndingClassificator, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained(pretrained, output_attentions=True)
        self.linear = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, tokens):
        embeddings = self.bert(tokens)[0]
        predictions = self.linear(embeddings)

        return predictions


class BertNERBiLSTM(nn.Module):
    def __init__(self, num_classes, pretrained='bert-base-multilingual-cased'):
        super(BertNERBiLSTM, self).__init__()
        self.embedding_dim = 768
        self.hidden_dim = 768
        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained(pretrained, output_attentions=True)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True)
        self.linear = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, tokens):
        embeddings = self.bert(tokens)[0]
        lstm = self.lstm(embeddings)[0]
        predictions = self.linear(lstm)

        return predictions


class XLMRoBERTaEndingClassificator(nn.Module):
    def __init__(self, num_classes):
        super(XLMRoBERTaEndingClassificator, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes

        self.RoBERTa = XLMRobertaModel.from_pretrained("xlm-roberta-base", output_attentions=True)
        self.linear = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, tokens):
        embeddings = self.RoBERTa(tokens)[0]
        predictions = self.linear(embeddings)

        return predictions


class BertGRUEndingClassificator(nn.Module):
    def __init__(self, num_classes, pretrained='bert-base-multilingual-cased'):
        super(BertGRUEndingClassificator, self).__init__()
        self.embedding_dim = 768
        self.hidden_dim = 256
        self.dropout = 0.25
        self.n_layers = 2
        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained(pretrained, output_attentions=True)

        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.n_layers,
                          bidirectional=True, batch_first=True,
                          dropout=0 if self.n_layers < 2 else self.dropout)

        self.drop = nn.Dropout(self.dropout)

        self.linear = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, tokens):
        embeddings = self.bert(tokens)[0]
        output, _ = self.gru(embeddings)
        dropped = self.drop(output)
        predictions = self.linear(dropped)

        return predictions
