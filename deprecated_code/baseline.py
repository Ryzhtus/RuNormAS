from os import listdir, mkdir
from os.path import join, exists

from tqdm import tqdm

import shutil

import os

from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc,
)

from natasha.doc import DocSpan

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

if os.path.exists(f"../baseline"):
    shutil.rmtree(f"../baseline")

mkdir(f"../baseline")
mkdir(f"../baseline/named")

f = open(f"../baseline/named/927141.norm", 'w', encoding='utf-8')
text = open(f"../data/train_new/named/texts_and_ann/927141.txt", encoding='utf-8').read()
ann = open(f"../data/train_new/named/texts_and_ann/927141.ann", encoding='utf-8').read().strip().split('\n')

for line in ann:
    spans = list(map(int, line.strip().split()))
    entry = ''
    while spans:
        start, stop = spans[0], spans[1]
        entry += text[start:stop] + " "

        spans = spans[2:]

    entry = entry.strip()

    doc = Doc(entry)

    doc.segment(segmenter)

    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    found = False
    for s in doc.spans:
        if s.text == entry:
            span = s
            found = True
            break

    if not found:
        span = DocSpan(
            start=0
            , stop=len(entry)
            , type='ORG'
            , text=entry
            , tokens=[token for token in doc.tokens]
        )
    print(doc.spans)
    span.normalize(morph_vocab)

    f.write(f"{span.normal}\n")


f.close()