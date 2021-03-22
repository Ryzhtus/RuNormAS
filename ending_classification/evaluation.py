import os
import tqdm
from nltk.stem.snowball import SnowballStemmer
from natasha import Segmenter, Doc
from ending_classification.reader import is_abbreviation, clean_entity
import torch

def read_test(text_filename, annotation_filename):
    text = open(text_filename, 'r', encoding='utf-8').read()
    annotation = open(annotation_filename, 'r', encoding='utf-8').read().strip().split('\n')

    segmenter = Segmenter()

    doc_text = Doc(text)
    doc_text.segment(segmenter)

    entities = []
    number_of_entities_in_annotation = 0

    for line in annotation:
        spans = list(map(int, line.strip().split()))
        entry = ''

        while spans:
            start, stop = spans[0], spans[1]
            entry = text[start: stop] + ' '
            spans = spans[2:]

        entry = entry.strip()
        entry = clean_entity(entry)

        doc = Doc(entry)
        doc.segment(segmenter)

        entities += [doc.tokens]

    entities_stem = []

    for entity in entities:
        stems_and_endings = []
        for token in entity:
            if is_abbreviation(token.text):
                stem = token.text
                ending = ''
            else:
                stem = get_stem(token.text)
                ending = token.text.replace(stem, '')

            stems_and_endings.append(stem)
            stems_and_endings.append(ending)
        entities_stem.append(stems_and_endings)

    return entities_stem

def get_stem(word):
    stemmer = SnowballStemmer('russian')
    if is_abbreviation(word):
        return word
    else:
        word_stem = stemmer.stem(word)
        if word[0].isupper():
            word_stem = word_stem.capitalize()

        return word_stem

def markup_entity(model, tokenizer, reader, idx2tag, entity, device, preprocessing=False):
    if preprocessing:
        if is_abbreviation(entity):
            stem = entity
            ending = ''
        else:
            stem = reader.get_stem(entity)
            ending = entity.replace(stem, '')
        preprocessed_entity = [stem, ending]
    else:
        preprocessed_entity = entity

    result = []
    while preprocessed_entity:
        stem, ending = preprocessed_entity[0], preprocessed_entity[1]
        if ending == '':
            result.append(stem)
        else:
            current_entity = [stem, ending]
            tokenized_entity = []
            for part in current_entity:
                subtokens = tokenizer.tokenize(part)
                tokenized_entity.extend(subtokens)

            tokens = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenized_entity))
            preds = model(tokens.unsqueeze(0).to(device))
            preds = preds.view(-1, preds.shape[-1])
            preds = preds.argmax(1)
            preds = preds.cpu().numpy()
            preds = [idx2tag[tag] for tag in preds]

            preds = [ending for ending in preds if ending != '<NO>' and ending != '<UNK>']
            if len(preds) != 0:
                end = []
                for ending in preds:
                    if ending not in end:
                        end.append(ending)
                end = ''.join(end)
            else:
                end = ''
            answer = ''

            for token in tokenized_entity:
                if answer == stem:
                    answer += end
                    break
                else:
                    answer += token.replace('##', '')

            result.append(answer)
        preprocessed_entity = preprocessed_entity[2:]

    output = ' '.join(result)
    return output


def evaluate_test(model, tokenizer, reader, idx2tag, entity, device):
    filenames = []
    for _, _, files in os.walk("../data/test_new/named"):
        for filename in sorted(files):
            filenames.append(filename.split('.')[0])

    filenames = sorted(set(filenames), reverse=True)

    for filename in tqdm.tqdm(filenames, total=len(filenames)):
        text_file = "../data/test_new/named/" + filename + ".txt"
        ann_file = "../data/test_new/named/" + filename + ".ann"
        output_file = "named/" + filename + '.norm'

        outputs = []
        entities = read_test(text_file, ann_file)

        for entity in entities:
            outputs.append(markup_entity(model, tokenizer, reader, idx2tag, entity, device))

        with open(output_file, 'w') as file:
            for out in outputs:
                file.write(out)
                file.write('\n')