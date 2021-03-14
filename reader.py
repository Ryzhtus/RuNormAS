import os


class RuNormASReader():
    def read(self, text_filename, annotation_filename, normalization_filename):

        document = open(text_filename, 'r').read()
        annotations = open(annotation_filename, 'r').read().split('\n')

        entities_intervals = []
        for interval in annotations:
            if interval != '':
                entity = interval.split(' ')
                entity[0], entity[1] = int(entity[0]), int(entity[1])
                entities_intervals.append(entity)

        normalized_entities = open(normalization_filename, 'r').read().split('\n')
        normalized_entities.pop()

        entities = []
        for entity_interval in entities_intervals:
            entities.append(document[entity_interval[0] : entity_interval[1]])

        entity2norm = {}
        for idx in range(len(entities)):
            entity2norm[entities[idx]] = normalized_entities[idx]

        entities_intervals = sorted(entities_intervals, key=lambda x: x[0])

        entities = []
        for entity_interval in entities_intervals:
            entities.append(document[entity_interval[0]: entity_interval[1]])

        normalized_entities = []
        for entity in entities:
            normalized_entities.append(entity2norm[entity])

        entities_copy = entities.copy()
        normalized_entities_copy = normalized_entities.copy()

        while entities_copy:
            entity = entities_copy.pop(0)
            norm = normalized_entities_copy.pop(0)
            document = document.replace(entity, norm, 1)

        sentences = open(text_filename, 'r').read().split('\u2028')
        # document - это normalized document
        document = document.split('\u2028')

        sentences = [sentence.split() for sentence in sentences]
        normalized_sentences = [sentence.split() for sentence in document]
        return sentences, normalized_sentences


def collect_sentences():
    reader = RuNormASReader()

    sentences = []
    normalized_sentences = []

    filenames = []
    for _, _, files in os.walk("data/train/named/texts_and_ann"):
        for filename in sorted(files):
            filenames.append(filename.split('.')[0])

    filenames = sorted(set(filenames), reverse=True)

    for filename in filenames:
        text_filename = "data/train/named/texts_and_ann/" + filename + ".txt"
        annotation_filename = "data/train/named/texts_and_ann/" + filename + ".ann"
        normalization_filename = "data/train/named/norm/" + filename + ".norm"
        text, normalized_text = reader.read(text_filename, annotation_filename, normalization_filename)

        sentences += text
        normalized_sentences += normalized_text

    return sentences, normalized_sentences


if __name__ == '__main__':
    # sentences, sentences_tags, normalized_sentences, normalized_sentences_tags = collect_sentences()
    """text_filename = "data/train/named/texts_and_ann/1009448.txt"
    annotation_filename = "data/train/named/texts_and_ann/1009448.ann"
    normalization_filename = "data/train/named/norm/1009448.norm"
    reader = RuNormASReader()
    sentences, normalization, entities, normalized_entities = reader.read(text_filename, annotation_filename, normalization_filename)
    print(sentences)
    print(normalization)
    print(entities)
    print(normalized_entities)"""

    sentences, normalized_sentences = collect_sentences()
    for i in range(100):
        print(sentences[i])
        print(normalized_sentences[i])
        print('-' * 75)

    print(len(sentences), len(normalized_sentences))

    collect_sentences()