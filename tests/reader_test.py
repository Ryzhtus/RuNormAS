import os
import unittest
from reader import RuNormASReader

class ReaderTestCase(unittest.TestCase):
    def test_equal_lengths(self):
        filenames = []
        reader = RuNormASReader()

        for _, _, files in os.walk("data/train/named/texts_and_ann"):
            for filename in files:
                filenames.append(filename.split('.')[0])

        filenames = set(filenames)

        for filename in filenames:
            print(filename)
            text_filename = "data/train/named/texts_and_ann/" + filename + ".txt"
            annotation_filename = "data/train/named/texts_and_ann/" + filename + ".ann"
            normalization_filename = "data/train/named/norm/" + filename + ".norm"

            _, entities, normalized_entities = reader.read(text_filename, annotation_filename, normalization_filename)
            self.assertEqual(len(entities), len(normalized_entities))

if __name__ == '__main__':
    unittest.main()
