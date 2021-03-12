"""
while all(x in sentence for x in entities_copy[0].split(' ')):
    entity = entities_copy.pop(0)
    norm_entity = entity2norm[entity].split(' ')
    entity = entity.split(' ')

    for idx in range(len(entity)):
        sentence_word_id = sentence.index(entity[idx])
        stem = self.get_stem(sentence[sentence_word_id])
        ending = self.find_ending(norm_entity[idx], normalization=True)

        sentence[sentence_word_id] = stem
        sentence_endings[sentence_word_id] = ending

sentences_endings.append(sentence_endings)"""