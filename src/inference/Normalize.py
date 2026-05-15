import re
from collections import defaultdict


class Normalize:
    def __init__(self, raw_entities):
        self.raw_entities = raw_entities

    def normalize(self):
        group = defaultdict(list)

        for item in self.raw_entities:
            word = item.get('word')
            label = item.get('label')

            word = re.sub(r'\s*-\s*', '-', word)  # fine - tuning -> fine-tuning
            word = re.sub(r'\s*\.\s*', '. ', word)  # им . Ломоносова -> им. Ломоносова
            word = re.sub(r'\s+', ' ', word).strip()  # схлопывание пробелов

            group[label].append(word)

        for i in group.items():
            print(i)