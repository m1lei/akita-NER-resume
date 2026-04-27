import json

class DatasetLoader:
    def __init__(self, train: str, n:int = 40):
        self.train = train # файл обучения
        self.n = n#кол-во записей которые берем из jsonl file

    def load(self):
        """
        Конвентирует jsonl в dict
        :return: dict[{},{}]
        """
        with open(self.train, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return data[:self.n]