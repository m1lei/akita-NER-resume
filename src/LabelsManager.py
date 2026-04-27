import json


class LabelsManager:
    def __init__(self, data: list[dict]):
        """

        :param data: jsonl datasets с данными labels
        :labels list: существущие labels
        :label2id dict: из labels в математическое значение, кодировка
        :id2label dict: из математического значения в labels, расшифровка
        """
        self.data = data
        self.labels = []
        self.label2id = {}
        self.id2label = {}

    def build_labels_list(self) -> list:
        """
        Из data(datasets с labels) вынимает labels
        :return: список меток(labels) из датасета
        """
        tags = set()

        for item in self.data:
            for _,_, tag in item['label']:
                tags.add(tag)
        self.labels = ['O']
        for tag in sorted(tags):
            self.labels.append(f"B-{tag}")
            self.labels.append(f"I-{tag}")
        return self.labels

    def build_convert_labels(self):
        """
        на основе self.labels: list создает два массива, первый для кодировки в математический язык
         второй для расшифровки из математического
        :return: dict, dict
        """
        for i ,label in enumerate(self.labels):
            self.label2id[label] = i
            self.id2label[i] = label
        return self.label2id, self.id2label

    def save_labels(self, labels_file: str) -> None:
        """
        Сохраняет файл с labels
        :param labels_file: навзание файла для сохранения
        :return:
        """
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "labels": self.labels,
                    "label2id": self.label2id,
                    "id2label": self.id2label
                },
                f,
                ensure_ascii=False,
                indent=2,  # красивый отступ в 2 пробела
            )

    def run(self, labels_file: str):
        self.build_labels_list()
        self.build_convert_labels()
        self.save_labels(labels_file)
        return self.labels, self.label2id, self.id2label