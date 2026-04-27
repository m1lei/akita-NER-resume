class EntityExtractor:
    def __init__(self, id2label: dict):
        self.id2label = id2label

    def extract(self, tokens, pred_ids):
        results = []

        for token, pred_id in zip(tokens, pred_ids):
            label = self.id2label[pred_id]

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if label != "O":
                results.append((token, label))

        return results