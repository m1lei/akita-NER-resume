import json
from Loader import Loader
from predictor import Predictor
from Extractor import EntityExtractor

class NERInferencePipeline:
    def __init__(self, model_path: str, labels_path: str):
        self.model_path = model_path
        self.labels_path = labels_path

        self.tokenizer = None
        self.model = None
        self.id2label = None

    def load(self):
        loader = Loader(self.model_path)
        self.tokenizer, self.model = loader.load()

        with open(self.labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        self.id2label = {
            int(k): v for k, v in labels["id2label"].items()
        }

    def predict(self, text: str):
        predictor = Predictor(self.model, self.tokenizer)
        tokens, pred_ids = predictor.predict(text)

        extractor = EntityExtractor(self.id2label)
        return extractor.extract(tokens, pred_ids)