from transformers import AutoTokenizer, AutoModelForTokenClassification
#загрузка модели
class Loader:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def load(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        model.eval()
        return tokenizer, model