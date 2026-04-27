import torch


class Predictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )
        with torch.no_grad():#отключаем градиентное вычисления, для инференса это не требуется
            outputs = self.model(**inputs)

        predictions = outputs.logits.argmax(dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pred_ids = predictions[0].tolist()

        return tokens, pred_ids