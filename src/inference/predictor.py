import torch
from typing import List
from src.inference.base import BasePredictor


class Predictor(BasePredictor):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def predict(self, text: str):
        """
        Принимает сырой текст, токенизирует его, отдает токены и предсказания
        :param text:
        :return:
        """
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

    def predict_from_ids(self, input_ids: List[int], attention_mask: List[int]) -> List[int]:
        """
        Принимает уже готовые цифры из датасета, делает инференс и возвращает ТОЛЬКО предсказанные ID.
        """
        inputs = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask])
        }

        # Переносим на GPU/CPU туда же, где модель
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = outputs.logits.argmax(dim=-1)
        return predictions[0].tolist()