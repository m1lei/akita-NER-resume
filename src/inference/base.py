from abc import ABC, abstractmethod
from typing import Tuple, List

class BasePredictor(ABC):
    """Абстрактный интерфейс для NER predictor(предсказателя)"""
    @abstractmethod
    def predict(self, text: str) -> Tuple[list[str],list[int]]:
        """Предсказывает по сырому тексту"""
        pass

    @abstractmethod
    def predict_from_ids(self, input_ids: List[int], attention_mask: List[int]) -> List[int]:
        """Предсказывает по числовым ID(ids)"""
        pass