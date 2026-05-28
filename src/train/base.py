from abc import ABC, abstractmethod

class BaseDataPreprocessor(ABC):
    """Абстрактный интерфейс для подготовки и токенизации NER-данных"""
    @abstractmethod
    def tokenize_and_align_labels(self, data: dict, tokenizer, label2id: dict, max_length: int = 128)-> dict:
        pass