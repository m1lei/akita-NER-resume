from src.evalute.EvaluationDataPreprocess import EvaluationDataPreprocess
from src.train.DatasetLoader import DatasetLoader
from src.train.DatasetPreprocessor import DatasetPreprocessor
from src.inference.predictor import Predictor
from src.inference.Loader import Loader
from src.evalute.NEREvaluation import NEREvaluation
import json
from transformers import AutoTokenizer
from config import Config

cfg = Config.from_yaml('config.yaml')


preprocessor = DatasetPreprocessor()
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
loader = DatasetLoader(cfg.path_for_evalution)
data = loader.load()
with open(cfg.labels_for_evaluation, 'r', encoding='utf-8') as f:
    # Загружаем данные в переменную
    labels = json.load(f)
loader = Loader(cfg.output_dir)
_, model = loader.load()
predict = Predictor(model, tokenizer)
evaluationDataPreprocess = EvaluationDataPreprocess()

#TODO: Реализовать base_class interface для Extractor and Loader. interface следует реализовывать только для inference, т.к они самые часто заменяемые, для остальный избыточно

NEREvaluation = NEREvaluation(
    labels_config=labels,
    preprocessor=preprocessor,
    tokenizer=tokenizer,
    predict=predict,
    data_cleaner=evaluationDataPreprocess
)
report = NEREvaluation.run(data)
print(report)

