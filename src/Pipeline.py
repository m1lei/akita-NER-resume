from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

from src.DatasetLoader import DatasetLoader
from src.DatasetPreprocessor import DatasetPreprocessor
from src.LabelsManager import LabelsManager
from src.TrainerModel import TrainerModel



class NERPipeline:
    def __init__(self, train: str, output_dir: str, output_file,model_name:str, per_device_train:int,epoch:int):
        self.train = train
        self.output_dir = output_dir
        self.output_file = output_file
        self.model_name = model_name
        self.per_device_train = per_device_train
        self.epoch = epoch

    def run(self):
        loader = DatasetLoader(self.train, 40)
        data = loader.load()

        label_manager = LabelsManager(data)
        label, label2id, id2label = label_manager.run(self.output_file)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(label),
            label2id=label2id,
            id2label=id2label
        )

        preprocessor = DatasetPreprocessor()
        dataset = preprocessor.prepare_dataset(data, tokenizer, label2id)

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"#убрать ограничение по памяти_

        trainer = TrainerModel(tokenizer=tokenizer, model=model,output_dir=self.output_dir,datasets=dataset )
        trainer.train(self.per_device_train, self.epoch)











