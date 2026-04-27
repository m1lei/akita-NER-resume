import yaml
from dataclasses import dataclass

@dataclass
class Config:
    train: str
    labels: str
    output_dir: str
    output_file: str

    model_name: str

    max_length_tok: int
    per_device_train_batch_size: int
    num_train_epochs: int
    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)