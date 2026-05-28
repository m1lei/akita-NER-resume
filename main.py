from config import Config
from src.train.Pipeline import NERPipeline

config = Config.from_yaml('config.yaml')

pipeline = NERPipeline(
    train=config.train,
    output_dir=config.output_dir,
    model_name=config.model_name,
    per_device_train=config.per_device_train_batch_size,
    epoch=config.num_train_epochs,
    output_file=config.output_file,
    )

pipeline.run()

