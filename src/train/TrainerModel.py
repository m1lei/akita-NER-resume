from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments


class TrainerModel:
    def __init__(self, model , tokenizer, datasets, output_dir: str):
        self.dataset = datasets
        self.tokenizer = tokenizer
        self.model = model
        self.output_dir = output_dir

    def train(self, per_device_train_batch_size: int, num_train_epochs: int):
        data_collator = DataCollatorForTokenClassification(self.tokenizer)


        #данные настройки можно и нужно изменить в зависимости от мощности устройства на котором происходит обучение
        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=5e-5,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            gradient_accumulation_steps=8,
            fp16=True,  # Включаем Mixed Precision (16-бит)
            dataloader_pin_memory=False,  # Отключаем фиксацию памяти (важно для M1)
            optim="adamw_torch",  # Стандартный оптимизатор
            dataloader_num_workers=0,  # Чтобы не плодить процессы
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset,
            data_collator=data_collator,
        )

        trainer.train()
        self.tokenizer.save_pretrained(self.output_dir)
        trainer.save_model()
