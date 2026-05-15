from datasets import Dataset

class DatasetPreprocessor:
    def __init__(self):
        pass

    def prepare_dataset(self, data, tokenizer, label2id)->Dataset:
        features = []
        for item in data:
            tokenizer_item = self.tokenize_and_align_labels(item, tokenizer, label2id)
            features.append(tokenizer_item)
        dataset = Dataset.from_list(features)

        return dataset

    def tokenize_and_align_labels(self, data, tokenizer , label2id, max_length=128):
        text = data["text"]
        labels = data["label"]

        enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)#токенизируем текст
        offset = enc["offset_mapping"]#смещение токенов
        input_ids = enc["input_ids"]#в виде цифр
        align_labels = [-100] * len(input_ids)  # изначально каждое слово -100 пропускается при обучении

        for i, (start, end) in enumerate(offset):
            if start == end:
                continue
            token_label = "O"
            for lab_start, lab_end, tag in labels:
                if start >= int(lab_start) and end <= int(lab_end):
                    if start == int(lab_start):
                        token_label = f"B-{tag}"
                    else:
                        token_label = f"I-{tag}"
                    break
            align_labels[i] = label2id.get(token_label,label2id["O"])

        return {
            "input_ids": input_ids,
            "attention_mask": enc['attention_mask'],
            "labels": align_labels
        }
