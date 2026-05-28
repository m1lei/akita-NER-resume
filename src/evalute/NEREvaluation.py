import seqeval.metrics


class NEREvaluation:
    """
    class запуска NLP evaluation pipeline
    """
    def __init__(self, labels_config:dict, preprocessor, tokenizer, predict, data_cleaner):
        self.labels_config = labels_config
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.predict = predict
        self.data_cleaner = data_cleaner

    def run(self, test_data: list, ):
        all_true_evaluation = []
        all_pred_evaluation = []

        id2label = self.labels_config["id2label"]
        label2id = self.labels_config["label2id"]

        for item in test_data:
            processed_item = self.preprocessor.tokenize_and_align_labels(item, self.tokenizer, label2id)

            tokens = self.tokenizer.convert_ids_to_tokens(processed_item['input_ids'])
            y_true_ids = processed_item['labels']
            y_pred_ids = self.predict.predict_from_ids(
                input_ids=processed_item['input_ids'],
                attention_mask=processed_item['attention_mask']
            )

            clean_true, clean_pred = self.data_cleaner.convert_ids_to_clean_tags(
                y_true_ids,
                id2labels=id2label,
                tokens=tokens,
                pred_labels=y_pred_ids,
            )
            all_true_evaluation.append(clean_true)
            all_pred_evaluation.append(clean_pred)

        report = seqeval.metrics.classification_report(all_true_evaluation, all_pred_evaluation)

        return report
