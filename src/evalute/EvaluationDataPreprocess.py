

class EvaluationDataPreprocess:
    def __init__(self,):
        pass

    def convert_ids_to_clean_tags(self,true_labels, id2labels, tokens, pred_labels):
        clean_true_labels = []
        clean_pred_labels = []

        for token, t_id, p_id in zip(tokens, true_labels, pred_labels):

            if t_id == -100:
                continue

            if token.startswith("##"):
                continue

            clean_true_labels.append(id2labels.get(str(t_id)))
            clean_pred_labels.append(id2labels.get(str(p_id)))

        return clean_true_labels, clean_pred_labels