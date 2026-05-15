class EntityExtractor:
    '''
    class для работы с преобразованием предсказывания модели в понятные сущности типа [labels: word]
    '''
    def __init__(self, id2label: dict):
        self.id2label = id2label

    def extract(self, tokens, pred_ids):
        '''

        :param tokens: список токенов после tokenizer
        :param pred_ids: id предсказанных labels для каждого токена
        :return: готовый список сущностей
        '''
        entities = []
        current_word = ""
        current_label = None

        for token, pred_id in zip(tokens, pred_ids):
            label = self.id2label[pred_id]
            # Убираем B- или I- префиксы для простоты сравнения
            clean_label = label.replace("B-", "").replace("I-", "")

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            # Если метка "O", закрываем текущую сущность
            if label == "O":
                if current_word:
                    entities.append({"word": current_word.strip(), "label": current_label})
                    current_word = ""
                    current_label = None
                continue

            # Если это продолжение слова (##)
            if token.startswith("##"):
                current_word += token[2:]
            else:
                # Если сменился тип сущности (например, с NAME на POSITION)
                if current_label and clean_label != current_label:
                    entities.append({"word": current_word.strip(), "label": current_label})
                    current_word = token
                else:
                    # Добавляем пробел между отдельными словами одной сущности
                    current_word += " " + token

            current_label = clean_label

        # Не забываем добавить последнюю сущность
        if current_word:
            entities.append({"word": current_word.strip(), "label": current_label})

        return entities
