from inferece_pipeline import NERInferencePipeline

pipeline = NERInferencePipeline(
    model_path="/Users/maksim/PycharmProjects/rezume_NER/debug_model_checkpoints",
    labels_path="/Users/maksim/PycharmProjects/rezume_NER/debug_model_checkpoints.json"
)

pipeline.load()

text = 'Синицына Анна Владимировна. 2005-2009 — МГУ, факультет экономики. Работала аналитиком в Сбербанке с 2010 по 2015, потом перешла в McKinsey, где работала до 2020. Сейчас ведущий data scientist в Тинькофф. Навыки: python, pandas, numpy, ml, tableau, excel.'

result = pipeline.predict(text)

print(result)