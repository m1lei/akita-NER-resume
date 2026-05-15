from src.inference.inferece_pipeline import NERInferencePipeline
from src.inference.Normalize import Normalize


pipeline = NERInferencePipeline(
    model_path="/Users/maksim/PycharmProjects/rezume_NER/modelPavel/debug_model_checkpoints",
    labels_path="/Users/maksim/PycharmProjects/rezume_NER/modelPavel/debug_model_checkpoints.json"
)

pipeline.load()

text = '''
Анна Соколова
Junior Data Scientist

Опыт работы:
- Стажёр в отделе аналитики, ООО "ТехноПрогресс" (2023–2024)
  • Сбор и очистка данных с помощью pandas, numpy
  • Построение простых моделей регрессии в scikit-learn
  • Визуализация результатов в Matplotlib и Plotly

Навыки:
• Языки: Python, SQL, Bash
• Библиотеки: pandas, scikit-learn, seaborn, Hugging Face Transformers
• Инструменты: Git, Docker, Jupyter, FastAPI
• ML: классификация, кластеризация, feature engineering, cross-validation
• NLP: токенизация, NER, fine-tuning BERT, inference

Образование:
МГУ им. Ломоносова, Прикладная математика и информатика, бакалавр (2020–2024)

Дополнительно:
- Пет-проект: Telegram-бот для классификации новостей (Python + FastAPI + SQLite)
- Сертификат: "Machine Learning Specialization" (Coursera, 2023)
- Английский: B2 (Upper-Intermediate)
'''

result = pipeline.predict(text)

norm = Normalize(result)

norm.normalize()