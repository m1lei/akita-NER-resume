Resume NER

Исследовательский ML-project для training and inference NER-models, которая извлекает сущности из текста резюме.

Система поддерживает настройку через YAML-конфиг, работу с датасетами и запуск предсказаний на пользовательском тексте

В проекте существует собственный dataset, состоящий из ~90 синтетических резюме с labels, размечанные в docanno - admin_2.jsonl

Используется BIO разметка

Модель обучается находить ключевые сущности в тексте резюме:

NAME
COMPANY
POSITION
SKILL
LOCATION
CONTACT 
EDUCATION 
SPECIALTY

# Installation & Run

## 1. install uv


```bash
pip install uv
```


---

## 2. install зависимостей

```bash
uv sync
```



## Config
чтобы внести свои данные можете поменять значения в config.yaml


---
## 3. Запуск Training

```bash
uv run python main.py
```

После запуска:

* обучится модель
* создастся папка `./debug_model_checkpoints`
* сохранится `debug_model_checkpoints.json`

---

## 4. Запуск inference

```bash
uv run python inference.py
```

---

## 5. Проблемы и решения

### MPS (Mac M1/M2)

Если возникают проблемы с памятью:

```python
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
```


# Quick Start

```bash
uv sync
uv run python main.py - запуск обучения
uv run python src/inference/app.py - запуск inference
```

# Пример работы

```text
text resume: Анна Соколова
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
```

```text
Ответ model:{'NAME': ['Анна Соколова'],
 'POSITION': ['Junior Data Scientist'], 
 'COMPANY': ['ТехноПрогресс'], 
 'SKILLS': ['Bash', 'Docker', 'FastAPI', 'Git', 'Hugging Face Transformers', 'Jupyter',
            'ML', 'Matplotlib', 'NER', 'NLP', 'Plotly', 'Python', 'SQL', 'SQLite', 'Telegram-бот',
             'Upper-Intermediate', 'cross-validation', 'feature engineering', 'fine-tuning BERT', 
             'inference', 'numpy', 'pandas', 'scikit-learn', 'seaborn', 'Визуализация', 'Сбор', 
             'классификации новостей', 'классификация', 'кластеризация', 'очистка данных', 'простых моделей регрессии',
              'токенизация'],
  'EDUCATION': ['МГУ им. Ломоносова'], 
  'SPECIALTY': ['Прикладная математика и информатика']}


```
