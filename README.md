# ML-project for Named Entity Recognition (NER)

---

## Возможности проекта:

- fast fine-tuning transformer-model
- fast inference, на пользовательском тексте
- fast evalution, по отдельному dataset

## Review
* **Автоматическое выравнивание меток:** Корректная обработка Subword-токенизации (WordPiece) с маскированием технических токенов (`-100`) и генерацией префиксов `B-` и `I-` для длинных сущностей.
* **Изолированная архитектура:** Компоненты предобработки, обучения и оценки полностью разделены, что позволяет легко менять базовую модель (например, переходить на SentencePiece токенизаторы).
* **Продвинутая метрика качества:** Оценка модели с помощью `seqeval`


## Стек

- BIO разметка
- Python 3.12
- uv

---
## Запуск
### 1. install uv

для mac/linux
```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

for win
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

### 2.  install зависимостей
```bach
uv sync
```

### 3.Запуск Train
```
uv run python main.py
```
После запуска:
* обучится модель
* создастся папка ./debug_model_checkpoints
* сохранится debug_model_checkpoints.json

### 4. Запуск inference
внутри файла можно отредактировать пользовательский текст для тестирования
```bash
uv run python inference.py
```

### 5. Запуск evaluation
```bash
uv run python Evaluation_pipeline.py
```
отобразиться таблица оценка качества модели

### 6. Декларативное управление
Весь пайплайн (параметры токенизации, пути к данным, словари маппинга id2label для инференса и валидации) гибко конфигурируется через единый файл настроек.


---

# Пример работы

В качестве демонстрации возможностей pipeline представлена модель для анализа различных резюме.

###  Описание демонстрационной модели:
* **Базовая архитектура:** `DeepPavlov/rubert-base-cased` (BERT)
* **Аргументы обучения** 
  1. max_length_tok: 512;
  2. per_device_train_batch_size: 8
  3. num_train_epochs: 30
  4. Обучение проходило на dataset состоящем из **74** разных синтетических resume на русском языке 
* **Задача:** Извлечение ключевых сущностей из неструктурированного текста резюме.
* **Поддерживаемые теги (Entities):**

| Сущность   | Что выделять                                            | Пример                                    |
| ---------- | ------------------------------------------------------- | ----------------------------------------- |
| `NAME`     | Имя и фамилия кандидата                                 | `Ivan Petrov`                             |
| `COMPANY`  | Названия компаний-работодателей                         | `Yandex`, `Sber`                          |
| `POSITION` | Должности / роли                                        | `Data Scientist`, `Python Developer`      |
| `SKILL`    | Навыки и технологии                                     | `Python`, `SQL`, `PyTorch`                |
| LOCATION   | Город                                                   | Москва                                    |
| CONTACT    | как связаться с соискателем почта, phone, username в тг | @PavelDurov, gumi@gmail.com, +79156707855 |
| EDUCATION  | Название образ учереждения                              | МГУ, Garvard                              |
| SPECIALTY  | название специальности                                  | плотник                                   |
---

### 1. Входной текст (Текст резюме)
Этот неструктурированный текст передается на вход скрипту `inference.py`:
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
### 2. Результат работы модели:
```json
{
  "NAME": ["Анна Соколова"],
 "POSITION": ["Junior Data Scientist"], 
 "COMPANY": ["ТехноПрогресс"], 
 "SKILLS": ["Bash", "Docker", "FastAPI", "Git", "Hugging Face Transformers", "Jupyter",
            "ML", "Matplotlib", "NER", "NLP", "Plotly", "Python", "SQL", "SQLite", "Telegram-бот",
             "Upper-Intermediate", "cross-validation", "feature engineering", "fine-tuning BERT", 
             "inference", "numpy", "pandas", "scikit-learn", "seaborn", "Визуализация", "Сбор", 
             "классификации новостей", "классификация", "кластеризация", "очистка данных", "простых моделей регрессии",
              "токенизация"],
  "EDUCATION": ["МГУ им. Ломоносова"],
  "SPECIALTY": ["Прикладная математика и информатика"]}
```

### 3. Scope model:
Оценка произведена в соответствии с Strict Match CoNLL-2003:

|      | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| COMPANY   | 0.50      | 0.80   | 0.62     | 15      |
| CONTACT   | 0.89      | 0.92   | 0.91     | 26      |
| EDUCATION | 0.87      | 1.00   | 0.93     | 20      |
| LOCATION  | 0.88      | 0.83   | 0.86     | 18      |
| NAME      | 0.95      | 1.00   | 0.98     | 21      |
| POSITION  | 0.60      | 0.81   | 0.68     | 31      |
| SKILL     | 0.51      | 0.81   | 0.62     | 177     |
| SPECIALTY | 0.60      | 0.83   | 0.70     | 18      |
| micro avg | 0.60    | 0.84   | 0.70     | 326     |
| macro avg | 0.72    | 0.88   | 0.79     | 326     |
| weighted avg | 0.62 | 0.84   | 0.71     | 326     |
