Resume NER

Исследовательский NER-проект для извлечения сущностей из резюме с помощью Transformer.

В проекте существует собственный dataset состоящий из 40 синтетических резюме с labels размечанный в docanno - admin.jsonl

Модель обучается находить ключевые данные в тексте резюме:

NAME
COMPANY
POSITION
SKILL


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
uv run python src/inference/app.py
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
