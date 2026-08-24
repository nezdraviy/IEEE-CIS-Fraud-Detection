# IEEE-CIS Fraud Detection: GPU-Accelerated Anti-Fraud

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-cuDF%20%7C%20cuML-green.svg)](https://rapids.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UV](https://img.shields.io/badge/Package%20Manager-UV-purple.svg)](https://github.com/astral-sh/uv)

Репозиторий содержит полное решение для соревнования [IEEE-CIS Fraud Detection (Kaggle, 2019)](https://www.kaggle.com/competitions/ieee-fraud-detection/overview). 
Проект реализует end-to-end пайплайн обработки транзакционных данных с использованием **GPU-ускорения (RAPIDS cuDF/cuML)**, продвинутого Feature Engineering и ансамбля градиентных бустингов с настройкой через Optuna.

## Ключевые особенности

* **GPU-Accelerated EDA & FE:** Использование `cudf` и `cuml` для молниеносной обработки миллионов строк и PCA прямо на видеокарте.
* **Feature Engineering:** 
  * Генерация `uid` (User ID) на основе `card1`, `addr1` и временных сдвигов (`D1n`).
  * Агрегации (mean, std) по `uid` для выявления поведенческих паттернов.
  * PCA для сжатия высокоразмерных и скоррелированных `V-features`.
* **Hyperparameter Tuning:** Автоматический подбор гиперпараметров для XGBoost, LightGBM и CatBoost с помощью `Optuna`.
* **Blending:** Усреднение предсказаний разнородных моделей для повышения стабильности и метрики ROC-AUC.

##  Результаты (Cross-Validation ROC-AUC)
| Model | ROC-AUC Score |
| :--- | :--- |
| **XGBoost (GPU)** | **0.9810** |
| LightGBM (CPU) | 0.9733 |
| CatBoost (GPU) | 0.9709 |
| Blending	     | 0.9766 |

## Технологический стек
* **Язык:** Python 3.13+
* **Data Manipulation:** `cudf` (GPU), `pandas`, `numpy`, `cupy`
* **Machine Learning:** `xgboost`, `lightgbm`, `catboost`, `scikit-learn`, `cuml`
* **Tuning & Viz:** `optuna`, `matplotlib`, `seaborn`
* **Environment:** `uv` (Astral)

## Структура проекта
```text
├── data/                   # Сырые и обработанные данные (игнорируется в git)
├── notebooks/              # Jupyter ноутбуки (EDA, Feature Engineering & Modeling)
├── src/                    # Техническая папка
├── pyproject.toml          # Конфигурация зависимостей и проекта (uv)
├── README.md               # Описание проекта
└── LICENSE                 # MIT License