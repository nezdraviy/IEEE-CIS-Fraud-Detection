# IEEE-CIS Fraud Detection: GPU-Accelerated Anti-Fraud

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-cuDF%20%7C%20cuML-green.svg)](https://rapids.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UV](https://img.shields.io/badge/Package%20Manager-UV-purple.svg)](https://github.com/astral-sh/uv)

Репозиторий содержит полное решение для соревнования [IEEE-CIS Fraud Detection (Kaggle, 2019)](https://www.kaggle.com/competitions/ieee-fraud-detection/overview). 
Проект реализует end-to-end пайплайн обработки транзакционных данных с использованием **GPU-ускорения (RAPIDS cuDF/cuML)**, продвинутого Feature Engineering и ансамбля градиентных бустингов с настройкой через Optuna.

**Результат Private/Public LB:**
<img width="1665" height="190" alt="image" src="https://github.com/user-attachments/assets/6f469412-b67f-467c-8300-4a44710b5415" />

## Ключевые особенности

### 1. Загрузка и предобработка данных
- Использование **RAPIDS (`cudf`, `cupy`)** для молниеносной загрузки и обработки данных на GPU.
- Автоматическое удаление признаков с критически высоким процентом пропусков на основе предварительного анализа (`nans.json`).
- Приведение `V`-признаков к типу `float32` для оптимизации потребления видеопамяти.

### 2. Feature Engineering
Реализован в виде единого переиспользуемого пайплайна `FeatureEngineer`, который обучается на train-данных и применяется к validation/holdout:
- **Frequency Encoding**: Замена категориальных признаков на их частоту встречаемости.
- **Временные признаки и UID**: Генерация нормализованных временных дельт (`D1n`, `D4n`, `D10n`, `D15n`) и создание композитного идентификатора `uid` (`card1` + `addr1` + `D1n`).
- **Групповые агрегации**: Расчет статистик (`count`, `mean`, `std`) по группам `card1`, `card1 + день` и `uid` для ключевых переменных (`TransactionAmt`, `dist1`, `D`-признаки).
- **Признаки отклонения (Deviation)**: Расчет z-score отклонения суммы транзакции и дистанции от групповых средних (например, `Amt_dev_card1`).
- **Импутация**: Заполнение оставшихся пропусков медианой с помощью `SimpleImputer`.

### 3. Стратегия валидации
- **Time Series Split** (4 фолда), гарантирующий, что validation-выборка всегда хронологически следует за train-выборкой. Это максимально точно имитирует предсказание будущих транзакций и предотвращает утечку данных.
- Выделение последних 20% данных в `holdout` для финальной оценки качества.

### 4. Подбор гиперпараметров
- Использование **Optuna** для оптимизации гиперпараметров `XGBoost` (`max_depth`, `n_estimators`, `learning_rate`).
- *Примечание:* Оптимизация проводилась только для XGBoost (с `tree_method='hist'` и `device='cuda'`) из-за технических проблем запуска LightGBM на CUDA и CatBoost внутри Optuna в данной среде.

### 5. Моделирование
Обучение ансамбля разнородных моделей для повышения робастности предсказаний:
- **XGBoost**: Основная модель, обученная на GPU с лучшими параметрами из Optuna.
- **LightGBM**: Обучена на CPU для внесения разнообразия в ансамбль.
- **CatBoost**: Обучена на CPU с ранней остановкой (early stopping).
- **MLP (PyTorch)**: Дополнительная полносвязная нейронная сеть, обучаемая на GPU. Архитектура включает `BatchNorm1d`, `ReLU`, `Dropout` и функцию потерь `BCEWithLogitsLoss` с `pos_weight` для борьбы с дисбалансом классов.

### 6. Blending 
Итоговый прогноз формируется путем взвешенного усреднения вероятностей моделей:
- **XGBoost**: 50%
- **LightGBM**: 35%
- **CatBoost**: 15%

## Результаты

Оценка качества (ROC-AUC) на отложенной временной выборке (`holdout`, 20% данных):

| Модель | ROC-AUC (Holdout) |
| :--- | :---: |
| **XGBoost** (GPU) | `0.9281` |
| **LightGBM** (CPU) | `0.9178` |
| **CatBoost** (CPU) | `0.8999` |
| **Blending** (0.5 / 0.35 / 0.15) | **`0.9266`** |
| **MLP** (PyTorch, последний фолд) | `0.8751` |

## Технологический стек
* **Язык:** Python 3.13+
* **Data Manipulation:** `cudf` (GPU), `pandas`, `numpy`, `cupy`
* **Machine Learning:** `xgboost`, `lightgbm`, `catboost`, `scikit-learn`, `cuml`, `PyTorch`
* **Tuning & Viz:** `optuna`, `matplotlib`, `seaborn`
* **Environment:** `uv` (Astral), `WSL2`, `Docker`

## Структура проекта
```text
├── data/                   # Сырые и обработанные данные (игнорируется в git)
├── notebooks/              # Jupyter ноутбуки (EDA, Feature Engineering & Modeling)
├── src/                    # Техническая папка
├── pyproject.toml          # Конфигурация зависимостей и проекта (uv)
├── README.md               # Описание проекта
├── LICENSE                 # MIT License
└── Dockerfile              # Конфигурация контейнера