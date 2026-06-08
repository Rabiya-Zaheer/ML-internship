# DevelopersHub Corp — AI/ML Internship

This repo contains my work for the AI/ML Internship at DevelopersHub Corp.
Each task has its own folder with a Jupyter notebook, code, and outputs.

---

## Task 1 — Iris Dataset: Exploration & Visualization

The goal was to get comfortable with loading a dataset, running basic inspection,
and creating visualizations to understand what the data looks like before any modeling.

### Dataset

| Property | Detail |
|---|---|
| Name | Iris Dataset |
| Source | `sklearn.datasets.load_iris()` |
| Size | 150 rows × 5 columns |
| Features | sepal_length, sepal_width, petal_length, petal_width |
| Target | species — Setosa, Versicolor, Virginica (50 each) |
| Missing Values | None |

### Libraries Used

```python
pandas, numpy, matplotlib, seaborn, scipy, sklearn
```

### What I did

- Loaded the dataset and inspected it using `.head()`, `.info()`, and `.describe()`
- Plotted a scatter matrix to see relationships between features
- Used histograms to understand how each feature is distributed across species
- Used box plots to check for outliers

### Key Findings

- Setosa is clearly separable from the other two species — especially by petal size
- Petal Length and Petal Width are the most useful features for telling species apart
- Sepal Width had the most outliers, mostly in Setosa
- Versicolor and Virginica overlap a bit, so a simple linear model might struggle there

---

## Task 2 — AAPL Stock Price Prediction (Short-Term)

The goal was to predict the next day's closing price of Apple stock (AAPL) using
historical OHLCV data and engineered time-series features. Two regression models
were trained and compared.

### Dataset

| Property | Detail |
|---|---|
| Stock | Apple Inc. (AAPL) |
| Source | Yahoo Finance via `yfinance` |
| Period | January 2022 – December 2024 |
| Size | ~782 trading days |
| Raw Features | Open, High, Low, Close, Volume |
| Target | Next day's Close price |

### Libraries Used

```python
pandas, numpy, matplotlib, sklearn, yfinance, scipy
```

### What I did

- Fetched 3 years of AAPL daily stock data using `yfinance`
- Engineered 12 features from previous-day OHLCV data, moving averages,
  momentum, and volatility — careful to avoid data leakage
- Split data 80/20 by time (no shuffling — time series rule)
- Trained Linear Regression and Random Forest (300 trees) models
- Evaluated with RMSE, MAE, and R²
- Plotted actual vs predicted prices, feature importance, and residuals

### Key Findings

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | $3.05 | $2.44 | 0.797 |
| Random Forest | $3.14 | $2.52 | 0.786 |

- Linear Regression slightly outperformed Random Forest — stock prices have
  strong linear autocorrelation, which LR handles well
- Prev_Close was the most important feature (~42% RF importance)
- Both models explained ~80% of variance in test data
- Residuals were centered around zero with no systematic bias
- Prediction accuracy dropped during high-volatility periods — expected
  since no news or sentiment data was included

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy yfinance

# Task 1
jupyter notebook Task1_Iris_EDA/task1_iris_analysis.ipynb

# Task 2
jupyter notebook Task2_Stock_Prediction/task2_stock_prediction.ipynb
```
---

## Task 3 — Heart Disease Prediction (Binary Classification)

The goal was to predict whether a patient is at risk of heart disease using
clinical health measurements. Built and compared two classification models.

### Dataset

| Property | Detail |
|---|---|
| Name | Heart Disease UCI Dataset |
| Source | Kaggle / UCI Machine Learning Repository |
| Size | 303 patients × 14 columns |
| Features | 13 clinical features (age, sex, chest pain type, cholesterol, etc.) |
| Target | 0 = No Disease, 1 = Heart Disease |
| Missing Values | 6 (handled via imputation) |

### Libraries Used

```python
pandas, numpy, matplotlib, seaborn, sklearn
```

### What I did

- Inspected and cleaned the dataset — imputed 6 missing values in `ca` and `thal`
- Performed EDA: age distribution, chest pain type analysis, correlation heatmap,
  scatter plots, and box plots of key clinical features
- Split data 80/20 with stratification to preserve class balance
- Trained Logistic Regression and Decision Tree classifiers
- Evaluated using Accuracy, ROC-AUC, 5-fold Cross Validation, and Confusion Matrix
- Analyzed feature importance from both models

### Key Findings

| Model | Accuracy | ROC-AUC | CV Score |
|---|---|---|---|
| Logistic Regression | 96.7% | 0.993 | 89.3% |
| Decision Tree | 73.8% | 0.754 | 72.7% |

- Logistic Regression significantly outperformed Decision Tree
- Most important features: Major Vessels (ca), Thalassemia type (thal),
  ST Depression (oldpeak), Exercise Angina (exang)
- Heart disease patients showed notably lower max heart rate during exercise
- Cholesterol alone was a weak predictor — consistent with medical literature

---

## Task 4 — News Topic Classifier Using BERT (NLP + Transformers)

Fine-tuned a BERT-style transformer model to classify news headlines into
4 topic categories using the AG News dataset. Built a live Gradio web app
for real-time classification.

### Dataset

| Property | Detail |
|---|---|
| Name | AG News Dataset |
| Source | Hugging Face Datasets (`ag_news`) |
| Size | 127,600 samples (120K train / 7.6K test) |
| Classes | World, Sports, Business, Sci/Tech (balanced) |
| Input | News headline text |

### Libraries Used

```python
transformers, datasets, torch, gradio, scikit-learn, numpy
```

### What I did

- Loaded AG News via Hugging Face `datasets` library
- Tokenized headlines using `bert-base-uncased` WordPiece tokenizer
  (adding [CLS], [SEP], attention masks, padding)
- Fine-tuned `bert-base-uncased` (110M params) with a classification head
  using Hugging Face `Trainer` API (3 epochs, lr=2e-5, batch=32)
- Evaluated with accuracy, F1 macro, per-class F1, and confusion matrix
- Deployed the model as a live Gradio web interface (`app.py`)

### Key Results

| Metric | Score |
|---|---|
| Accuracy | ~94–95% |
| F1 Macro | ~0.94–0.95 |

- Sports headlines were easiest to classify (domain-specific vocabulary)
- Business/World sometimes overlap (geopolitical-economic news)
- Fine-tuning 3 epochs on 8K examples achieves 94%+ — power of transfer learning
- Gradio `share=True` gives instant public URL for live demo

### How to run Gradio app

```bash
python app.py
# Opens at http://localhost:7860
# share=True gives a public URL
```

---

## Task 5 — Customer Churn Prediction (End-to-End ML Pipeline)

Built a production-ready ML pipeline using scikit-learn's Pipeline API for
predicting telecom customer churn. Full preprocessing, training, GridSearchCV
tuning, and joblib export in one reusable object.

### Dataset

| Property | Detail |
|---|---|
| Name | Telco Customer Churn |
| Source | IBM / Kaggle |
| Size | 7,043 customers × 20 features |
| Target | Churn: Yes/No |
| Class Balance | ~81% No, ~19% Yes |
| Missing Values | 11 in TotalCharges (handled via pipeline imputation) |

### Libraries Used

```python
pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
```

### What I did

- Built ColumnTransformer with separate pipelines per feature type:
  numeric (median impute → StandardScaler), binary (mode impute → OHE),
  nominal (mode impute → OneHotEncoder)
- Chained preprocessing + classifier into a single sklearn Pipeline
- Fitted Logistic Regression and Random Forest baselines
- Ran GridSearchCV (5-fold StratifiedKFold) over 16 LR + 12 RF param combos
- Evaluated with Accuracy, F1 Macro, ROC-AUC, confusion matrix
- Extracted feature importances from best Random Forest
- Exported both tuned pipelines as .joblib files for production use

### Key Results

| Model | Accuracy | ROC-AUC | Best CV AUC |
|---|---|---|---|
| Logistic Regression | 81.1% | 0.761 | 0.738 |
| Random Forest | 81.0% | 0.756 | 0.730 |

- Contract type was by far the most important predictor (~50% RF importance)
- Month-to-month customers churn ~4× more than 2-year contract customers
- Tenure, monthly charges, and internet service type were next most important
- ROC-AUC used as primary metric (accuracy misleading on 81/19 imbalanced data)

### Using the exported pipeline

```python
import joblib, pandas as pd

pipeline = joblib.load('pipeline_logistic_regression.joblib')
new_data = pd.DataFrame([{...}])  # raw customer data, no preprocessing needed
prediction   = pipeline.predict(new_data)        # 0=No Churn, 1=Churn
probability  = pipeline.predict_proba(new_data)[:,1]  # churn probability
```
---
---

**Intern Name:** Rabiya Zaheer
**Intern ID:** DHC 1622
**Organization:** DevelopersHub Corp
