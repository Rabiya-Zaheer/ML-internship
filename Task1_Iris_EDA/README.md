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

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
jupyter notebook task1_iris_analysis.ipynb
```

---

**Intern:** Rabiya Zaheer
**Organization:** DevelopersHub Corp
