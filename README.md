

# 🔍 Model Evaluation and Cross-Validation
The most basic way to evaluate a predictive model involves the following steps:

- 1️⃣ Split the dataset into a training set and a test set.
- 2️⃣ Train the model on the training set.
- 3️⃣ Measure the training error on the training set.
- 4️⃣ Measure the testing error on the test set.

## 🔁 Cross-Validation
Cross-validation extends this approach by repeatedly splitting the dataset into different training and test subsets.

- ✅ This method helps evaluate the robustness and stability of a model.
- ✅ It provides multiple estimates of generalization performance, reducing the dependence on a single train/test split.

Some techniques include:
- 1️⃣ ShuffleSplit performs n_splits independent iterations, where each time:
  - The full dataset is randomly shuffled
  - A train/test split is created from the shuffled data
  - A new model is trained on the new training set
  - The model is tested on the corresponding test set

```python
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
cv_results = cross_validate(
    regressor, data, target, cv=cv, scoring="neg_mean_absolute_error")
```
Note that `cv_results` is a Python dictionary data type.

- 2️⃣ KFold
  K-Fold splits data into K equal parts (folds). The model trains on `K-1` folds and tests on the remaining fold, repeating this `K` times, so each fold serves as the test set once. This gives a robust estimate of model performance.

```python
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LogisticRegression

# Create K-Fold (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Use in cross_validate
results = cross_validate(LogisticRegression(), X, y, cv=kf, scoring="neg_mean_absolute_error")
```

## ⚙️ Using Error Metrics in Cross-Validation
In scikit-learn, all error metrics—such as mean_absolute_error—can be transformed into a score for use in cross_validate().

🧩 To do this, prepend the metric name with the prefix "neg_" and pass it to the scoring parameter.

### 📘 Example:

```python
  from sklearn.model_selection import cross_validate
  
  scoring = "neg_mean_absolute_error"
  results = cross_validate(model, X, y, scoring=scoring, cv=5)
  print(results["test_score"])
```

In this case, scikit-learn computes the negative of the mean absolute error, effectively converting the error metric into a score (since higher scores indicate better performance).

**Note:** We might instead choose a metric relative to the target value to predict: the mean absolute percentage error would have been a much better choice.

## 🏠 California Housing Dataset
All cross-validation techniques demonstrated here are applied to scikit-learn's California Housing dataset (20,640 samples, 8 features).

**Key characteristics:**
- Features: Median income, house age, rooms, bedrooms, population, occupancy, latitude/longitude
- Target: Median house value (in 100,000 USD units)
- Perfect for regression tasks and cross-validation experiments


