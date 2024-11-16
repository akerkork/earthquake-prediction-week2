# Earthquake Prediction - Task 2 - Abdülkerim Korkmaz

For this week's task, I have trained and compared performance of different machine learning models with not using random split in earthquake data. Majority Class Model and Random Class Model are also included to assess whether the predictive models have learned meaningful patterns from the data.

---

## **Models and Hyperparameters**

### **1. CatBoost Classifier**
- **Hyperparameters:**
  - `iterations=1000`
  - `learning_rate=0.1`
  - `depth=6`
  - `random_seed=15`
  - `verbose=False`

### **2. LightGBM Classifier**
- **Hyperparameters:**
  - `num_leaves=128`
  - `max_depth=6`
  - `min_split_gain=0.5`
  - `min_child_samples=20`
  - `random_seed=15`

### **3. Logistic Regression**
- **Hyperparameters:**
  - `multi_class='multinomial'` (for multiclass classification)
  - `solver='lbfgs'` (recommended solver for multinomial problems)
  - `max_iter=2000`
  - `random_seed=15`

### **4. Random Forest Classifier**
- **Hyperparameters:**
  - `n_estimators=100`
  - `random_seed=15`

### **5. XGBoost Classifier**
- **Hyperparameters:**
  - `n_estimators=1000`
  - `learning_rate=0.1`
  - `max_depth=6`
  - `random_seed=15`
  - `use_label_encoder=False`
  - `eval_metric='mlogloss'`

### **6. Neural Network**
- **Hyperparameters:**
  - **Architecture:**
    - Input Layer: Number of neurons equal to the number of features
    - Hidden Layers: 128, 64 (with Dropout of 0.5), and 32 neurons
    - Output Layer: Number of neurons equal to the number of classes with softmax activation
  - **Training Parameters:**
    - `epochs=50`
    - `batch_size=32`
    - Optimizer: Adam
    - Loss Function: Categorical Crossentropy

---

## **Results and Analysis**

### **1. CatBoost Classifier**
- **Accuracy:** 28.21%

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| 1         | 0.39          | 0.32       | 0.35         | 975         |
| 2         | 0.23          | 0.54       | 0.33         | 784         |
| 3         | 0.42          | 0.33       | 0.37         | 830         |
| 4         | 0.00          | 0.00       | 0.00         | 309         |
| 5         | 0.17          | 0.04       | 0.07         | 580         |
| 6         | 0.00          | 0.00       | 0.00         | 202         |

---

### **2. LightGBM Classifier**
- **Accuracy:** 30.76%

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| 1         | 0.39          | 0.31       | 0.35         | 975         |
| 2         | 0.29          | 0.60       | 0.39         | 784         |
| 3         | 0.35          | 0.37       | 0.36         | 830         |
| 4         | 0.00          | 0.00       | 0.00         | 309         |
| 5         | 0.28          | 0.08       | 0.13         | 580         |
| 6         | 0.00          | 0.00       | 0.00         | 202         |

---

### **3. Logistic Regression**
- **Accuracy:** 21.74%

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| 1         | 0.41          | 0.07       | 0.12         | 975         |
| 2         | 0.22          | 0.72       | 0.34         | 784         |
| 3         | 0.23          | 0.17       | 0.19         | 830         |
| 4         | 0.08          | 0.08       | 0.08         | 309         |
| 5         | 0.00          | 0.00       | 0.00         | 580         |
| 6         | 0.00          | 0.00       | 0.00         | 202         |

---

### **4. Random Forest Classifier**
- **Accuracy:** 29.92%

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| 1         | 0.38          | 0.27       | 0.32         | 975         |
| 2         | 0.27          | 0.67       | 0.38         | 784         |
| 3         | 0.38          | 0.34       | 0.36         | 830         |
| 4         | 0.01          | 0.01       | 0.01         | 309         |
| 5         | 0.30          | 0.05       | 0.09         | 580         |
| 6         | 0.00          | 0.00       | 0.00         | 202         |

---

### **5. XGBoost Classifier**
- **Accuracy:** 30.87%

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| 1         | 0.40          | 0.36       | 0.38         | 975         |
| 2         | 0.26          | 0.53       | 0.35         | 784         |
| 3         | 0.40          | 0.40       | 0.40         | 830         |
| 4         | 0.00          | 0.00       | 0.00         | 309         |
| 5         | 0.22          | 0.07       | 0.11         | 580         |
| 6         | 0.00          | 0.00       | 0.00         | 202         |

---

### **6. Neural Network**
- **Validation Accuracy:** 25.05%

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| 1         | 0.25          | 0.16       | 0.19         | 975         |
| 2         | 0.25          | 0.53       | 0.34         | 784         |
| 3         | 0.37          | 0.33       | 0.35         | 830         |
| 4         | 0.06          | 0.07       | 0.07         | 309         |
| 5         | 0.19          | 0.10       | 0.13         | 580         |
| 6         | 0.00          | 0.00       | 0.00         | 202         |

---

## **Baseline Models**

### **1. Random Class Model**
- **Accuracy:** 16.44%

### **2. Majority Class Model**
- **Accuracy:** 21.30%

---

## **Conclusions**
All predictive models outperform the random and majority class baselines, indicating they have learned some patterns from the data.

The models consistently perform poorly on classes 4, 5, and 6. This suggests a class imbalance problem where minority classes are underrepresented, leading to poor recall and precision for those classes.

XGBoost and LightGBM achieve the highest accuracy (~31%), showing better performance in predicting the majority classes.

Despite using different hyperparameters, the models' performance did not significantly improve. This suggests that hyperparameter tuning alone may not be sufficient to enhance model performance.
