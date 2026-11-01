# Customer Segmentation & Fraud Detection Pipeline 🤖

This repository contains an end-to-end Machine Learning project divided into two main stages: **Unsupervised Clustering** for customer segmentation and **Supervised Classification** for predictive modeling based on transactional data.

---

## 📁 Project Structure

```text
├── data_clustering.csv                         # Processed dataset after clustering
├── data_clustering_inverse.csv                 # Inverse-transformed dataset for classification
├── model_clustering.h5                         # Trained K-Means clustering model
├── PCA_model_clustering.h5                     # PCA model for dimensionality reduction
├── decision_tree_model.h5                      # Decision Tree baseline classification model
├── explore_random_forest_classification.h5     # Random Forest classification model
├── tuning_classification.h5                    # Optimized Random Forest model via GridSearchCV
└── README.md                                   # Project documentation

```

---

## 📊 Phase 1: Unsupervised Clustering & Data Preprocessing

This phase focuses on data exploration, cleaning, and customer segmentation using transactional data attributes (e.g., `TransactionAmount`, `CustomerAge`, `AccountBalance`, etc.).

### Key Steps & Implementation:

1. **Exploratory Data Analysis (EDA):** Analyzed feature correlations, data distributions via histograms, and boxplots grouped by customer occupation.


2. **Data Cleaning & Preprocessing:**
    * Checked and handled missing values (`isnull().sum()`) and duplicate rows.


    * Dropped irrelevant identifier columns (containing `id`, `ip`, or `date`).


    * Applied **Label Encoding** for categorical features.


    * Handled outliers using the Interquartile Range (IQR) method and scaled numerical features using **StandardScaler**.




3. **Clustering Model (K-Means):**
    * Determined optimal clusters ($k=3$) using the **Elbow / Silhouette Method** (achieving a Silhouette Score around $0.496 - 0.573$).


    * Applied **PCA (Principal Component Analysis)** for 2D visualization and centroid mapping.




4. **Cluster Interpretation & Inverse Transformation:**
    * Evaluated clusters into meaningful segments: *Conservative Customers*, *Regular Young Customers*, and *Priority Senior Customers with Higher Balances*.


    * Renamed cluster labels to `Target` and performed inverse transformation back to normal ranges for subsequent supervised learning.





---

## 🤖 Phase 2: Supervised Classification & Hyperparameter Tuning

Using the labeled dataset generated from the clustering phase (`data_clustering_inverse.csv`), this phase trains predictive models to classify transaction targets.

### Key Steps & Implementation:

1. **Feature Engineering:** Applied **One-Hot Encoding** (`pd.get_dummies`) on categorical variables.


2. **Data Splitting:** Split the dataset (1,945 total entries) into training (80% / 1,556 samples) and testing sets (20% / 389 samples) using `train_test_split` with `stratify=y`.


3. **Model Training & Comparison:**
    * **Decision Tree Classifier:** Baseline model achieving ~95% accuracy.


    * **Random Forest Classifier:** Advanced ensemble model achieving ~98% accuracy.




4. **Hyperparameter Tuning:** Optimized the Random Forest model using **GridSearchCV** with parameters (`n_estimators`, `max_depth`, `min_samples_split`), pushing evaluation metrics to **99% accuracy**.



---

## 🛠️ Technologies & Libraries Used

  * **Language:** Python


* **Data Manipulation & Analysis:** Pandas, NumPy


* **Data Visualization:** Matplotlib, Seaborn, Yellowbrick


* **Machine Learning (Scikit-Learn):**
* *Preprocesing:* StandardScaler, LabelEncoder


* *Clustering:* KMeans, PCA


* *Classification:* DecisionTreeClassifier, RandomForestClassifier


* *Evaluation & Tuning:* train_test_split, GridSearchCV, classification_report




* **Model Serialization:** Joblib



---

## 🚀 How to Run

1. Clone this repository:
```bash
git clone [https://github.com/Namira320/your-repo-name.git](https://github.com/Namira320/your-repo-name.git)

```


2. Install the required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick joblib

```


3. Run the Jupyter Notebooks or load the pre-trained `.h5` models using `joblib`:
```python
import joblib
model = joblib.load('tuning_classification.h5')

```

