### **Customer Churn Prediction — End-to-End ML System**

##### 

##### **Project Overview**

This project implements a full end-to-end machine learning system for predicting customer churn.

It goes beyond model training to include uncertainty estimation, cost-sensitive decision making, calibration, deployment, and containerization.



**The system outputs:**

* a probability of churn
* a business-aligned decision based on an optimized threshold

The entire workflow is reproducible, auditable, and deployable.

##### 

##### **Problem Statement**

Customer churn is costly. The objective is to:

&nbsp;- Identify customers at risk of churning early enough to take preventive action, while minimizing unnecessary interventions.



This is framed as a binary classification problem:

&nbsp;- 1 → customer will churn

&nbsp;- 0 → customer will not churn

The key challenge is not classification accuracy, but balancing false positives vs false negatives under asymmetric business costs.



##### **Dataset**

&nbsp;- Source: Telco Customer Churn dataset (Kaggle)

&nbsp;- Target variable: churn

&nbsp;- Observations: ~7,000 customers

&nbsp;- Feature types:

&nbsp;	Numerical (tenure, charges)

&nbsp;	Binary indicators

&nbsp;	Categorical features (encoded during preprocessing)

The cleaned dataset is stored in:

&nbsp;- data/processed/churn\_clean.csv



##### **Machine Learning Workflow**

This project follows a disciplined ML engineering pipeline:

**1. Data Ingestion \& Cleaning**

&nbsp;- Raw data is never modified

&nbsp;- Cleaning is fully reproducible

&nbsp;- Output: cleaned dataset

**2. Exploratory Data Analysis (EDA)**

&nbsp;- Target distribution analysis (class imbalance)

&nbsp;- Feature scale and variance inspection

&nbsp;- Correlation awareness (multicollinearity)

&nbsp;- EDA informs modeling decisions, not code execution.

**3. Feature Engineering**

&nbsp;- Automatic separation of numerical and categorical features

&nbsp;- Standard scaling for numeric variables

&nbsp;- One-hot encoding for categorical variables

&nbsp;- Stratified train/test split

All transformations are encapsulated in a single preprocessing pipeline.

**4. Model Training**

Baseline model: Logistic Regression (L2 regularization)

Chosen for:

&nbsp;- probabilistic output

&nbsp;- interpretability

&nbsp;- calibration friendliness

Model and preprocessing are trained as a single pipeline artifact

**5. Model Evaluation with Uncertainty**

Metrics are reported with confidence intervals (bootstrap):

&nbsp;- Accuracy

&nbsp;- Precision

&nbsp;- Recall

&nbsp;- ROC–AUC

&nbsp;- Confusion matrix

This avoids over-interpreting noisy point estimates.

**6. Threshold Optimization (Cost-Sensitive Decisions)**

Rather than using a default 0.5 threshold, we:

Explicitly define business costs:

&nbsp;- False Negatives (missed churners) are expensive

&nbsp;- False Positives (unnecessary offers) are cheaper

&nbsp;- Sweep thresholds from 0 → 1

&nbsp;- Select the threshold that minimizes expected cost

Chosen threshold: 0.14

This prioritizes high recall to capture most churners.

**7. Probability Calibration**

&nbsp;- Calibration curves (reliability diagrams)

&nbsp;- Brier score evaluation

This ensures predicted probabilities are interpretable as real risk estimates.

**8. Inference Pipeline**

A unified inference module:

&nbsp;- loads the trained (and calibrated) model

&nbsp;- applies the chosen threshold

&nbsp;- outputs probability + decision

This logic is reused by:

&nbsp;- local tests

&nbsp;- API

&nbsp;- Docker container

**9. API Deployment (FastAPI)**

The model is exposed as a REST API with:

&nbsp;- input validation (Pydantic)

&nbsp;- transparent output contract

&nbsp;- health check endpoint

**10. Dockerization**

The entire system is containerized for:

&nbsp;- portability

&nbsp;- reproducibility

&nbsp;- deployment readiness



##### **Repository Structure**

.

├── data/

│   ├── raw/

│   └── processed/

├── notebooks/

│   └── 01\_eda.ipynb

├── src/

│   ├── data/

│   ├── features/

│   ├── models/

│   └── utils/

├── app/

│   └── api.py

├── models/

│   ├── logistic\_model.pkl

│   ├── logistic\_model\_calibrated.pkl

│   └── decision\_policy.json

├── requirements.txt

├── Dockerfile

└── README.md



