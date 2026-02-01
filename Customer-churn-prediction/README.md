# **Customer Churn Prediction — End-to-End ML System**

## **Project Overview**

This project implements a full end-to-end machine learning system for predicting customer churn.

It goes beyond model training to include uncertainty estimation, cost-sensitive decision making, calibration, deployment, and containerization.

**The system outputs:**

* a probability of churn
* a business-aligned decision based on an optimized threshold

The entire workflow is reproducible, auditable, and deployable.

## **Problem Statement**

Customer churn is costly. The objective is to:

&nbsp;- Identify customers at risk of churning early enough to take preventive action, while minimizing unnecessary interventions.



This is framed as a binary classification problem:

&nbsp;- 1 → customer will churn

&nbsp;- 0 → customer will not churn

The key challenge is not classification accuracy, but balancing false positives vs false negatives under asymmetric business costs.

## **Key Design Principles**
- Separation of concerns
- training ≠ evaluation ≠ inference ≠ deployment
- Probabilities first, decisions second
- Cost-aware thresholding
- Calibration before deployment
- Single source of truth for inference


## **Dataset**

&nbsp;- Source: Telco Customer Churn dataset (Kaggle)

&nbsp;- Target variable: churn

&nbsp;- Observations: ~7,000 customers

&nbsp;- Feature types:

&nbsp;	Numerical (tenure, charges)

&nbsp;	Binary indicators

&nbsp;	Categorical features (encoded during preprocessing)

The cleaned dataset is stored in:

&nbsp;- data/processed/churn\_clean.csv



## **Machine Learning Workflow**

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



## **Repository Structure**
```bash 
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
```
## **How to run the Project**
### **Create and activate virtual env**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate  # Linux / macOS
```
### **Install dependencies**
```bash
pip install -r requirements.txt
```
### **Run the API**
```bash
python -m uvicorn app.api:app --reload
```
### **Open in browser**
```bash
http://127.0.0.1:8000/docs
```
### **API Usage**
After visiting the address above, follow steps below:
1. Click POST /predict
2. Click Try it out
3. Paste JSON input
4. Click Execute
for step-3, I have provided a sample input data in the ./data/raw/test.json,
you can copy and paste directly
you can also generate new examples by using the notebook (01_eda.ipynb), i have provided a code in the last section of the notbook which you can change the index values to get new input example.
```bash
exmp = df.iloc[2].to_dict() # change [2] to another index and run the cell to generate new example in the test.json file
```
### **Running with Docker**
```bash 
docker build -t churn-api . #build image
docker run -p 8000:8000 churn-api # Run the container
http://127.0.0.1:8000/docs # access through browser
```

## **Limitations & Future work**
 - Add monitoring and data-drift detection
 - Periodic recalibration under distribution shift
 - CI/CD pipeline
 - Batch inference jobs
 - Extend to other supervised tasks (regression, time-series)
