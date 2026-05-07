# **Credit Default Prediction System:** A production-oriented, end-to-end binary classification system with monitoring and automated retraining capabilities.

## **Project Overview**
in this project we implemented a production-grade full end-to-end machine learning system for predicting credit default risk using supervised binary classification. The system implements the folloction function beyond notebook-style:
- Reproducible data pipelines
- Feature engineering pipelines
- Multi-model experimentation
- Statistical evaluation with uncertainty
- Cost-sensitive threshold optimization
- FastAPI deployment
- Monitoring and drift detection
- Automatic retraining orchestration
- Model versioning and registry management
The final deployed system predicts the probability that a customer will default on credit repayment and returns an approval/rejection decision based on optimized business cost.

## **Business Problem**

Financial institutions face significant losses when borrowers fail to repay credit obligations.

The objective of this system is to:

- Estimate the probability of customer default
- Minimize financial loss from missed defaulters
- Avoid unnecessary rejection of good customers
- Support risk-aware decision making

This project treats the problem as a cost-sensitive binary classification task.

## **Machine Learning Formulation**
the credit defualt problem is formulated as machine learning problem as follows:

**Problem Type**
Binary Classification (customer will either defaul or not-default)

**Target Variable**
```bash 
0 : Non-default
1 : Default
```
**Model Output**
```bash 
P(default | customer features) 
```

**Final Decision Rule**
```bash 
Reject customer if:
P(default) >= threshold
```
The threshold is optimized using business cost rather than arbitrary 0.5 classification.

## **Dataset**
**Dataset Name:** Default of Credit Card Clients Dataset

**Source:** UCI Machine Learning Repository

**Dataset Characteristics**
|---------|----------------------------|
|Property |	Value                      |
|---------|----------------------------|
|Rows	  | 30,000                     |
|---------|----------------------------|
|Features |	23                         |
|---------|----------------------------|
|Target	  | default.payment.next.month |
|---------|----------------------------|
|Domain	  | Credit Risk                |
|---------|----------------------------|
|Problem Type |	Binary Classification  |
|-------------|------------------------|

## **Project Flow**
```bash
Data Ingestion
      ↓
EDA
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Evaluation + Uncertainty
      ↓
Threshold Optimization
      ↓
Deployment API
      ↓
Monitoring
      ↓
Drift Detection
      ↓
Automatic Retraining
      ↓
Model Versioning
```
## **Reposiroty Structure**
```bash
project-credit-default/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   ├── models/
│   ├── registry/
│   ├── selections/
│   ├── evaluation_results.json
│   ├── model_selection.json
│   ├── inference_logs.csv
│   └── feature_pipeline.joblib
│
├── src/
│   ├── api/
│   ├── data/
│   ├── eda/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   ├── retraining/
│   └── versioning/
│
├── config.yaml
├── requirements.txt
└── README.md
```

## **Models Implemented**
* **Logistic Regression:** Used as interpretable probabilistic baseline.
* **Decision Tree** Introduced non-linear rule learning.
* **Random Forest** Used ensemble learning for stronger ranking ability.

## **Training Design**
- Shared preprocessing pipeline
- Consistent feature space across models
- Class weighting for imbalance handling
- Conservative tree depth to avoid overfitting

## **Evaluation Metrics Used**
* **ROC-AUC** for ranking quality
* **Precision** for	false positive control
* **Recall** for default capture
* **Bootstrap CI** for	statistical uncertainty estimation

## **Threshold Optimization**
Because the default threshold of 0.5 is arbitrary, we implement an optimized theshold that fit with the business goals

**Business cost matrix**
The cost of missing defaulter is 5x more than rejecting a good customer
|----------------|----------------------|
| Error type     | Cost                 |
|----------------|----------------------|
| False Positive | 1                    |
|----------------|----------------------|
| False Negative | 5                    |
|----------------|----------------------|

## **Deployment API**
The project exposes a production-ready FastAPI service.
**End point:** POST /predict, and additional two (GET /health and GET /metadata)
**Input:** JSON request body containing 23 customer features
**Output** the output in the format:
```bash 
{
  "model_name": "random_forest_v2",
  "default_probability": 0.31,
  "decision": "reject",
  "threshold_used": 0.1725
}
```
## **Monitoring and Iteration**
The system implements an operational monitoring for drift detection using training and live inference distributions.

**Inference logging:** each prediction logs timestamp, model version, probability, decision, and input features

**Drift detection:** the system then compares training distribustions and live inference distributions to detect *input drift*, *probability drift*, and *distribution shift*

**Threshold Monitoring**
Thresholds is recalibrated based on:
- default rate changes
- business costs change
- calibration degredation

## **Automatic Retraining**
The implementation includes automated retraining archestration triggered by any of the following conditions:
- prediction drift
- data distribution shift
- performance degredation 
- accumulation of new data

**Retraining flow** 
```bash 
Drift Detection
      ↓
Feature Rebuild
      ↓
Model Retraining
      ↓
Evaluation
      ↓
Threshold Optimization
      ↓
Production Registration
```

## **Model Versioning**
The implementation includes lightweight model registry management for model version control. the system include features such as:
- versioned models
- deployment lineage
- rollback capability
- production model tracking
The registry tracks:
- model version
- timestamp
- ROC-AUC
- expected cost
- active production model

## **Techonologies Used**
|Category	         | Tools           |
|--------------------|-----------------|
| Language	         | Python          |
| ML	             | scikit-learn    |
| Data	             | pandas, numpy   |
| API	             | FastAPI         |       
| Serialization	     | joblib          |
| Visualization	     | matplotlib      |
| Validation	     | Pydantic        |
|--------------------|-----------------|

## **Installation Steps:**
1. Clone master repository 
```bash
git clone https://github.com/mbrk-tofa/Machine-Learning-Projects.git
```
2. move to project root
```bash
cd Credit-default-prediction
```
3. run the full pipeline script
```bash
python run_pipeline.py
```