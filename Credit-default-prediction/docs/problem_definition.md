# **PROBLEM DEFINITION**

## **Business Context**

A financial institution issues short- to medium-term credit to customers. Some customers default (fail to repay), causing financial loss.

The institution wants to:
- Minimize losses from defaults (not repaying loan)
- Avoid unnecessarily rejecting good customers 
- Make risk-aware, probabilistic decisions, not hard yes/no guesses

## **Business Objective**

Estimate the probability that a customer will default on their credit obligation, and make an approval/rejection decision based on expected cost.

**Key emphasis:**
- Probability, not just class label
- Decision informed by asymmetric business costs (not flagging defaulter != flagging non-defualter)

## **Machine Learning Problem Formulation**

**Task Type:**

Supervised binary classification

**Target Variable**
- default = 1  >> customer defaults
- default = 0  >> customer does not default

**Input Features (High-level)**
- Demographic attributes (e.g., age, education)
- Financial capacity (credit limit, bill amounts)
- Behavioral indicators (repayment history, payment delays)

**Model Output:**
- P(default | customer_features) >> [0,1]
- Decision rule based on optimized threshold (not default 0.5)

## **Constraints & Assumptions**

**Constraints:**
- Class imbalance expected (defaults are minority)
- Misclassification costs are asymmetric (not flagging defaulter != flagging non-defualter)
- Dataset is static and tabular (no time series modeling yet)

**Assumptions:**
- Historical repayment behavior is predictive
- Data reflects real decision-time information (no leakage)
- Labels are reliable

## **Risk Mitigation**
| Risk                        | Mitigation                        |
| --------------------------- | --------------------------------- |
| Severe class imbalance      | Class weighting, threshold tuning |
| Overconfident probabilities | Calibration + evaluation          |
| Data leakage                | Strict feature audit              |
| Metric gaming               | Cost-based evaluation             |

