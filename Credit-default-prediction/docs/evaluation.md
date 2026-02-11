#  Model Comparison — Proper Interpretation

## First Step

### 🔹 ROC-AUC (Ranking Ability)

| Model               | ROC-AUC   | 95% CI         |
| ------------------- | --------- | -------------- |
| Logistic Regression | **0.711** | [0.694, 0.728] |
| Decision Tree       | **0.751** | [0.735, 0.767] |
| Random Forest       | **0.773** | [0.758, 0.789] |

### Interpretation

* Random Forest is best at ranking risk.
* Decision Tree is second.
* Logistic Regression lags significantly.
* Confidence intervals **barely overlap**, especially between RF and LR.

This suggests:

> The performance difference is statistically meaningful.

✔ Random Forest wins on discrimination.

---

## 🔹 Precision vs Recall (At Threshold = 0.5)

Now this is where it gets interesting.

---

### Logistic Regression

* Precision: **0.368**
* Recall: **0.632**

Interpretation:

* Finds many defaulters
* But many false positives
* Conservative toward risk detection

This is typical of linear models under class weighting.

---

### Decision Tree

* Precision: **0.670**
* Recall: **0.338**

Interpretation:

* When it predicts default, it's usually correct
* But it misses many defaulters
* Conservative toward approval

---

### Random Forest

* Precision: **0.669**
* Recall: **0.318**

Interpretation:

* Very precise
* Very selective
* Similar to tree but slightly stronger ranking

---

# 3️⃣ The Important Insight

At threshold = 0.5:

* Logistic Regression favors **recall**
* Trees favor **precision**
* Random Forest has best ranking ability but low recall at 0.5

But 0.5 is arbitrary.

You have not optimized threshold yet.

This means:

> These precision/recall comparisons are not final conclusions.

---

# 4️⃣ Which Model Is “Best”?

At this stage, the correct answer is:

> Random Forest is the strongest base model due to highest ROC-AUC and non-overlapping confidence interval superiority.

But:

* Logistic regression may be better calibrated.
* Business cost may favor recall over precision.
* Threshold tuning will change everything.

We cannot finalize selection yet.

---

# 5️⃣ Stability Assessment (Using CI Width)

Look at CI widths:

* RF CI width ≈ 0.030
* Tree CI width ≈ 0.031
* Logistic CI width ≈ 0.034

All are reasonably tight.

This suggests:

> Dataset size is sufficient. Estimates are stable.

No red flags.

---

# 6️⃣ Engineering-Level Conclusion

At this point:

✔ Random Forest → best discriminator
✔ Decision Tree → interpretable, moderate performer
✔ Logistic Regression → weaker ranking but high recall

But none are business-optimized yet.

We have:

* A ranking engine
* Not a decision engine

---

# NEXT STEP(Threshold Optimization & Model Selection)

we will:

* Define cost matrix
* Compute expected loss
* Find optimal threshold
* Possibly select model based on business cost (not ROC)

This is where the project becomes production-grade.

---

If you’re ready:

Reply:

> **step 7**

Now we enter decision theory.
