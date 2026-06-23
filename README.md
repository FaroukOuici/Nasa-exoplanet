# 🌌 NASA Exoplanet Mass Regressor

This repository contains an **XGBoost Regressor** machine learning pipeline, along with a collection of space data retrieved from official NASA archives. The model is trained to predict an exoplanet's mass with excellent scientific accuracy ($R^2 \approx 89.5\%$) based on its physical properties and stellar hosting conditions.

---

## 🚀 How to Use the Script

1. **Data Acquisition:** The main script `training.py` is ready to run out of the box. You can modify the data volume in the initial data-fetching function, which queries the NASA Exoplanet Archive via the `astroquery` library. Simply change the row limit parameter to download your desired dataset size (subject to server/API limits).
2. **Domain Adaptation:** If you want to adapt this architecture for other astrophysical structures like black holes or galaxies, you just need to change the data source. *(Please note: transferring the task to black holes requires rewriting the target features/labels, though the underlying gradient boosting architecture remains unchanged).*
3. **Hyperparameter Tuning:** The current configuration is highly optimized but fully adjustable within the training function:
   * `n_estimators`: The total number of sequential decision trees built.
   * `max_depth`: Limits how deep an individual tree can grow (increasing this enhances the risk of *Overfitting*).
   * `subsample`: Dictates that each subsequent tree trains on a random $80\%$ subset of the exoplanet rows to prevent outlier domination.
   * `colsample_bytree`: Dictates that each tree randomly selects $80\%$ of the physical features (columns) to evaluate splits, enforcing structural diversity.

---

## 🛠️ Deep Architectural Breakdown

### 1. Mean Squared Error (MSE) Loss Function

In regression tasks, we monitor and minimize the Mean Squared Error:

$$L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$$

* **$\hat{y}$ (The Model Prediction):** The target log-mass value estimated by the collective tree ensemble.
* **$y$ (The True Label):** The actual observed planetary log-mass from NASA's baseline measurements.
* **The Square Power $(\cdot)^2$:** This eliminates negative signs (preventing errors in opposite directions from canceling each other out) and heavily penalizes large deviation faults (e.g., a residual of $2$ scales to a penalty of $4$).
* **The Constant Fraction $\frac{1}{2}$:** A purely mathematical convenience that cancels out with the power of $2$ during partial differentiation ($\frac{\partial}{\partial \hat{y}}$), yielding a clean derivative.

### 2. Gradient Boosting vs. Classical Gradient Descent

Unlike traditional linear models that adjust a set of fixed coefficients/weights ($w_i$) globally using basic Gradient Descent:

$$\Delta w_i = - \alpha \frac{\partial L}{\partial w_i}$$

`XGBoost` builds an ensemble of non-linear decision trees sequentially. Instead of updating global weights, the model calculates the **Partial Derivative** (Gradient $g_i$) of the loss function with respect to the *previous step's total prediction* ($\hat{y}^{t-1}$):

$$g_i = \frac{\partial L(y_i, \hat{y}^{t-1})}{\partial \hat{y}^{t-1}} = - (y_i - \hat{y}^{t-1})$$

The algorithm uses these calculated gradients to establish the perfect structural splits and compute the optimal target weights stored at the **leaves** of the newly added tree, effectively neutralizing the remaining residual errors.

---

## 🎯 Explaining the 89% Validation Accuracy ($R^2$)

Achieving an $R^2 \text{ score} = 0.8949$ is due to rigorous data cleaning, logarithmic scaling to compress extreme astronomical scales, and strict prevention of **Data Leakage**.

### What is Data Leakage?
Data leakage occurs when target or future information accidentally slips into the training feature matrix ($X$). If your model yields an unrealistic $>99\%$ accuracy on raw celestial data, check for these fatal flaws:
* **Proxy Column Inclusion:** Passing raw parameters that directly correlate or mathematically compose the target mass (e.g., leaving a raw gravity/density column when trying to predict mass).
* **Target Mirroring:** Accidentally failing to drop the target column or its direct operational variant from the training dataframe.

---

## 🌌 Epilogue

This pipeline aims to help anyone passionate about astronomy, astrophysics, and computational data science to reliably model stellar and planetary interactions. Remember: **You can build a functional model without understanding its math, but you cannot debug its failures without understanding its mechanics.**

