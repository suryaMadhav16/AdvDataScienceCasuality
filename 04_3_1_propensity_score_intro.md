# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.3.1

### 4.3 Propensity Score Methods

> 🎯 **Step 3**: Implement propensity score-based methods for causal inference

Propensity score methods estimate the probability of receiving treatment given covariates, then use these scores to adjust for confounding.

#### 4.3.1 Estimating Propensity Scores

```python
# Assuming the imports and data preparation from previous sections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, brier_score_loss

# Estimate propensity scores using different methods
def estimate_propensity_scores(X, T, method='logistic'):
    """
    Estimate propensity scores using various methods
    
    Parameters:
    -----------
    X : DataFrame of covariates
    T : Series of treatment assignments
    method : str, Method to use ('logistic' or 'rf')
    
    Returns:
    --------
    ps : Series of propensity scores
    model : Fitted model
    """
    if method == 'logistic':
        model = LogisticRegression(max_iter=1000, C=1.0)
    elif method == 'rf':
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model.fit(X, T)
    ps = model.predict_proba(X)[:, 1]
    return ps, model

# Estimate propensity scores for both methods
ps_logistic_train, ps_model_logistic = estimate_propensity_scores(X_train_scaled, T_train, 'logistic')
ps_rf_train, ps_model_rf = estimate_propensity_scores(X_train_scaled, T_train, 'rf')

# Add propensity scores to our data
train_data = X_train_scaled.copy()
train_data['treatment'] = T_train
train_data['outcome'] = Y_train
train_data['ps_logistic'] = ps_logistic_train
train_data['ps_rf'] = ps_rf_train

# Evaluate propensity score models
print("Propensity Score Model Evaluation:")
for name, ps in [('Logistic Regression', ps_logistic_train), ('Random Forest', ps_rf_train)]:
    auc = roc_auc_score(T_train, ps)
    ll = log_loss(T_train, ps)
    bs = brier_score_loss(T_train, ps)
    print(f"{name}: AUC = {auc:.4f}, Log Loss = {ll:.4f}, Brier Score = {bs:.4f}")

# Plot propensity score distributions
plt.figure(figsize=(12, 5))

# Plot 1: Propensity score distributions by treatment group
plt.subplot(1, 2, 1)
sns.histplot(data=train_data, x='ps_logistic', hue='treatment', bins=30, 
            element="step", common_norm=False, stat='density')
plt.title('Propensity Score Distributions\n(Logistic Regression)')
plt.xlabel('Propensity Score')
plt.ylabel('Density')

# Plot 2: Calibration curve
plt.subplot(1, 2, 2)
prob_true, prob_pred = calibration_curve(T_train, ps_logistic_train, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Logistic Regression')

prob_true_rf, prob_pred_rf = calibration_curve(T_train, ps_rf_train, n_bins=10)
plt.plot(prob_pred_rf, prob_true_rf, marker='s', linewidth=1, label='Random Forest')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Calibration Curve')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()

plt.tight_layout()
plt.show()

# Check for overlap/positivity assumption
def check_overlap(ps, T):
    """Check overlap assumption and trim propensity scores if needed"""
    # Calculate min and max propensity scores for treated and control
    treated_min = ps[T == 1].min()
    treated_max = ps[T == 1].max()
    control_min = ps[T == 0].min()
    control_max = ps[T == 0].max()
    
    # Find common support region
    common_min = max(treated_min, control_min)
    common_max = min(treated_max, control_max)
    
    # Calculate percentage of units in common support
    in_support = (ps >= common_min) & (ps <= common_max)
    pct_in_support = in_support.mean() * 100
    
    # Check for extreme propensity scores
    extreme_ps = (ps < 0.05) | (ps > 0.95)
    pct_extreme = extreme_ps.mean() * 100
    
    print(f"Common support region: [{common_min:.4f}, {common_max:.4f}]")
    print(f"Percentage of units in common support: {pct_in_support:.2f}%")
    print(f"Percentage of units with extreme propensity scores: {pct_extreme:.2f}%")
    
    return common_min, common_max, in_support

# Check overlap for logistic regression propensity scores
print("Checking overlap for logistic regression propensity scores:")
common_min, common_max, in_support_logistic = check_overlap(ps_logistic_train, T_train)

# Check overlap for random forest propensity scores
print("Checking overlap for random forest propensity scores:")
common_min_rf, common_max_rf, in_support_rf = check_overlap(ps_rf_train, T_train)
```

**Analysis:** Propensity score estimation is crucial for propensity score-based methods. We've used both logistic regression and random forest models to estimate the probability of treatment given covariates.

The propensity score distributions show how treatment probability varies between treated and control groups. Ideally, these distributions should overlap substantially, indicating that each unit has a non-zero probability of receiving either treatment (the positivity assumption).

The calibration curves show how well the predicted probabilities match the observed frequencies. Points closer to the diagonal represent better calibration. Well-calibrated propensity scores are important for methods like inverse probability weighting.

The overlap check helps identify regions of common support and extreme propensity scores. Units with very high or very low propensity scores may need special attention in the analysis.

Let's examine feature importance in the propensity score model:

```python
# Visualize important features for propensity score model
if hasattr(ps_model_logistic, 'coef_'):
    # For logistic regression, we can directly extract coefficients
    coef = ps_model_logistic.coef_[0]
    features = X_train_scaled.columns
    
    # Create DataFrame for visualization
    ps_coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coef,
        'Abs_Coefficient': np.abs(coef)
    })
    
    # Sort by absolute coefficient
    ps_coef_df = ps_coef_df.sort_values('Abs_Coefficient', ascending=False)
    
    # Plot top features
    plt.figure(figsize=(10, 8))
    plt.barh(y=ps_coef_df['Feature'].head(15), width=ps_coef_df['Coefficient'].head(15))
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.3)
    plt.title('Top 15 Features for Propensity Score Model')
    plt.xlabel('Coefficient')
    plt.tight_layout()
    plt.show()
```

**Analysis:** The feature importance plot shows which covariates are most predictive of treatment assignment. Features with larger coefficients have a stronger influence on the propensity score. Understanding these relationships is valuable for interpreting the selection mechanism and potential confounding.