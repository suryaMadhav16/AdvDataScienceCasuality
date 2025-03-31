# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.4.3

#### 4.4.2 Doubly Robust Methods

Doubly robust methods combine outcome modeling and propensity score weighting, providing protection against misspecification of either model.

```python
# Implement doubly robust estimator (AIPW)
def doubly_robust_estimator(X, T, Y, outcome_model=None, propensity_model=None):
    """
    Estimate ATE using doubly robust estimation (AIPW)
    
    Parameters:
    -----------
    X : DataFrame of covariates
    T : Series of treatment assignments
    Y : Series of outcomes
    outcome_model : Sklearn regressor or None
    propensity_model : Sklearn classifier or None
    
    Returns:
    --------
    ate : Estimated average treatment effect
    """
    # Default models
    if outcome_model is None:
        outcome_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    if propensity_model is None:
        propensity_model = LogisticRegression(max_iter=1000)
    
    # Split data into treated and control groups
    X_t = X[T == 1]
    Y_t = Y[T == 1]
    X_c = X[T == 0]
    Y_c = Y[T == 0]
    
    # Fit outcome models
    outcome_model_t = clone(outcome_model)
    outcome_model_c = clone(outcome_model)
    outcome_model_t.fit(X_t, Y_t)
    outcome_model_c.fit(X_c, Y_c)
    
    # Fit propensity model
    propensity_model.fit(X, T)
    ps = propensity_model.predict_proba(X)[:, 1]
    
    # Predict potential outcomes for all units
    mu_1 = outcome_model_t.predict(X)
    mu_0 = outcome_model_c.predict(X)
    
    # Calculate AIPW estimator
    aipw_1 = mu_1 + T * (Y - mu_1) / ps
    aipw_0 = mu_0 + (1 - T) * (Y - mu_0) / (1 - ps)
    
    # Calculate ATE
    ate = (aipw_1 - aipw_0).mean()
    
    return ate

# Implement double machine learning (DML)
def double_machine_learning(X, T, Y, outcome_model=None, propensity_model=None, cv=5):
    """
    Estimate ATE using Double Machine Learning (DML)
    
    Parameters:
    -----------
    X : DataFrame of covariates
    T : Series of treatment assignments
    Y : Series of outcomes
    outcome_model : Sklearn regressor or None
    propensity_model : Sklearn classifier or None
    cv : int, Number of cross-validation folds
    
    Returns:
    --------
    ate : Estimated average treatment effect
    """
    from sklearn.model_selection import KFold
    
    # Default models
    if outcome_model is None:
        outcome_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    if propensity_model is None:
        propensity_model = LogisticRegression(max_iter=1000)
    
    # Initialize cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Initialize arrays for predictions
    n = len(Y)
    Y_pred = np.zeros(n)
    T_pred = np.zeros(n)
    
    # Cross-fitting
    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        T_train, T_test = T.iloc[train_idx], T.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        
        # Fit propensity model
        propensity_model_cv = clone(propensity_model)
        propensity_model_cv.fit(X_train, T_train)
        T_pred[test_idx] = propensity_model_cv.predict_proba(X_test)[:, 1]
        
        # Fit outcome model
        outcome_model_cv = clone(outcome_model)
        outcome_model_cv.fit(X_train, Y_train)
        Y_pred[test_idx] = outcome_model_cv.predict(X_test)
    
    # Calculate residuals
    T_resid = T - T_pred
    Y_resid = Y - Y_pred
    
    # Estimate ATE
    ate = (T_resid * Y_resid).sum() / (T_resid * T).sum()
    
    return ate

# Implement doubly robust methods with different base models
dr_results = []

for method_name, outcome_model, propensity_model in [
    ('AIPW (RF, Logistic)', 
     RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
     LogisticRegression(max_iter=1000)),
    ('AIPW (GB, Logistic)', 
     GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
     LogisticRegression(max_iter=1000)),
    ('AIPW (RF, RF)', 
     RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
     RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42))
]:
    # Estimate ATE using AIPW
    aipw_ate = doubly_robust_estimator(
        X_train_scaled, T_train, Y_train,
        outcome_model=outcome_model,
        propensity_model=propensity_model
    )
    
    # Save result
    dr_results.append({
        'Method': method_name,
        'ATE': aipw_ate,
        'Bias': aipw_ate - true_ate_train,
        'Abs Bias': abs(aipw_ate - true_ate_train)
    })

for method_name, outcome_model, propensity_model in [
    ('DML (RF, Logistic)', 
     RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
     LogisticRegression(max_iter=1000)),
    ('DML (GB, Logistic)', 
     GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
     LogisticRegression(max_iter=1000)),
    ('DML (RF, RF)', 
     RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
     RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42))
]:
    # Estimate ATE using DML
    dml_ate = double_machine_learning(
        X_train_scaled, T_train, Y_train,
        outcome_model=outcome_model,
        propensity_model=propensity_model
    )
    
    # Save result
    dr_results.append({
        'Method': method_name,
        'ATE': dml_ate,
        'Bias': dml_ate - true_ate_train,
        'Abs Bias': abs(dml_ate - true_ate_train)
    })

# Convert to DataFrame and sort by absolute bias
dr_df = pd.DataFrame(dr_results)
dr_df = dr_df.sort_values('Abs Bias')

print("Doubly Robust Method Results:")
print(dr_df)

# Visualize comparison
plt.figure(figsize=(12, 8))
plt.barh(y=dr_df['Method'], width=dr_df['ATE'], color='skyblue')
plt.axvline(x=true_ate_train, color='red', linestyle='--', label=f'True ATE = {true_ate_train:.4f}')
plt.title('Comparison of ATE Estimates from Doubly Robust Methods')
plt.xlabel('ATE Estimate')
plt.ylabel('Method')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Select the best doubly robust method
best_dr_idx = dr_df['Abs Bias'].idxmin()
best_dr = dr_df.loc[best_dr_idx]

print(f"\nBest Doubly Robust Method:")
print(f"Method: {best_dr['Method']}")
print(f"ATE: {best_dr['ATE']:.4f}")
print(f"True ATE: {true_ate_train:.4f}")
print(f"Bias: {best_dr['Bias']:.4f}")
```

**Analysis:** Doubly robust methods combine outcome modeling and propensity score approaches, providing protection against misspecification of either model:

1. **Augmented Inverse Probability Weighting (AIPW)** combines outcome regression and IPW by using the outcome model to impute missing potential outcomes and IPW to correct for residual confounding.

2. **Double Machine Learning (DML)** uses cross-fitting to address issues of overfitting and regularization bias. It first fits models for the outcome and propensity score, then uses the residuals to estimate the treatment effect.

These methods are considered "doubly robust" because they provide consistent estimates if either the outcome model or the propensity score model is correctly specified (but not necessarily both). This robustness is especially valuable when we're uncertain about the true data-generating process.

Key advantages of doubly robust methods:

- Provide more robust estimates with less bias
- Can leverage flexible machine learning models without compromising consistency
- Achieve faster convergence rates
- Handle high-dimensional data more effectively

The results show how different doubly robust methods compare to the true ATE. The choice of base models (Random Forest, Gradient Boosting, Logistic Regression) affects performance significantly. In general, doubly robust methods often outperform simpler methods, especially when the relationships between covariates, treatment, and outcome are complex.