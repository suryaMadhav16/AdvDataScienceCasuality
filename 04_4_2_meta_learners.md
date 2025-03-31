# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.4.2

#### 4.4.1 Meta-Learners

Meta-learners use machine learning algorithms to estimate treatment effects by combining multiple prediction models in different ways.

```python
# S-Learner (Single model)
def s_learner(X, T, Y, X_test=None, model=None):
    """
    Estimate treatment effects using S-Learner
    
    Parameters:
    -----------
    X : DataFrame of covariates
    T : Series of treatment assignments
    Y : Series of outcomes
    X_test : DataFrame of test covariates or None
    model : Sklearn model or None
    
    Returns:
    --------
    ate : Estimated average treatment effect
    cate : Estimated conditional average treatment effects
    """
    # Default model
    if model is None:
        model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    
    # Create combined dataset
    X_combined = X.copy()
    X_combined['treatment'] = T
    
    # Fit the model
    model.fit(X_combined, Y)
    
    # Predict on test set if provided, otherwise on training set
    if X_test is not None:
        X_pred = X_test
    else:
        X_pred = X
    
    # Create counterfactual datasets
    X_pred_1 = X_pred.copy()
    X_pred_1['treatment'] = 1
    
    X_pred_0 = X_pred.copy()
    X_pred_0['treatment'] = 0
    
    # Predict potential outcomes
    y_pred_1 = model.predict(X_pred_1)
    y_pred_0 = model.predict(X_pred_0)
    
    # Calculate treatment effects
    cate = y_pred_1 - y_pred_0
    ate = cate.mean()
    
    return ate, cate, model

# T-Learner (Two models)
def t_learner(X, T, Y, X_test=None, model_t=None, model_c=None):
    """
    Estimate treatment effects using T-Learner
    
    Parameters:
    -----------
    X : DataFrame of covariates
    T : Series of treatment assignments
    Y : Series of outcomes
    X_test : DataFrame of test covariates or None
    model_t : Sklearn model for treated group or None
    model_c : Sklearn model for control group or None
    
    Returns:
    --------
    ate : Estimated average treatment effect
    cate : Estimated conditional average treatment effects
    """
    # Default models
    if model_t is None:
        model_t = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    if model_c is None:
        model_c = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=43)
    
    # Split data into treated and control groups
    X_t = X[T == 1]
    Y_t = Y[T == 1]
    X_c = X[T == 0]
    Y_c = Y[T == 0]
    
    # Fit models
    model_t.fit(X_t, Y_t)
    model_c.fit(X_c, Y_c)
    
    # Predict on test set if provided, otherwise on training set
    if X_test is not None:
        X_pred = X_test
    else:
        X_pred = X
    
    # Predict potential outcomes
    y_pred_1 = model_t.predict(X_pred)
    y_pred_0 = model_c.predict(X_pred)
    
    # Calculate treatment effects
    cate = y_pred_1 - y_pred_0
    ate = cate.mean()
    
    return ate, cate, (model_t, model_c)

# X-Learner
def x_learner(X, T, Y, X_test=None, models_t=None, models_c=None, propensity_model=None):
    """
    Estimate treatment effects using X-Learner
    
    Parameters:
    -----------
    X : DataFrame of covariates
    T : Series of treatment assignments
    Y : Series of outcomes
    X_test : DataFrame of test covariates or None
    models_t : List of two Sklearn models for treated group or None
    models_c : List of two Sklearn models for control group or None
    propensity_model : Sklearn classifier for propensity scores or None
    
    Returns:
    --------
    ate : Estimated average treatment effect
    cate : Estimated conditional average treatment effects
    """
    # Default models
    if models_t is None:
        models_t = [
            RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
            RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=43)
        ]
    if models_c is None:
        models_c = [
            RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=44),
            RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=45)
        ]
    if propensity_model is None:
        propensity_model = LogisticRegression(max_iter=1000)
    
    # Split data into treated and control groups
    X_t = X[T == 1]
    Y_t = Y[T == 1]
    X_c = X[T == 0]
    Y_c = Y[T == 0]
    
    # Step 1: Estimate the response surfaces using the first stage models
    model_t1, model_c1 = models_t[0], models_c[0]
    model_t1.fit(X_t, Y_t)
    model_c1.fit(X_c, Y_c)
    
    # Predict responses for all units using both models
    mu_t = model_t1.predict(X)
    mu_c = model_c1.predict(X)
    
    # Step 2: Compute the imputed treatment effects
    D_t = Y_t - model_c1.predict(X_t)  # Imputed effect for treated units
    D_c = model_t1.predict(X_c) - Y_c  # Imputed effect for control units
    
    # Step 3: Estimate the CATE functions
    model_t2, model_c2 = models_t[1], models_c[1]
    model_t2.fit(X_t, D_t)
    model_c2.fit(X_c, D_c)
    
    # Step 4: Combine the CATE functions using propensity scores
    propensity_model.fit(X, T)
    g = propensity_model.predict_proba(X)[:, 1]  # Propensity scores
    
    # Predict on test set if provided, otherwise on training set
    if X_test is not None:
        X_pred = X_test
        g_pred = propensity_model.predict_proba(X_pred)[:, 1]
    else:
        X_pred = X
        g_pred = g
    
    # Predict treatment effects
    tau_t = model_t2.predict(X_pred)
    tau_c = model_c2.predict(X_pred)
    
    # Weighted average of treatment effects
    cate = g_pred * tau_c + (1 - g_pred) * tau_t
    ate = cate.mean()
    
    return ate, cate, (models_t, models_c, propensity_model)

# Implement meta-learners
np.random.seed(42)  # For reproducibility

# Initialize results list
meta_learner_results = []

# S-Learner with different base models
for model_name, model in [
    ('Random Forest', RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))
]:
    # Train S-Learner
    s_ate, s_cate, _ = s_learner(X_train_scaled, T_train, Y_train, model=model)
    
    # Save result
    meta_learner_results.append({
        'Method': f'S-Learner ({model_name})',
        'ATE': s_ate,
        'Bias': s_ate - true_ate_train,
        'Abs Bias': abs(s_ate - true_ate_train)
    })

# T-Learner with different base models
for model_name, model_t, model_c in [
    ('Random Forest', 
     RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
     RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=43)),
    ('Gradient Boosting', 
     GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
     GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=43))
]:
    # Train T-Learner
    t_ate, t_cate, _ = t_learner(X_train_scaled, T_train, Y_train, model_t=model_t, model_c=model_c)
    
    # Save result
    meta_learner_results.append({
        'Method': f'T-Learner ({model_name})',
        'ATE': t_ate,
        'Bias': t_ate - true_ate_train,
        'Abs Bias': abs(t_ate - true_ate_train)
    })

# X-Learner with different base models
for model_name, models_t, models_c, prop_model in [
    ('Random Forest', 
     [RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
      RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=43)],
     [RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=44),
      RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=45)],
     LogisticRegression(max_iter=1000)),
    ('Gradient Boosting', 
     [GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
      GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=43)],
     [GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=44),
      GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=45)],
     LogisticRegression(max_iter=1000))
]:
    # Train X-Learner
    x_ate, x_cate, _ = x_learner(X_train_scaled, T_train, Y_train, 
                               models_t=models_t, models_c=models_c, 
                               propensity_model=prop_model)
    
    # Save result
    meta_learner_results.append({
        'Method': f'X-Learner ({model_name})',
        'ATE': x_ate,
        'Bias': x_ate - true_ate_train,
        'Abs Bias': abs(x_ate - true_ate_train)
    })

# Convert to DataFrame and sort by absolute bias
meta_learner_df = pd.DataFrame(meta_learner_results)
meta_learner_df = meta_learner_df.sort_values('Abs Bias')

print("Meta-Learner Results:")
print(meta_learner_df)

# Visualize comparison
plt.figure(figsize=(12, 8))
plt.barh(y=meta_learner_df['Method'], width=meta_learner_df['ATE'], color='skyblue')
plt.axvline(x=true_ate_train, color='red', linestyle='--', label=f'True ATE = {true_ate_train:.4f}')
plt.title('Comparison of ATE Estimates from Meta-Learners')
plt.xlabel('ATE Estimate')
plt.ylabel('Method')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Select the best meta-learner method
best_ml_idx = meta_learner_df['Abs Bias'].idxmin()
best_ml = meta_learner_df.loc[best_ml_idx]

print(f"\nBest Meta-Learner Method:")
print(f"Method: {best_ml['Method']}")
print(f"ATE: {best_ml['ATE']:.4f}")
print(f"True ATE: {true_ate_train:.4f}")
print(f"Bias: {best_ml['Bias']:.4f}")

# Plot treatment effect heterogeneity for the best meta-learner
# Let's use X-Learner with Random Forest for this demonstration
_, best_cate, _ = x_learner(
    X_train_scaled, T_train, Y_train,
    models_t=[RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
             RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=43)],
    models_c=[RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=44),
             RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=45)],
    propensity_model=LogisticRegression(max_iter=1000)
)

# Create a DataFrame with CATE estimates and covariates
cate_df = X_train.copy()
cate_df['cate'] = best_cate
cate_df['treatment'] = T_train
cate_df['outcome'] = Y_train

# Plot CATE distribution
plt.figure(figsize=(10, 6))
sns.histplot(cate_df['cate'], bins=30, kde=True)
plt.axvline(x=cate_df['cate'].mean(), color='red', linestyle='--', 
            label=f'Mean CATE = {cate_df["cate"].mean():.4f}')
plt.axvline(x=true_ate_train, color='green', linestyle=':', 
            label=f'True ATE = {true_ate_train:.4f}')
plt.title('Distribution of Conditional Average Treatment Effects (CATE)')
plt.xlabel('CATE')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Analysis:** Meta-learners combine machine learning algorithms to estimate treatment effects in different ways:

1. **S-Learner (Single model)** uses one model with treatment as a feature. It's simple but may underestimate treatment effects when treatment assignment is highly imbalanced.

2. **T-Learner (Two models)** fits separate models for treated and control groups. It can capture heterogeneous treatment effects but may struggle with limited data in each group.

3. **X-Learner** extends T-Learner by directly modeling treatment effects. It performs well when treatment groups are imbalanced and there's heterogeneity in treatment effects.

The results show that meta-learners can provide accurate estimates of the average treatment effect (ATE). The choice of base learner (Random Forest, Gradient Boosting) affects performance significantly.

The distribution of Conditional Average Treatment Effects (CATE) shows how treatment effects vary across individuals. This heterogeneity is valuable for targeting interventions to those who would benefit most.

Meta-learners offer a flexible approach to causal inference that can capture complex relationships between covariates and outcomes, providing more accurate and nuanced estimates than traditional methods.