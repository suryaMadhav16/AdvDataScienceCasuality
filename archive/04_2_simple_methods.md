# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.2

### 4.2 Simple Methods for Causal Inference

> 🔄 **Step 2**: Start with simple methods before moving to more complex approaches

Let's begin with simple methods for estimating causal effects before moving to more advanced techniques.

#### 4.2.1 Naive Mean Difference

The simplest approach is to directly compare the mean outcomes between treated and control groups without adjusting for covariates.

```python
# Calculate naive mean difference
def naive_estimator(T, Y):
    """Calculate naive mean difference between treated and control outcomes"""
    treated_mean = Y[T == 1].mean()
    control_mean = Y[T == 0].mean()
    ate = treated_mean - control_mean
    return ate

# Calculate on training set
naive_ate_train = naive_estimator(T_train, Y_train)

# Calculate true ATE for comparison
true_ate_train = (Y1_train - Y0_train).mean()

print(f"Naive ATE estimate (training): {naive_ate_train:.4f}")
print(f"True ATE (training): {true_ate_train:.4f}")
print(f"Bias: {naive_ate_train - true_ate_train:.4f}")

# Visualize the distribution of outcomes by treatment group
plt.figure(figsize=(10, 6))
sns.kdeplot(data=pd.DataFrame({'Outcome': Y_train, 'Treatment': T_train}), 
            x='Outcome', hue='Treatment', common_norm=False, fill=True, alpha=0.5)
plt.axvline(x=Y_train[T_train == 1].mean(), color='blue', linestyle='--', 
            label=f'Treated Mean: {Y_train[T_train == 1].mean():.2f}')
plt.axvline(x=Y_train[T_train == 0].mean(), color='orange', linestyle='--', 
            label=f'Control Mean: {Y_train[T_train == 0].mean():.2f}')
plt.title('Distribution of Outcomes by Treatment Group')
plt.xlabel('Outcome')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Analysis:** The naive estimator simply calculates the difference in means between the treated and control groups. Due to selection bias in the IHDP dataset, this estimate is likely biased. The bias represents how far our estimate is from the true causal effect.

The visualization shows the distribution of outcomes for the treated and control groups. The difference between the dashed lines represents the naive ATE estimate. However, this doesn't account for confounding variables that might influence both treatment assignment and outcomes.

#### 4.2.2 Regression Adjustment

Next, let's use regression adjustment, which controls for covariates by including them in a regression model.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def regression_adjustment(X, T, Y):
    """
    Estimate ATE using regression adjustment
    
    Parameters:
    -----------
    X : DataFrame of covariates
    T : Series of treatment assignments
    Y : Series of outcomes
    
    Returns:
    --------
    ate : Estimated average treatment effect
    model : Fitted model
    """
    # Create dataframe with treatment and covariates
    data = X.copy()
    data['treatment'] = T
    
    # Fit the model
    model = LinearRegression()
    model.fit(data, Y)
    
    # Get the treatment coefficient
    treatment_idx = data.columns.get_loc('treatment')
    ate = model.coef_[treatment_idx]
    
    return ate, model

# Estimate ATE using regression adjustment
reg_ate_train, reg_model = regression_adjustment(X_train_scaled, T_train, Y_train)

print(f"Regression adjustment ATE estimate: {reg_ate_train:.4f}")
print(f"True ATE (training): {true_ate_train:.4f}")
print(f"Bias: {reg_ate_train - true_ate_train:.4f}")

# Assess model fit
y_pred = reg_model.predict(pd.concat([X_train_scaled, pd.Series(T_train, name='treatment')], axis=1))
mse = mean_squared_error(Y_train, y_pred)
r2 = r2_score(Y_train, y_pred)

print(f"Model MSE: {mse:.4f}")
print(f"Model R²: {r2:.4f}")

# Visualize residuals
residuals = Y_train - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)
plt.show()

# Plot feature importance
coef_df = pd.DataFrame({
    'Feature': list(X_train_scaled.columns) + ['treatment'],
    'Coefficient': list(reg_model.coef_)
})
coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(10))
plt.title('Top 10 Features by Coefficient Magnitude')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.axvline(x=0, color='red', linestyle='--')
plt.grid(True, alpha=0.3)
plt.show()
```

**Analysis:** The regression adjustment method estimates the treatment effect by including the treatment indicator and covariates in a regression model. The coefficient of the treatment variable represents the estimated ATE. This method reduces bias compared to the naive approach because it controls for observed confounders.

The residual plot helps us assess the model fit. Ideally, the residuals should be randomly scattered around zero with no clear pattern. Any pattern might indicate model misspecification.

The feature importance plot shows which covariates have the strongest association with the outcome. This gives us insights into which variables are important predictors and potentially important confounders.

#### 4.2.3 Stratification by Covariate

Another simple approach is to stratify the data by important covariates and estimate the treatment effect within each stratum.

```python
def stratified_estimator(X, T, Y, stratify_var, n_bins=5):
    """
    Estimate ATE by stratifying on a continuous covariate
    
    Parameters:
    -----------
    X : DataFrame of covariates
    T : Series of treatment assignments
    Y : Series of outcomes
    stratify_var : str, Name of variable to stratify on
    n_bins : int, Number of bins for stratification
    
    Returns:
    --------
    ate : Estimated average treatment effect
    """
    # Create bins for the stratification variable
    data = pd.DataFrame({
        'X': X[stratify_var],
        'T': T,
        'Y': Y
    })
    
    data['bin'] = pd.qcut(data['X'], n_bins, labels=False)
    
    # Calculate stratified ATE
    stratum_effects = []
    stratum_sizes = []
    
    for bin_idx in range(n_bins):
        bin_data = data[data['bin'] == bin_idx]
        if (bin_data['T'] == 1).sum() > 0 and (bin_data['T'] == 0).sum() > 0:
            treated_mean = bin_data[bin_data['T'] == 1]['Y'].mean()
            control_mean = bin_data[bin_data['T'] == 0]['Y'].mean()
            effect = treated_mean - control_mean
            stratum_effects.append(effect)
            stratum_sizes.append(len(bin_data))
    
    # Weight by stratum size
    if stratum_effects:
        weighted_ate = sum(effect * size for effect, size in zip(stratum_effects, stratum_sizes)) / sum(stratum_sizes)
        return weighted_ate, stratum_effects, stratum_sizes
    else:
        return np.nan, [], []

# Choose an important continuous covariate for stratification
# We'll use birth weight (x_0) as it's likely an important confounder
stratified_ate, stratum_effects, stratum_sizes = stratified_estimator(X_train, T_train, Y_train, 'x_0', n_bins=5)

print(f"Stratified ATE estimate (by birth weight): {stratified_ate:.4f}")
print(f"True ATE (training): {true_ate_train:.4f}")
print(f"Bias: {stratified_ate - true_ate_train:.4f}")

# Try stratifying by another variable (mother's education)
stratified_ate2, stratum_effects2, stratum_sizes2 = stratified_estimator(X_train, T_train, Y_train, 'x_4', n_bins=5)

print(f"Stratified ATE estimate (by mother's education): {stratified_ate2:.4f}")
print(f"Bias: {stratified_ate2 - true_ate_train:.4f}")

# Visualize the stratum-specific effects
plt.figure(figsize=(12, 6))

# Plot 1: Effects by birth weight stratum
plt.subplot(1, 2, 1)
plt.bar(range(len(stratum_effects)), stratum_effects, alpha=0.7)
plt.axhline(y=stratified_ate, color='red', linestyle='--', label=f'Overall ATE: {stratified_ate:.4f}')
plt.axhline(y=true_ate_train, color='green', linestyle=':', label=f'True ATE: {true_ate_train:.4f}')
plt.title('Treatment Effects by Birth Weight Stratum')
plt.xlabel('Birth Weight Stratum (low to high)')
plt.ylabel('Stratum-Specific ATE')
plt.legend()

# Plot 2: Effects by mother's education stratum
plt.subplot(1, 2, 2)
plt.bar(range(len(stratum_effects2)), stratum_effects2, alpha=0.7)
plt.axhline(y=stratified_ate2, color='red', linestyle='--', label=f'Overall ATE: {stratified_ate2:.4f}')
plt.axhline(y=true_ate_train, color='green', linestyle=':', label=f'True ATE: {true_ate_train:.4f}')
plt.title('Treatment Effects by Mother\'s Education Stratum')
plt.xlabel('Mother\'s Education Stratum (low to high)')
plt.ylabel('Stratum-Specific ATE')
plt.legend()

plt.tight_layout()
plt.show()
```

**Analysis:** Stratification divides the data into subgroups based on an important covariate and estimates the treatment effect within each subgroup. The overall estimate is a weighted average of these stratum-specific effects. This method helps account for the confounding effect of the stratification variable but doesn't adjust for other confounders.

The visualizations show how the treatment effect varies across different strata of birth weight and mother's education. This heterogeneity suggests that the treatment effect might depend on these covariates, which is valuable information for targeted interventions.

#### 4.2.4 Comparing Simple Methods

Let's compare the performance of these simple methods:

```python
# Compile results from simple methods
simple_methods = pd.DataFrame({
    'Method': ['Naive Mean Difference', 'Regression Adjustment', 
               'Stratification (Birth Weight)', 'Stratification (Mother\'s Education)'],
    'ATE Estimate': [naive_ate_train, reg_ate_train, stratified_ate, stratified_ate2],
    'True ATE': [true_ate_train] * 4,
    'Bias': [naive_ate_train - true_ate_train, 
             reg_ate_train - true_ate_train, 
             stratified_ate - true_ate_train, 
             stratified_ate2 - true_ate_train],
    'Absolute Bias': [abs(naive_ate_train - true_ate_train), 
                     abs(reg_ate_train - true_ate_train), 
                     abs(stratified_ate - true_ate_train), 
                     abs(stratified_ate2 - true_ate_train)]
})

# Sort by absolute bias
simple_methods = simple_methods.sort_values('Absolute Bias')
print("Comparison of Simple Methods:")
print(simple_methods)

# Visualize the comparison
plt.figure(figsize=(10, 6))
plt.barh(y=simple_methods['Method'], width=simple_methods['ATE Estimate'], color='skyblue')
plt.axvline(x=true_ate_train, color='red', linestyle='--', label=f'True ATE = {true_ate_train:.4f}')
plt.xlabel('ATE Estimate')
plt.title('Comparison of ATE Estimates from Simple Methods')
plt.legend()
plt.tight_layout()
plt.show()
```

**Analysis:** This comparison helps us understand how different simple methods perform in estimating the causal effect. Methods with lower bias provide estimates closer to the true ATE. The visualization makes it easy to see which methods are most accurate.

These simple methods provide a foundation for understanding causal inference. However, they have limitations, especially when dealing with complex confounding. In the next section, we'll explore more advanced methods that can provide more robust causal estimates.

#### References and Resources

- Imbens, G. W., & Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences. Cambridge University Press.
- [Causal Inference in Python: Simple Methods](https://matheusfacure.github.io/python-causality-handbook/03-Stats-Tools-for-Causal-Inference.html)
- [Introduction to Causal Inference: Regression Adjustment](https://www.bradyneal.com/causal-inference-course)