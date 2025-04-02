# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.3.2

#### 4.3.2 Inverse Probability Weighting (IPW)

IPW creates a pseudo-population where treatment assignment is independent of covariates by weighting observations inversely to their propensity scores.

```python
def ipw_estimator(T, Y, ps, stabilized=True, trimming=None):
    """
    Estimate ATE using inverse probability weighting
    
    Parameters:
    -----------
    T : Series of treatment assignments
    Y : Series of outcomes
    ps : Series of propensity scores
    stabilized : bool, Whether to use stabilized weights
    trimming : float or None, Percentile for trimming extreme weights
    
    Returns:
    --------
    ate : Estimated average treatment effect
    weights : Series of weights
    """
    # Calculate IPW weights
    if stabilized:
        # Stabilized weights
        p_treatment = T.mean()
        weights = np.where(T == 1, p_treatment / ps, (1 - p_treatment) / (1 - ps))
    else:
        # Unstabilized weights
        weights = np.where(T == 1, 1 / ps, 1 / (1 - ps))
    
    # Trim weights if requested
    if trimming is not None:
        max_weight = np.percentile(weights, trimming)
        weights = np.minimum(weights, max_weight)
    
    # Calculate weighted means
    weighted_treated = np.sum(weights[T == 1] * Y[T == 1]) / np.sum(weights[T == 1])
    weighted_control = np.sum(weights[T == 0] * Y[T == 0]) / np.sum(weights[T == 0])
    
    # Calculate ATE
    ate = weighted_treated - weighted_control
    
    return ate, weights

# Estimate ATE using IPW with different settings
ipw_results = []

for ps_method, ps_values in [('Logistic', ps_logistic_train), ('RF', ps_rf_train)]:
    for stabilized in [True, False]:
        for trimming in [None, 95]:
            # Calculate IPW estimate
            ipw_ate, weights = ipw_estimator(T_train, Y_train, ps_values, 
                                             stabilized=stabilized, trimming=trimming)
            
            # Save result
            ipw_results.append({
                'PS Method': ps_method,
                'Stabilized': stabilized,
                'Trimming': trimming,
                'ATE': ipw_ate,
                'Bias': ipw_ate - true_ate_train,
                'Abs Bias': abs(ipw_ate - true_ate_train),
                'Max Weight': np.max(weights),
                'Weight SD': np.std(weights)
            })

# Convert to DataFrame for easier visualization
ipw_results_df = pd.DataFrame(ipw_results)
print("IPW Estimation Results:")
print(ipw_results_df.sort_values('Abs Bias'))

# Find best IPW method
best_ipw_idx = ipw_results_df['Abs Bias'].idxmin()
best_ipw = ipw_results_df.loc[best_ipw_idx]
print(f"\nBest IPW Method:")
print(f"PS Method: {best_ipw['PS Method']}")
print(f"Stabilized: {best_ipw['Stabilized']}")
print(f"Trimming: {best_ipw['Trimming']}")
print(f"ATE: {best_ipw['ATE']:.4f}")
print(f"Bias: {best_ipw['Bias']:.4f}")

# Visualize the weights
best_ps = ps_rf_train if best_ipw['PS Method'] == 'RF' else ps_logistic_train
_, best_weights = ipw_estimator(T_train, Y_train, best_ps, 
                              stabilized=best_ipw['Stabilized'], 
                              trimming=best_ipw['Trimming'])

plt.figure(figsize=(10, 6))
plt.scatter(best_ps, best_weights, alpha=0.6, c=T_train, cmap='coolwarm')
plt.title('Propensity Scores vs Weights')
plt.xlabel('Propensity Score')
plt.ylabel('Weight')
plt.colorbar(label='Treatment')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Assess covariate balance after IPW
def assess_balance(X, T, weights=None):
    """
    Assess covariate balance before and after weighting
    
    Parameters:
    -----------
    X : DataFrame of covariates
    T : Series of treatment assignments
    weights : Series of weights or None
    
    Returns:
    --------
    balance_df : DataFrame with standardized mean differences
    """
    # Calculate standardized mean differences
    balance_stats = []
    
    for col in X.columns:
        # Unweighted means and std devs
        treated_mean = X.loc[T == 1, col].mean()
        control_mean = X.loc[T == 0, col].mean()
        treated_std = X.loc[T == 1, col].std()
        control_std = X.loc[T == 0, col].std()
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
        
        # Calculate standardized mean difference (SMD)
        if pooled_std == 0:
            smd_before = 0
        else:
            smd_before = (treated_mean - control_mean) / pooled_std
        
        # If weights are provided, calculate weighted means and std devs
        if weights is not None:
            # Calculate weighted means
            treated_weights = weights[T == 1]
            control_weights = weights[T == 0]
            
            treated_weighted_mean = np.average(X.loc[T == 1, col], weights=treated_weights)
            control_weighted_mean = np.average(X.loc[T == 0, col], weights=control_weights)
            
            # Calculate weighted std devs
            treated_weighted_std = np.sqrt(np.average((X.loc[T == 1, col] - treated_weighted_mean)**2, weights=treated_weights))
            control_weighted_std = np.sqrt(np.average((X.loc[T == 0, col] - control_weighted_mean)**2, weights=control_weights))
            
            # Calculate weighted pooled standard deviation
            weighted_pooled_std = np.sqrt((treated_weighted_std**2 + control_weighted_std**2) / 2)
            
            # Calculate weighted SMD
            if weighted_pooled_std == 0:
                smd_after = 0
            else:
                smd_after = (treated_weighted_mean - control_weighted_mean) / weighted_pooled_std
        else:
            smd_after = None
        
        # Add to balance stats
        balance_stats.append({
            'Variable': col,
            'SMD_Before': smd_before,
            'SMD_After': smd_after,
            'Mean_Diff_Before': treated_mean - control_mean
        })
    
    # Convert to DataFrame
    balance_df = pd.DataFrame(balance_stats)
    
    # Sort by absolute SMD before weighting
    balance_df = balance_df.sort_values('SMD_Before', key=abs, ascending=False)
    
    return balance_df

# Assess balance before and after IPW
balance_df = assess_balance(X_train_scaled, T_train, best_weights)

# Plot balance before and after weighting
plt.figure(figsize=(12, 10))
variables = balance_df['Variable'].head(15)  # Top 15 variables by initial imbalance

# Create plot
balance_plot = pd.DataFrame({
    'Before Weighting': balance_df.loc[balance_df['Variable'].isin(variables), 'SMD_Before'],
    'After Weighting': balance_df.loc[balance_df['Variable'].isin(variables), 'SMD_After']
})
balance_plot.index = variables

balance_plot.plot(kind='barh', figsize=(12, 10))
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.7)
plt.axvline(x=-0.1, color='red', linestyle='--', alpha=0.7)
plt.title('Standardized Mean Differences Before and After IPW')
plt.xlabel('Standardized Mean Difference')
plt.ylabel('Covariate')
plt.tight_layout()
plt.show()
```

**Analysis:** Inverse Probability Weighting (IPW) creates a pseudo-population where the distribution of covariates is balanced between treatment groups. This is achieved by weighting each observation inversely to its probability of receiving the treatment it actually received.

Key options in IPW:
1. **Stabilized weights**: Multiply by the marginal probability of treatment to reduce variance
2. **Weight trimming**: Truncate extreme weights to prevent a few observations from dominating

The weights vs. propensity scores plot illustrates how units with propensity scores close to 0 or 1 receive higher weights. This can lead to instability if there are units with very extreme propensity scores.

The balance plot shows standardized mean differences (SMDs) before and after weighting. SMDs closer to zero indicate better balance. Values within the red dashed lines (±0.1) are generally considered acceptable. IPW should improve balance across most or all covariates.

The best IPW method is selected based on the smallest absolute bias compared to the true ATE. In practice, where the true effect is unknown, we would rely on theoretical properties of the methods and covariate balance assessments.