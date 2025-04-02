# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.3.4

#### 4.3.4 Propensity Score Stratification

Stratification divides the sample into subgroups (strata) based on propensity scores and estimates treatment effects within each stratum.

```python
def ps_stratification(T, Y, ps, n_strata=5):
    """
    Estimate ATE using propensity score stratification
    
    Parameters:
    -----------
    T : Series of treatment assignments
    Y : Series of outcomes
    ps : Series of propensity scores
    n_strata : int, Number of strata
    
    Returns:
    --------
    ate : Estimated average treatment effect
    stratum_effects : List of treatment effects by stratum
    stratum_sizes : List of stratum sizes
    """
    # Create a DataFrame with all necessary variables
    data = pd.DataFrame({
        'treatment': T,
        'outcome': Y,
        'ps': ps
    })
    
    # Create strata based on propensity scores
    data['stratum'] = pd.qcut(data['ps'], n_strata, labels=False)
    
    # Calculate treatment effect within each stratum
    stratum_effects = []
    stratum_sizes = []
    
    for stratum in range(n_strata):
        stratum_data = data[data['stratum'] == stratum]
        
        # Check if both treated and control units exist in this stratum
        if (stratum_data['treatment'] == 1).sum() > 0 and (stratum_data['treatment'] == 0).sum() > 0:
            # Calculate treatment effect
            treated_mean = stratum_data.loc[stratum_data['treatment'] == 1, 'outcome'].mean()
            control_mean = stratum_data.loc[stratum_data['treatment'] == 0, 'outcome'].mean()
            effect = treated_mean - control_mean
            
            # Save effect and size
            stratum_effects.append(effect)
            stratum_sizes.append(len(stratum_data))
        else:
            print(f"Stratum {stratum} does not have both treated and control units.")
    
    # Calculate weighted average of stratum-specific effects
    if len(stratum_effects) > 0:
        weights = np.array(stratum_sizes) / sum(stratum_sizes)
        ate = sum(weights * np.array(stratum_effects))
        return ate, stratum_effects, stratum_sizes
    else:
        return np.nan, [], []

# Apply stratification with different propensity score models
strat_results = []

for ps_method, ps_values in [('Logistic', ps_logistic_train), ('RF', ps_rf_train)]:
    for n_strata in [5, 10]:
        # Calculate stratification estimate
        strat_ate, stratum_effects, stratum_sizes = ps_stratification(T_train, Y_train, ps_values, n_strata)
        
        if not np.isnan(strat_ate):
            # Save result
            strat_results.append({
                'PS Method': ps_method,
                'n_strata': n_strata,
                'ATE': strat_ate,
                'Bias': strat_ate - true_ate_train,
                'Abs Bias': abs(strat_ate - true_ate_train),
                'Stratum Effects': stratum_effects,
                'Stratum Sizes': stratum_sizes
            })

# Convert to DataFrame for easier visualization
strat_results_df = pd.DataFrame(strat_results)
print("Propensity Score Stratification Results:")
print(strat_results_df.sort_values('Abs Bias'))

# Find best stratification method
if not strat_results_df.empty:
    best_strat_idx = strat_results_df['Abs Bias'].idxmin()
    best_strat = strat_results_df.loc[best_strat_idx]
    print(f"\nBest Stratification Method:")
    print(f"PS Method: {best_strat['PS Method']}")
    print(f"Number of strata: {best_strat['n_strata']}")
    print(f"ATE: {best_strat['ATE']:.4f}")
    print(f"Bias: {best_strat['Bias']:.4f}")
    
    # Visualize stratum-specific effects
    plt.figure(figsize=(10, 6))
    strata = list(range(len(best_strat['Stratum Effects'])))
    plt.bar(strata, best_strat['Stratum Effects'], alpha=0.7)
    plt.axhline(y=best_strat['ATE'], color='red', linestyle='--', 
                label=f'Overall ATE: {best_strat["ATE"]:.4f}')
    plt.axhline(y=true_ate_train, color='green', linestyle=':', 
                label=f'True ATE: {true_ate_train:.4f}')
    plt.title('Treatment Effects by Propensity Score Stratum')
    plt.xlabel('Propensity Score Stratum (low to high)')
    plt.ylabel('Stratum-Specific ATE')
    plt.xticks(strata)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Visualize distribution of units across strata
    plt.figure(figsize=(10, 6))
    
    # Create DataFrame for counting units in each stratum
    stratum_counts = pd.DataFrame()
    
    # Use best propensity scores
    best_ps = ps_rf_train if best_strat['PS Method'] == 'RF' else ps_logistic_train
    
    # Create strata
    strata_data = pd.DataFrame({
        'treatment': T_train,
        'ps': best_ps
    })
    strata_data['stratum'] = pd.qcut(strata_data['ps'], int(best_strat['n_strata']), labels=False)
    
    # Count treated and control units in each stratum
    stratum_counts = strata_data.groupby(['stratum', 'treatment']).size().unstack()
    stratum_counts.columns = ['Control', 'Treated']
    
    # Plot
    stratum_counts.plot(kind='bar', figsize=(10, 6))
    plt.title('Distribution of Treated and Control Units Across Strata')
    plt.xlabel('Propensity Score Stratum (low to high)')
    plt.ylabel('Number of Units')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Assess covariate balance within strata
    # For simplicity, let's focus on the first stratum as an example
    stratum_0 = strata_data[strata_data['stratum'] == 0].index
    
    # Create indicator for units in this stratum
    stratum_indicator = pd.Series(0, index=X_train_scaled.index)
    stratum_indicator.loc[stratum_0] = 1
    
    # Assess balance in this stratum
    balance_stratum_0 = assess_balance(X_train_scaled, T_train, stratum_indicator)
    
    # Plot balance for the most imbalanced covariates
    plt.figure(figsize=(12, 10))
    variables = balance_stratum_0['Variable'].head(10)  # Top 10 variables by initial imbalance
    
    # Create plot
    balance_plot = pd.DataFrame({
        'Before Stratification': balance_stratum_0.loc[balance_stratum_0['Variable'].isin(variables), 'SMD_Before'],
        'Within Stratum 0': balance_stratum_0.loc[balance_stratum_0['Variable'].isin(variables), 'SMD_After']
    })
    balance_plot.index = variables
    
    balance_plot.plot(kind='barh', figsize=(12, 10))
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=-0.1, color='red', linestyle='--', alpha=0.7)
    plt.title('Standardized Mean Differences Before and Within Stratum 0')
    plt.xlabel('Standardized Mean Difference')
    plt.ylabel('Covariate')
    plt.tight_layout()
    plt.show()
```

**Analysis:** Propensity score stratification divides the sample into subgroups (strata) based on similar propensity scores. The treatment effect is estimated within each stratum, and the overall ATE is calculated as a weighted average of these stratum-specific effects.

Key considerations in stratification:
1. **Number of strata**: Typically 5-10 strata are used. More strata can reduce bias but may lead to strata with too few observations.
2. **Propensity score method**: How the propensity scores are estimated.
3. **Balance within strata**: Ideally, treated and control units within the same stratum should have similar covariate distributions.

The stratum-specific effects visualization shows how treatment effects vary across different propensity score values. Heterogeneity in these effects may indicate effect modification by variables related to treatment assignment.

The distribution of units plot shows how treated and control units are distributed across strata. Ideally, each stratum should contain both treated and control units for valid comparison.

The balance plot for a specific stratum demonstrates how stratification improves covariate balance within strata. Better balance leads to more reliable causal estimates.

Stratification has several advantages:
- It's easier to implement and understand than some other methods
- It allows for examination of treatment effect heterogeneity across strata
- It provides a good balance between bias reduction and variance

However, it may not achieve as good covariate balance as matching or weighting, especially with many covariates or strong selection bias.