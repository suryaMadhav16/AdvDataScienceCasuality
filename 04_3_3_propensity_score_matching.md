# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.3.3

#### 4.3.3 Propensity Score Matching

Propensity score matching pairs treated units with control units having similar propensity scores.

```python
from sklearn.neighbors import NearestNeighbors

def ps_matching_estimator(X, T, Y, ps, method='nearest', k=1, caliper=None):
    """
    Estimate ATE using propensity score matching
    
    Parameters:
    -----------
    X : DataFrame of covariates
    T : Series of treatment assignments
    Y : Series of outcomes
    ps : Series of propensity scores
    method : str, Matching method ('nearest' or 'caliper')
    k : int, Number of matches
    caliper : float or None, Caliper width (in standard deviations)
    
    Returns:
    --------
    ate : Estimated average treatment effect
    matched_data : DataFrame of matched data
    """
    # Create a dataframe with all necessary info
    data = pd.DataFrame({
        'treatment': T,
        'outcome': Y,
        'ps': ps
    })
    
    # Add covariates
    for col in X.columns:
        data[col] = X[col]
    
    # Separate treated and control
    treated = data[data['treatment'] == 1]
    control = data[data['treatment'] == 0]
    
    # Prepare for matching
    treated_ps = treated['ps'].values.reshape(-1, 1)
    control_ps = control['ps'].values.reshape(-1, 1)
    
    if method == 'nearest':
        # Nearest neighbor matching
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(control_ps)
        distances, indices = nn.kneighbors(treated_ps)
        
        # For each treated unit, find its matches
        matched_pairs = []
        
        for i, treated_idx in enumerate(treated.index):
            for j in range(k):
                control_idx = control.index[indices[i, j]]
                dist = distances[i, j]
                
                if caliper is None or dist < caliper * np.std(data['ps']):
                    matched_pairs.append({
                        'treated_idx': treated_idx,
                        'control_idx': control_idx,
                        'treated_ps': treated.loc[treated_idx, 'ps'],
                        'control_ps': control.loc[control_idx, 'ps'],
                        'treated_outcome': treated.loc[treated_idx, 'outcome'],
                        'control_outcome': control.loc[control_idx, 'outcome'],
                        'ps_diff': abs(treated.loc[treated_idx, 'ps'] - control.loc[control_idx, 'ps'])
                    })
        
        # Create dataframe of matched pairs
        matched_df = pd.DataFrame(matched_pairs)
        
        if matched_df.empty:
            print("No matches found with current settings")
            return np.nan, None
        
        # Calculate treatment effect
        ate = (matched_df['treated_outcome'] - matched_df['control_outcome']).mean()
        
        return ate, matched_df
    
    else:
        raise ValueError(f"Unknown matching method: {method}")

# Perform propensity score matching with different settings
matching_results = []

for ps_method, ps_values in [('Logistic', ps_logistic_train), ('RF', ps_rf_train)]:
    for k in [1, 5]:
        for caliper in [None, 0.2]:
            # Skip if not using a caliper with multiple neighbors
            if k > 1 and caliper is None:
                continue
                
            # Calculate matching estimate
            psm_ate, matched_data = ps_matching_estimator(
                X_train_scaled, T_train, Y_train, ps_values, 
                method='nearest', k=k, caliper=caliper
            )
            
            if matched_data is not None:
                # Save result
                matching_results.append({
                    'PS Method': ps_method,
                    'k': k,
                    'Caliper': caliper,
                    'ATE': psm_ate,
                    'Bias': psm_ate - true_ate_train,
                    'Abs Bias': abs(psm_ate - true_ate_train),
                    'Matches': len(matched_data)
                })

# Convert to DataFrame for easier visualization
matching_results_df = pd.DataFrame(matching_results)
print("Propensity Score Matching Results:")
print(matching_results_df.sort_values('Abs Bias'))

# Find best matching method
best_match_idx = matching_results_df['Abs Bias'].idxmin()
best_match = matching_results_df.loc[best_match_idx]
print(f"\nBest Matching Method:")
print(f"PS Method: {best_match['PS Method']}")
print(f"k: {best_match['k']}")
print(f"Caliper: {best_match['Caliper']}")
print(f"ATE: {best_match['ATE']:.4f}")
print(f"Bias: {best_match['Bias']:.4f}")
print(f"Number of matches: {best_match['Matches']}")

# Visualize matches for the best method
best_ps = ps_rf_train if best_match['PS Method'] == 'RF' else ps_logistic_train
_, best_matched_data = ps_matching_estimator(
    X_train_scaled, T_train, Y_train, best_ps, 
    method='nearest', k=int(best_match['k']), caliper=best_match['Caliper']
)

# Visualize the matches
if best_matched_data is not None:
    plt.figure(figsize=(10, 6))
    
    if len(best_matched_data) > 100:
        # Sample 100 matches for visualization
        sample_matches = best_matched_data.sample(100, random_state=42)
    else:
        sample_matches = best_matched_data
    
    for _, row in sample_matches.iterrows():
        plt.plot([row['treated_ps'], row['control_ps']], [0, 1], 'k-', alpha=0.1)
    
    plt.scatter(best_ps[T_train == 1], np.ones_like(best_ps[T_train == 1]), 
                color='red', alpha=0.5, label='Treated')
    plt.scatter(best_ps[T_train == 0], np.zeros_like(best_ps[T_train == 0]), 
                color='blue', alpha=0.5, label='Control')
    
    plt.title('Propensity Score Matching Visualization')
    plt.xlabel('Propensity Score')
    plt.yticks([0, 1], ['Control', 'Treated'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot quality of matches
    plt.figure(figsize=(10, 6))
    plt.hist(best_matched_data['ps_diff'], bins=30, alpha=0.7)
    plt.title('Distribution of Propensity Score Differences in Matched Pairs')
    plt.xlabel('Absolute Difference in Propensity Scores')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Assess covariate balance after matching
    # Create indicator for matched units
    matched_treated = best_matched_data['treated_idx'].unique()
    matched_control = best_matched_data['control_idx'].unique()
    matched_idx = np.concatenate([matched_treated, matched_control])
    
    # Create weight vector (1 for matched units, 0 for unmatched)
    match_weights = pd.Series(0, index=X_train_scaled.index)
    match_weights.loc[matched_idx] = 1
    
    # Assess balance
    balance_after_matching = assess_balance(X_train_scaled, T_train, match_weights)
    
    # Plot balance before and after matching
    plt.figure(figsize=(12, 10))
    variables = balance_after_matching['Variable'].head(15)  # Top 15 variables by initial imbalance
    
    # Create plot
    balance_plot = pd.DataFrame({
        'Before Matching': balance_after_matching.loc[balance_after_matching['Variable'].isin(variables), 'SMD_Before'],
        'After Matching': balance_after_matching.loc[balance_after_matching['Variable'].isin(variables), 'SMD_After']
    })
    balance_plot.index = variables
    
    balance_plot.plot(kind='barh', figsize=(12, 10))
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=-0.1, color='red', linestyle='--', alpha=0.7)
    plt.title('Standardized Mean Differences Before and After Matching')
    plt.xlabel('Standardized Mean Difference')
    plt.ylabel('Covariate')
    plt.tight_layout()
    plt.show()
```

**Analysis:** Propensity score matching creates pairs of treated and control units with similar propensity scores. This mimics a randomized experiment by creating comparable groups.

Key options in matching:
1. **Number of neighbors (k)**: How many control units to match to each treated unit
2. **Caliper**: Maximum allowed difference in propensity scores for a valid match
3. **Propensity score method**: How the propensity scores are estimated

The matching visualization shows the pairs of treated and control units connected by lines. Shorter lines indicate closer matches in terms of propensity scores.

The histogram of propensity score differences shows the quality of the matches. Smaller differences indicate better matching quality.

The balance plot shows how matching improves covariate balance. Good matching should reduce standardized mean differences across most covariates.

The best matching method is selected based on the smallest absolute bias compared to the true ATE. In practice, where the true effect is unknown, we would select the method that achieves the best covariate balance.

Matching has the advantage of being more intuitive than weighting, but it may discard units that cannot be matched, potentially reducing statistical power.