# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.3.5

#### 4.3.5 Comparing Propensity Score Methods

Let's compare all the propensity score methods we've implemented:

```python
# Combine results from all propensity score methods
ps_methods = []

# Add IPW results
for _, row in ipw_results_df.sort_values('Abs Bias').head(3).iterrows():
    ps_methods.append({
        'Method': f"IPW ({row['PS Method']}, stabilized={row['Stabilized']}, trimming={row['Trimming']})",
        'ATE': row['ATE'],
        'Bias': row['Bias'],
        'Abs Bias': row['Abs Bias']
    })

# Add matching results
for _, row in matching_results_df.sort_values('Abs Bias').head(3).iterrows():
    ps_methods.append({
        'Method': f"Matching ({row['PS Method']}, k={row['k']}, caliper={row['Caliper']})",
        'ATE': row['ATE'],
        'Bias': row['Bias'],
        'Abs Bias': row['Abs Bias']
    })

# Add stratification results
if not strat_results_df.empty:
    for _, row in strat_results_df.sort_values('Abs Bias').head(3).iterrows():
        ps_methods.append({
            'Method': f"Stratification ({row['PS Method']}, n_strata={row['n_strata']})",
            'ATE': row['ATE'],
            'Bias': row['Bias'],
            'Abs Bias': row['Abs Bias']
        })

# Convert to DataFrame and sort by absolute bias
ps_methods_df = pd.DataFrame(ps_methods)
ps_methods_df = ps_methods_df.sort_values('Abs Bias')

print("Comparison of Propensity Score Methods:")
print(ps_methods_df)

# Visualize comparison
plt.figure(figsize=(12, 8))
plt.barh(y=ps_methods_df['Method'], width=ps_methods_df['ATE'], color='skyblue')
plt.axvline(x=true_ate_train, color='red', linestyle='--', label=f'True ATE = {true_ate_train:.4f}')
plt.title('Comparison of ATE Estimates from Propensity Score Methods')
plt.xlabel('ATE Estimate')
plt.ylabel('Method')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Select the best overall method
best_method_idx = ps_methods_df['Abs Bias'].idxmin()
best_method = ps_methods_df.loc[best_method_idx]

print(f"\nBest Propensity Score Method:")
print(f"Method: {best_method['Method']}")
print(f"ATE: {best_method['ATE']:.4f}")
print(f"True ATE: {true_ate_train:.4f}")
print(f"Bias: {best_method['Bias']:.4f}")
```

**Analysis:** This comparison helps us identify which propensity score method performs best for estimating the causal effect in our dataset. The visualization makes it easy to see how different methods compare to the true ATE.

Each propensity score method has its own strengths and weaknesses:

1. **Inverse Probability Weighting (IPW):**
   - **Strengths**: Uses all data, relatively easy to implement, can be extended to more complex settings
   - **Weaknesses**: Sensitive to extreme propensity scores, may have high variance

2. **Propensity Score Matching:**
   - **Strengths**: Intuitive, preserves sample interpretability, good at reducing bias
   - **Weaknesses**: May discard data, can be computationally intensive with large datasets

3. **Propensity Score Stratification:**
   - **Strengths**: Simple to implement, preserves all data, allows examination of effect heterogeneity
   - **Weaknesses**: May not achieve optimal balance with many covariates

In practice, the choice of method should consider:
- The specific causal question and context
- Sample size and data structure
- Covariate balance achieved
- Computational resources available
- Interpretability of the results

When the true effect is unknown (as in real-world applications), we would select the method that best satisfies theoretical properties and achieves good covariate balance.

#### References and Resources

- Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. Multivariate Behavioral Research, 46(3), 399-424.
- Imbens, G. W., & Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences. Cambridge University Press.
- [Propensity Score Analysis with R](https://bookdown.org/mike/data_analysis/propensity-scores.html)
- [Causal Inference with Python: Propensity Score Methods](https://matheusfacure.github.io/python-causality-handbook/08-Propensity-Score.html)