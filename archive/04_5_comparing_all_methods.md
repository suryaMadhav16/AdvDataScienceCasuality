# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.5

### 4.5 Comparing All Methods

Let's compare all the causal inference methods we've implemented, from simple approaches to advanced machine learning techniques.

```python
# Combine results from all methods
all_methods = []

# Add simple methods
simple_methods = [
    {'Method': 'Naive Mean Difference', 'ATE': naive_ate_train, 'Bias': naive_ate_train - true_ate_train},
    {'Method': 'Regression Adjustment', 'ATE': reg_ate_train, 'Bias': reg_ate_train - true_ate_train}
]

for method in simple_methods:
    all_methods.append({
        'Method': method['Method'],
        'ATE': method['ATE'],
        'Bias': method['Bias'],
        'Abs Bias': abs(method['Bias']),
        'Type': 'Simple'
    })

# Add propensity score methods
# Get best methods from each category
if 'best_ipw' in locals():
    all_methods.append({
        'Method': f"IPW ({best_ipw['PS Method']}, stabilized={best_ipw['Stabilized']}, trimming={best_ipw['Trimming']})",
        'ATE': best_ipw['ATE'],
        'Bias': best_ipw['Bias'],
        'Abs Bias': abs(best_ipw['Bias']),
        'Type': 'Propensity Score'
    })

if 'best_match' in locals():
    all_methods.append({
        'Method': f"Matching ({best_match['PS Method']}, k={best_match['k']}, caliper={best_match['Caliper']})",
        'ATE': best_match['ATE'],
        'Bias': best_match['Bias'],
        'Abs Bias': abs(best_match['Bias']),
        'Type': 'Propensity Score'
    })

if 'best_strat' in locals():
    all_methods.append({
        'Method': f"Stratification ({best_strat['PS Method']}, n_strata={best_strat['n_strata']})",
        'ATE': best_strat['ATE'],
        'Bias': best_strat['Bias'],
        'Abs Bias': abs(best_strat['Bias']),
        'Type': 'Propensity Score'
    })

# Add meta-learners
if 'best_ml' in locals():
    all_methods.append({
        'Method': best_ml['Method'],
        'ATE': best_ml['ATE'],
        'Bias': best_ml['Bias'],
        'Abs Bias': abs(best_ml['Bias']),
        'Type': 'Meta-Learner'
    })

# Add doubly robust methods
if 'best_dr' in locals():
    all_methods.append({
        'Method': best_dr['Method'],
        'ATE': best_dr['ATE'],
        'Bias': best_dr['Bias'],
        'Abs Bias': abs(best_dr['Bias']),
        'Type': 'Doubly Robust'
    })

# Add causal forest if available
if 'cf_ate' in locals():
    all_methods.append({
        'Method': 'Causal Forest',
        'ATE': cf_ate,
        'Bias': cf_ate - true_ate_train,
        'Abs Bias': abs(cf_ate - true_ate_train),
        'Type': 'Causal Forest'
    })

# Convert to DataFrame
all_methods_df = pd.DataFrame(all_methods)

# Sort by absolute bias
all_methods_df = all_methods_df.sort_values('Abs Bias')

print("Comparison of All Causal Inference Methods:")
print(all_methods_df)

# Visualize comparison
plt.figure(figsize=(14, 10))
colors = {'Simple': 'skyblue', 'Propensity Score': 'lightgreen', 
         'Meta-Learner': 'salmon', 'Doubly Robust': 'purple', 
         'Causal Forest': 'orange'}

# Plot bars with colors by method type
for i, (idx, row) in enumerate(all_methods_df.iterrows()):
    plt.barh(i, row['ATE'], color=colors[row['Type']])

# Add method names and true ATE line
plt.yticks(range(len(all_methods_df)), all_methods_df['Method'])
plt.axvline(x=true_ate_train, color='red', linestyle='--', label=f'True ATE = {true_ate_train:.4f}')

# Add legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
plt.legend(handles, colors.keys(), loc='lower right')

plt.title('Comparison of ATE Estimates from All Methods')
plt.xlabel('ATE Estimate')
plt.ylabel('Method')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Identify best overall method
best_overall_idx = all_methods_df['Abs Bias'].idxmin()
best_overall = all_methods_df.loc[best_overall_idx]

print(f"\nBest Overall Method:")
print(f"Method: {best_overall['Method']}")
print(f"ATE: {best_overall['ATE']:.4f}")
print(f"True ATE: {true_ate_train:.4f}")
print(f"Bias: {best_overall['Bias']:.4f}")
print(f"Type: {best_overall['Type']}")

# Evaluate methods on test set
# For the best method in each category, evaluate on test set if applicable

# Example for best meta-learner (assuming it's X-Learner with Random Forest)
if 'best_ml' in locals() and 'X-Learner (Random Forest)' in best_ml['Method']:
    # Run X-Learner on test set
    x_ate_test, x_cate_test, _ = x_learner(
        X_test_scaled, T_test, Y_test,
        models_t=[RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42),
                 RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=43)],
        models_c=[RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=44),
                 RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=45)],
        propensity_model=LogisticRegression(max_iter=1000)
    )
    
    # Calculate true ATE on test set
    true_ate_test = (Y1_test - Y0_test).mean()
    
    print(f"\nX-Learner Test Performance:")
    print(f"ATE: {x_ate_test:.4f}")
    print(f"True ATE: {true_ate_test:.4f}")
    print(f"Bias: {x_ate_test - true_ate_test:.4f}")
```

**Analysis:** This comprehensive comparison brings together all the causal inference methods we've implemented, from simple approaches to advanced machine learning techniques. The visualization makes it easy to see which methods provide estimates closest to the true ATE.

Several key insights emerge from this comparison:

1. **Simple methods** like naive mean difference and regression adjustment can be biased due to confounding, but they provide a useful baseline for comparison.

2. **Propensity score methods** like IPW, matching, and stratification can reduce bias when the propensity score model is well-specified. They work well in many situations and are widely used in practice.

3. **Meta-learners** leverage flexible machine learning algorithms to capture complex relationships. Their performance depends on the choice of base learner and the specific approach (S, T, or X).

4. **Doubly robust methods** provide protection against model misspecification and often have lower bias. They combine the strengths of outcome modeling and propensity score approaches.

5. **Causal forests** are designed specifically for estimating heterogeneous treatment effects. They provide valuable insights into effect heterogeneity and feature importance.

The best method overall depends on several factors:
- The specific data and context
- The causal question of interest
- The available computational resources
- The need for interpretability vs. predictive accuracy
- The presence of treatment effect heterogeneity

In practice, it's valuable to implement multiple methods and compare their results, as we've done here. Agreement across different methods increases our confidence in the estimates, while disagreement suggests a need for further investigation.

This comprehensive analysis demonstrates that advanced machine learning methods can provide more accurate and nuanced causal estimates than simpler approaches, especially when the relationships between variables are complex. However, they also require more data, computational resources, and careful implementation.