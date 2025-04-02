# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.4.4

#### 4.4.3 Causal Forests

Causal forests extend random forests to directly estimate heterogeneous treatment effects.

```python
# Try to import causalml for Causal Forest
try:
    from causalml.inference.tree import CausalForestDML
    from sklearn.linear_model import LassoCV
    
    # Implement causal forest
    def causal_forest(X, T, Y, n_estimators=100, min_samples_leaf=5):
        """
        Estimate ATE and CATE using Causal Forest
        
        Parameters:
        -----------
        X : DataFrame of covariates
        T : Series of treatment assignments
        Y : Series of outcomes
        n_estimators : int, Number of trees
        min_samples_leaf : int, Minimum samples in leaf
        
        Returns:
        --------
        ate : Estimated average treatment effect
        cate : Estimated conditional average treatment effects
        model : Fitted causal forest model
        """
        # Initialize model
        cf = CausalForestDML(
            model_y=LassoCV(),
            model_t=LogisticRegression(max_iter=1000),
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # Fit model
        cf.fit(X=X, treatment=T, y=Y)
        
        # Predict CATE
        cate = cf.predict(X=X)
        
        # Calculate ATE
        ate = cate.mean()
        
        return ate, cate, cf
    
    # Run causal forest
    cf_ate, cf_cate, cf_model = causal_forest(X_train_scaled, T_train, Y_train)
    
    # Add to results
    dr_results.append({
        'Method': 'Causal Forest',
        'ATE': cf_ate,
        'Bias': cf_ate - true_ate_train,
        'Abs Bias': abs(cf_ate - true_ate_train)
    })
    
    # Update DataFrame
    dr_df = pd.DataFrame(dr_results)
    dr_df = dr_df.sort_values('Abs Bias')
    
    print("\nAdded Causal Forest to results.")
    print(f"Causal Forest ATE: {cf_ate:.4f}")
    print(f"Causal Forest Bias: {cf_ate - true_ate_train:.4f}")
    
    # Plot feature importance from causal forest
    if hasattr(cf_model, 'feature_importances_'):
        feature_imp = pd.DataFrame({
            'Feature': X_train_scaled.columns,
            'Importance': cf_model.feature_importances_
        })
        feature_imp = feature_imp.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
        plt.title('Feature Importance for Treatment Effect Heterogeneity')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Plot partial dependence for the most important feature
    try:
        if len(feature_imp) > 0:
            top_feature = feature_imp['Feature'].iloc[0]
            
            # Create grid of values for the feature
            feature_idx = list(X_train_scaled.columns).index(top_feature)
            grid = np.linspace(X_train_scaled[top_feature].min(), 
                            X_train_scaled[top_feature].max(), 
                            num=50)
            
            # For each value, predict CATE
            pd_vals = []
            for val in grid:
                X_pd = X_train_scaled.copy()
                X_pd[top_feature] = val
                cate_pd = cf_model.predict(X_pd)
                pd_vals.append(cate_pd.mean())
            
            # Plot partial dependence
            plt.figure(figsize=(10, 6))
            plt.plot(grid, pd_vals)
            plt.title(f'Partial Dependence Plot for {top_feature}')
            plt.xlabel(f'Value of {top_feature}')
            plt.ylabel('Average CATE')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Could not create partial dependence plot: {e}")

except ImportError:
    print("CausalML package not available. Skipping Causal Forest.")
    print("To install: pip install causalml")
```

**Analysis:** Causal forests directly estimate heterogeneous treatment effects by adapting random forests for causal inference. Developed by Wager and Athey (2018), they build upon the idea of honest trees, where the training sample is split into two parts: one for determining the splits and another for estimating the treatment effects within the leaves.

Key advantages of causal forests:

1. **Direct estimation of heterogeneity**: Unlike meta-learners, causal forests are specifically designed to identify treatment effect heterogeneity.

2. **Feature importance**: They provide measures of which variables contribute most to treatment effect heterogeneity, offering valuable insights for targeting interventions.

3. **Nonparametric flexibility**: They can capture complex, nonlinear relationships without requiring specific functional forms.

4. **Theoretical guarantees**: Under certain conditions, causal forest estimates are asymptotically normal and consistent.

The feature importance plot reveals which variables are most predictive of treatment effect heterogeneity. This information is particularly valuable for understanding which subgroups benefit most from the treatment and for designing targeted interventions.

The partial dependence plot shows how the conditional average treatment effect varies with the most important feature, providing insights into the nature of the treatment effect heterogeneity.

In practice, causal forests are especially useful when:
- The focus is on understanding heterogeneity rather than just the average effect
- There are many potential effect modifiers
- The relationships between variables are complex and nonlinear
- The sample size is large enough to reliably estimate heterogeneous effects

#### 4.4.4 Comparing All Advanced Methods

Now let's compare all the advanced machine learning methods we've implemented:

```python
# Combine results from all advanced methods
advanced_methods = []

# Add meta-learners
for _, row in meta_learner_df.sort_values('Abs Bias').head(3).iterrows():
    advanced_methods.append({
        'Method': row['Method'],
        'ATE': row['ATE'],
        'Bias': row['Bias'],
        'Abs Bias': row['Abs Bias']
    })

# Add doubly robust methods
for _, row in dr_df.sort_values('Abs Bias').head(3).iterrows():
    advanced_methods.append({
        'Method': row['Method'],
        'ATE': row['ATE'],
        'Bias': row['Bias'],
        'Abs Bias': row['Abs Bias']
    })

# Add causal forest if available
if 'cf_ate' in locals():
    advanced_methods.append({
        'Method': 'Causal Forest',
        'ATE': cf_ate,
        'Bias': cf_ate - true_ate_train,
        'Abs Bias': abs(cf_ate - true_ate_train)
    })

# Convert to DataFrame and sort by absolute bias
advanced_methods_df = pd.DataFrame(advanced_methods)
advanced_methods_df = advanced_methods_df.sort_values('Abs Bias')

print("Comparison of Advanced Machine Learning Methods:")
print(advanced_methods_df)

# Visualize comparison
plt.figure(figsize=(12, 8))
plt.barh(y=advanced_methods_df['Method'], width=advanced_methods_df['ATE'], color='skyblue')
plt.axvline(x=true_ate_train, color='red', linestyle='--', label=f'True ATE = {true_ate_train:.4f}')
plt.title('Comparison of ATE Estimates from Advanced ML Methods')
plt.xlabel('ATE Estimate')
plt.ylabel('Method')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Select the best advanced method
best_advanced_idx = advanced_methods_df['Abs Bias'].idxmin()
best_advanced = advanced_methods_df.loc[best_advanced_idx]

print(f"\nBest Advanced Method:")
print(f"Method: {best_advanced['Method']}")
print(f"ATE: {best_advanced['ATE']:.4f}")
print(f"True ATE: {true_ate_train:.4f}")
print(f"Bias: {best_advanced['Bias']:.4f}")
```

**Analysis:** The comparison of advanced machine learning methods shows their relative performance in estimating the average treatment effect (ATE). Each method has its strengths and weaknesses:

1. **Meta-learners** (S, T, X) are flexible and can incorporate any machine learning algorithm. They work well when the relationships are complex, but their performance depends heavily on the choice of base learner.

2. **Doubly robust methods** (AIPW, DML) provide protection against model misspecification. They tend to have lower bias and are more robust to model choice, making them a safe option when we're uncertain about the true data-generating process.

3. **Causal forests** excel at capturing treatment effect heterogeneity and provide valuable insights through feature importance. They're particularly useful when the focus is on understanding which subgroups benefit most from the treatment.

The best method overall depends on the specific context and objectives:
- For estimating the ATE with minimal bias, doubly robust methods often perform best
- For understanding heterogeneity, causal forests provide the most direct insights
- For flexibility and ease of implementation, meta-learners are a good choice

Advanced machine learning methods generally outperform simpler methods, especially when the relationships between variables are complex. However, they also require more data and computational resources, and their performance can be sensitive to hyperparameter choices.

In practice, it's often valuable to implement multiple methods and compare their results, as we've done here. Agreement across different methods increases our confidence in the estimates, while disagreement suggests a need for further investigation.