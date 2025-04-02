# Understanding Causal Inference with IHDP: From Theory to Practice - Part 4.1

## 4. Implementing Causal Inference on IHDP

### 4.1 Data Preparation

> 🔧 **Step 1**: Properly prepare the data for causal analysis

First, let's set up our environment and prepare the IHDP dataset for causal inference.

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Make plots look better
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Set random seed for reproducibility
np.random.seed(42)

# Function to load the IHDP dataset
def load_ihdp_data():
    """
    Load the IHDP dataset for causal inference
    
    Returns:
        DataFrame with treatment, outcome, and covariates
    """
    # Create a directory for the data if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download the data if it doesn't exist
    if not os.path.exists('data/ihdp_npci_1.csv'):
        print("Downloading IHDP dataset...")
        import urllib.request
        url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
        urllib.request.urlretrieve(url, 'data/ihdp_npci_1.csv')
    
    # Load the data
    data = pd.read_csv('data/ihdp_npci_1.csv')
    
    # Rename columns for clarity
    column_names = ['treatment']
    column_names.extend([f'y_{i}' for i in range(2)])  # factual and counterfactual outcomes
    column_names.extend([f'mu_{i}' for i in range(2)])  # expected outcomes without noise
    column_names.extend([f'x_{i}' for i in range(25)])  # covariates
    
    data.columns = column_names
    
    # Rename for more intuitive understanding
    data.rename(columns={
        'y_0': 'y_factual',
        'y_1': 'y_cfactual',
        'mu_0': 'mu_0',
        'mu_1': 'mu_1'
    }, inplace=True)
    
    return data

# Load the IHDP dataset
ihdp_data = load_ihdp_data()

# Display basic information
print(f"Dataset shape: {ihdp_data.shape}")
print(f"Treatment assignment rate: {ihdp_data['treatment'].mean():.2f}")
print(f"True ATE: {(ihdp_data['mu_1'] - ihdp_data['mu_0']).mean():.4f}")

# Create a more informative dataset with column descriptions
covariate_descriptions = {
    'x_0': 'Child\'s birth weight (grams)',
    'x_1': 'Child\'s birth order',
    'x_2': 'Head circumference at birth (cm)',
    'x_3': 'Mother\'s age at birth (years)',
    'x_4': 'Mother\'s education (years)',
    'x_5': 'Child\'s gender (1=male, 0=female)',
    'x_6': 'Twin (1=yes, 0=no)',
    'x_7': 'Number of previous neonatal deaths',
    'x_8': 'Mother\'s marital status (1=married, 0=not married)',
    'x_9': 'Mother smoked during pregnancy (1=yes, 0=no)',
    'x_10': 'Mother drank alcohol during pregnancy (1=yes, 0=no)',
    'x_11': 'Mother used drugs during pregnancy (1=yes, 0=no)',
    'x_12': 'Child\'s neonatal health index',
    'x_13': 'Mom white (1=yes, 0=no)',
    'x_14': 'Mom black (1=yes, 0=no)',
    'x_15': 'Mom Hispanic (1=yes, 0=no)',
    'x_16': 'Mom is employed (1=yes, 0=no)',
    'x_17': 'Family receives welfare (1=yes, 0=no)',
    'x_18': 'Mother works during pregnancy (1=yes, 0=no)',
    'x_19': 'Prenatal care visit in first trimester (1=yes, 0=no)',
    'x_20': 'Site 1 (1=yes, 0=no)',
    'x_21': 'Site 2 (1=yes, 0=no)',
    'x_22': 'Site 3 (1=yes, 0=no)',
    'x_23': 'Site 4 (1=yes, 0=no)',
    'x_24': 'Site 5 (1=yes, 0=no)'
}

# Identify continuous and binary variables
continuous_vars = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_12']
binary_vars = [f'x_{i}' for i in range(5, 25) if i != 12]

# Handle missing values (this dataset doesn't have any, but it's good practice)
print(f"Missing values before imputation:\n{ihdp_data.isnull().sum().sum()}")

# Check variable distributions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot continuous variables
sns.boxplot(data=ihdp_data[continuous_vars], ax=axes[0])
axes[0].set_title('Distribution of Continuous Covariates')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

# Plot binary variables
binary_means = ihdp_data[binary_vars].mean().sort_values(ascending=False)
sns.barplot(x=binary_means.index, y=binary_means.values, ax=axes[1])
axes[1].set_title('Proportion of 1s in Binary Covariates')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)

plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X = ihdp_data[[f'x_{i}' for i in range(25)]]
T = ihdp_data['treatment']
Y = ihdp_data['y_factual']
# True potential outcomes for evaluation (not available in real-world scenarios)
Y0 = ihdp_data['mu_0']
Y1 = ihdp_data['mu_1']

X_train, X_test, T_train, T_test, Y_train, Y_test, Y0_train, Y0_test, Y1_train, Y1_test = train_test_split(
    X, T, Y, Y0, Y1, test_size=0.2, random_state=42
)

# Scale continuous features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

for var in continuous_vars:
    X_train_scaled[var] = scaler.fit_transform(X_train[[var]])
    X_test_scaled[var] = scaler.transform(X_test[[var]])

print("Data splitting complete.")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
```

**Analysis:** This initial code prepares our dataset for causal inference. We've loaded the IHDP dataset, handled potential missing values, visualized the distributions of covariates, and split the data into training and testing sets. We've also scaled the continuous variables to improve the performance of our models.

The boxplots of continuous variables help us understand their distributions and identify potential outliers. The barplots of binary variables show the proportion of 1s for each variable, giving us insights into the prevalence of different characteristics in the sample.

Proper data preparation is crucial for causal inference. In this case, we've ensured that our data is clean, properly formatted, and split for evaluation. We've also identified the continuous and binary variables, which will be treated differently in some of our models.

#### References and Resources

- [IHDP Dataset Repository](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP)
- [Causal Inference with Python: Data Preparation](https://medium.com/towards-data-science/causal-inference-with-python-part-2-causal-graphical-models-365bebff4b8e)
- [Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics, 20(1), 217-240.](https://doi.org/10.1198/jcgs.2010.08162)