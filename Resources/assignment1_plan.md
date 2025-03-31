# Assignment 1: Causal Inference Notebook Plan

## Topic
**Understanding Causal Inference with IHDP: From Theory to Practice**

## Notebook Structure

### 1. Title and Introduction
- Clear title banner with visually appealing formatting
- Brief introduction to causal inference and its importance in data science
- Overview of what the notebook will cover

### 2. Abstract
- Concise summary of the notebook content
- Key learning objectives:
  - Understanding causal vs. associational reasoning
  - Learning the four-step causal inference process
  - Implementing causal methods using Python
  - Interpreting and validating causal results

### 3. Theory Section

#### 3.1 Foundations of Causality
- The problem with correlation vs. causation
- Rubin's potential outcomes framework
- Pearl's structural causal models and do-calculus
- The ladder of causation (association, intervention, counterfactuals)

#### 3.2 Causal Identification
- Causal graphs and their interpretation
- Confounding and selection bias
- Identification strategies:
  - Backdoor criterion
  - Frontdoor criterion
  - Instrumental variables

#### 3.3 Estimation Methods
- Matching methods
- Propensity score methods
- Doubly-robust estimators
- Machine learning for causal inference

#### 3.4 Evaluation and Validation
- Refutation methods
- Sensitivity analysis
- Common pitfalls in causal analysis

### 4. Practical Implementation

#### 4.1 Dataset Introduction
- IHDP (Infant Health and Development Program) dataset
  - Background and context
  - Variables description
  - Causal question: Effect of specialized childcare on cognitive test scores

#### 4.2 Data Preprocessing
- Loading and exploring the data
- Handling missing values
- Feature engineering
- Exploratory data analysis with causal focus

#### 4.3 Causal Modeling with DoWhy
- Setting up the environment
- Step 1: Model - Creating the causal graph
- Step 2: Identify - Determining identifiability
- Step 3: Estimate - Implementing multiple estimation methods:
  - Linear regression
  - Propensity score matching
  - Inverse probability weighting
  - Doubly robust estimation
- Step 4: Refute - Testing robustness of results:
  - Placebo treatment
  - Adding random common causes
  - Bootstrapping
  - Sensitivity analysis

#### 4.4 Interpretation and Discussion
- Comparing results across methods
- Visualizing causal effects
- Understanding heterogeneous treatment effects
- Limitations and considerations

### 5. Conclusion
- Summary of key findings
- Practical implications
- Future directions for causal inference
- Final thoughts on the importance of causal reasoning

### 6. References
- Comprehensive bibliography of all resources used
- Links to libraries and repositories
- Additional learning resources

### 7. License
- Clear statement of reuse permissions

## Technical Implementation Details

### Libraries to Use
- DoWhy - Primary causal inference framework
- CausalML - For heterogeneous treatment effect estimation
- NumPy/Pandas - For data manipulation
- Matplotlib/Seaborn - For visualization
- NetworkX - For causal graph visualization
- Scikit-learn - For machine learning components

### Code Structure
- Modular code organization with clearly documented functions
- Markdown cells explaining concepts before code implementation
- Interactive visualizations where possible
- Error handling and robustness checks

### Video Presentation Plan
- 3-5 minute screencast covering:
  - Introduction to the causal problem
  - Brief explanation of the dataset
  - Walkthrough of the four-step causal inference process
  - Demonstration of key visualizations and results
  - Conclusion and key takeaways

## Timeline
1. Research and outline development: 2 days
2. Theory section drafting: 2 days
3. Code implementation: 3 days
4. Results analysis and visualization: 1 day
5. Refining explanations and documentation: 1 day
6. Video recording and editing: 1 day
7. Final review and submission: 1 day
