# Assignment 2: Worked Examples Plan

## Overview
This plan outlines the approach for creating a Jupyter notebook with two worked examples of causal inference techniques, building on the theoretical foundation established in Assignment 1.

## Example 1: Hotel Booking Cancellations Analysis

### Dataset
- **Source**: TidyTuesday and DoWhy example
- **URL**: https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-02-11/readme.md
- **DoWhy Example**: https://www.pywhy.org/dowhy/v0.12/example_notebooks/DoWhy-The%20Causal%20Story%20Behind%20Hotel%20Booking%20Cancellations.html

### Implementation Plan

#### 1. Introduction and Business Context
- Explain the business problem: Understanding what causes hotel booking cancellations
- Discuss why this is a causal question rather than just a predictive one
- Outline the potential business impact of understanding causal factors

#### 2. Dataset Exploration
- Load and inspect the hotel bookings dataset
- Perform exploratory data analysis with visualizations
- Identify key variables and potential confounders
- Discuss the causal questions we want to answer:
  - What is the effect of deposit policies on cancellation rates?
  - How does lead time influence cancellation probability?

#### 3. Causal Model Construction
- Define treatment variables, outcome variables, and confounders
- Create a causal graph representing the hypothesized relationships
- Visualize the causal graph using networkx or DoWhy's visualization tools
- Explain the assumptions encoded in the graph

#### 4. Causal Effect Estimation
- Implement the four-step DoWhy workflow:
  - Model: Create causal model with the graph
  - Identify: Determine if effects are identifiable
  - Estimate: Apply multiple estimation methods
    - Propensity score matching
    - Inverse probability weighting
    - Stratification
  - Refute: Test the robustness of findings
- Compare the results across different estimation methods

#### 5. Interpretation and Business Insights
- Analyze the causal effects discovered
- Discuss practical implications for hotel management
- Suggest potential interventions to reduce cancellation rates
- Evaluate limitations of the analysis

## Example 2: Job Training Program Evaluation (Lalonde Dataset)

### Dataset
- **Source**: DoWhy example
- **URL**: https://www.pywhy.org/dowhy/v0.12/example_notebooks/dowhy_lalonde_example.html

### Implementation Plan

#### 1. Introduction and Social Policy Context
- Introduce the Lalonde dataset and its historical significance in causal inference
- Explain the policy question: Does job training increase future earnings?
- Discuss the challenges of evaluating social programs

#### 2. Dataset Exploration
- Load and examine the Lalonde dataset
- Analyze the characteristics of treatment and control groups
- Explore potential selection bias issues
- Visualize key relationships between variables

#### 3. Causal Model Development
- Define the causal estimand (Average Treatment Effect)
- Construct a causal graph representing the job training scenario
- Visualize the graph and discuss assumptions
- Address the challenge of selection bias in observational studies

#### 4. Advanced Causal Estimation Techniques
- Implement methods that differ from Example 1:
  - Regression adjustment
  - Doubly-robust estimation
  - Matching with different distance metrics
- Compare observational estimates with experimental benchmark
- Apply sensitivity analysis to assess unobserved confounding

#### 5. Critical Evaluation and Policy Implications
- Interpret the estimated treatment effects
- Discuss the reliability of the causal estimates
- Compare findings with historical analyses of this dataset
- Outline policy implications and considerations for future program evaluation

## Connecting to Assignment 1

### Theoretical Connections
- Reference concepts introduced in Assignment 1
- Highlight how these real-world examples illustrate theoretical principles
- Note differences in application between the healthcare context (IHDP) and these new domains

### Methodological Extensions
- Implement at least one advanced technique not covered in Assignment 1
- Demonstrate how causal inference methods can be adapted to different domains
- Address domain-specific challenges in each example

## Technical Implementation

### Libraries and Tools
- DoWhy for core causal inference workflow
- Pandas for data manipulation
- Matplotlib, Seaborn, and Plotly for advanced visualizations
- NetworkX for causal graph representations
- CausalML for heterogeneous treatment effects (if applicable)

### Code Structure
- Modular, well-documented functions
- Detailed markdown explanations connecting to theory
- Interactive components where appropriate
- Comparison tables for different estimation methods

## Timeline
1. Data acquisition and exploration: 1 day
2. Causal modeling for both examples: 2 days
3. Implementation of estimation methods: 2 days
4. Refutation and sensitivity analysis: 1 day
5. Visualization and results interpretation: 1 day
6. Documentation and finalization: 1 day
