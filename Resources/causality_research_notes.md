# Causality in Data Science - Research Notes

## What is Causality?

Causality refers to the relationship between cause and effect. In data science, causal inference explores how variables influence each other beyond simple correlation, which is crucial for making accurate predictions and informed decisions.

In data science, establishing causality is about understanding how one action directly influences another. It's the fundamental difference between recognizing patterns and making data-driven decisions that actually work.


![Correlation vs Causation](https://imgs.xkcd.com/comics/correlation.png)

*The classic XKCD comic illustrating the correlation vs. causation concept*
## The Importance of Causality vs. Correlation

Causal Inference is a very relevant subject for Data Science, as it allows us to go beyond the simple description of data and to understand what are the effects of a certain action or intervention on a variable of interest. 

The famous phrase "correlation does not imply causation" is critical in data science. For example:
- Ice cream sales and drowning incidents both increase in summer, but eating ice cream doesn't cause drowning
- Both are correlated because they share a common cause (summer weather)

## Key Concepts in Causal Inference

### Pearl's Three-Level Causal Hierarchy

Pearl's three-level causal hierarchy includes:
1. Association - Involves purely statistical relationships defined by data (e.g., "What does a symptom tell me about a disease?")
2. Intervention - Involves changing what we observe (e.g., "What if I take aspirin, will my headache be cured?")
3. Counterfactual - Involves retrospective reasoning (e.g., "What if I had acted differently?")

Questions at each level can only be answered with information from that level or higher.

### Potential Outcomes Framework

To estimate causal effects, we must define notation that represents the potential results of a variable of interest as a function of a treatment variable. The potential outcomes are the values that the variable of interest would have assumed if the treatment variable had been set at a certain level.

Key components:
- Treatment variable (T): The intervention we're studying
- Outcome variable (Y): The result we're measuring
- Potential outcomes: Y(t=0) and Y(t=1) representing what would happen under each treatment condition

### The Missing Data Problem

We can never observe both potential outcomes for the same unit simultaneously - we either observe Y(1) or Y(0) for each individual, but never both. This creates the fundamental problem of causal inference.

## Methods for Causal Discovery and Inference

### Randomized Experiments

The gold standard for establishing causality:
- Random assignment to treatment/control groups
- Eliminates confounding by balancing all factors across groups
- Allows direct comparison of outcomes

### Causal Inference Methods for Observational Data

When randomization isn't possible, several techniques help estimate causal effects:

1. **Matching Methods**: Associate each treated unit with control units having similar characteristics to create a counterbalanced population where treatment is independent of confounders.

2. **Weighting Methods**: Assign weights to units to balance confounder distribution between treatment and control groups, creating a synthetic population where treatment is independent of confounders.

3. **Stratification Methods**: Divide units into homogeneous subgroups based on confounder values, creating a stratified population where treatment is independent of confounders within each stratum.

4. **Regression Discontinuity Design (RDD)**: Exploits discontinuities in variables that determine treatment allocation. Units close to the discontinuity threshold are considered similar, allowing comparison of outcomes just above and below the threshold.

5. **Instrumental Variables**: Variables that influence treatment but not outcomes except through treatment. Must satisfy relevance, exclusion, exchangeability, and monotonicity assumptions.

6. **Difference in Differences (DID)**: Leverages time trends common among treatment and control groups, comparing the difference in outcomes before and after an intervention.

### Causal Discovery Methods

Methods to infer causal structures from observational data:

1. **Constraint-based methods**: Use conditional independence relationships to recover a Markov equivalence class of the underlying causal structure. Requires assumptions like the causal Markov condition and faithfulness.

2. **Functional causal models**: Represent effects as functions of direct causes plus independent noise. The causal direction is identifiable because model assumptions hold only for the true causal direction. Examples include:
   - Linear, Non-Gaussian, Acyclic Model (LiNGAM)
   - Additive noise models
   - Post-nonlinear causal models

## Applications of Causal Inference

### Healthcare Applications

In healthcare, understanding causality helps determine whether changes like increased staffing actually cause improved outcomes like reduced wait times. Without causal analysis, hospitals might attribute improvements to staffing changes when other factors like decreased patient volume might be responsible.

### Policy Decisions

For public policy, establishing causality is essential to avoid ineffective or harmful decisions. When evaluating interventions like vaccine programs, it's crucial to determine whether outcome improvements are caused by the intervention itself or other factors.

### Domain Adaptation in Machine Learning

In machine learning, causal knowledge helps address domain adaptation problems where training and test data follow different distributions. Understanding the causal structure allows breaking down adaptation into specific cases:

- **Target Shift**: When marginal probability of outcomes varies across domains
- **Conditional Shift**: When the relationship between features and outcomes changes across domains
- **Generalized Target Shift**: Combination of both shifts

### Handling Selection Bias

Selection bias occurs when data collection processes introduce systematic errors. A causal perspective distinguishes two types:

- **Selection on cause**: When missingness depends only on causes
- **Selection on outcome**: When missingness depends on outcomes, requiring special handling

## Challenges and Limitations

Causal Inference faces several challenges:

1. Identification assumptions are often difficult to verify with available data
2. Assumptions can be violated by unobserved factors, creating spurious correlations
3. Generalizing causal effects to other populations or contexts requires additional assumptions
4. Interpreting and communicating results requires nuanced understanding

## Modern Approaches and Tools

### Machine Learning + Causal Inference

Recent advances like the knockoff method allow researchers to perform causal inference using modern ML algorithms. This approach creates artificial features that mimic original ones but don't cause the outcome, allowing statistical comparison of feature importance between real and knockoff features.

## Recommended Resources for Further Learning

- Causal Inference: What If by Miguel A. Hernán and James M. Robins
- Causal Inference in Statistics: A Primer by Judea Pearl, Madelyn Glymour, and Nicholas P. Jewell
- Causal Inference: The Mixtape by Scott Cunningham
