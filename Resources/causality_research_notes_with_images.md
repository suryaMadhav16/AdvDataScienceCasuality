# Causality in Data Science - Research Notes

## What is Causality?

Causality refers to the relationship between cause and effect. In data science, causal inference explores how variables influence each other beyond simple correlation, which is crucial for making accurate predictions and informed decisions.

In data science, establishing causality is about understanding how one action directly influences another. It's the fundamental difference between recognizing patterns and making data-driven decisions that actually work.

## The Importance of Causality vs. Correlation

Causal Inference is a very relevant subject for Data Science, as it allows us to go beyond the simple description of data and to understand what are the effects of a certain action or intervention on a variable of interest. 

The famous phrase "correlation does not imply causation" is critical in data science. For example:
- Ice cream sales and drowning incidents both increase in summer, but eating ice cream doesn't cause drowning
- Both are correlated because they share a common cause (summer weather)

![Correlation vs Causation](https://imgs.xkcd.com/comics/correlation.png)
*The classic XKCD comic illustrating the correlation vs. causation concept*

## Key Concepts in Causal Inference

### Pearl's Three-Level Causal Hierarchy

Pearl's three-level causal hierarchy includes:
1. Association - Involves purely statistical relationships defined by data (e.g., "What does a symptom tell me about a disease?")
2. Intervention - Involves changing what we observe (e.g., "What if I take aspirin, will my headache be cured?")
3. Counterfactual - Involves retrospective reasoning (e.g., "What if I had acted differently?")

Questions at each level can only be answered with information from that level or higher.

![Pearl's Ladder of Causation](https://miro.medium.com/v2/resize:fit:720/format:webp/1*v_16OvtnrqLAu5OuHl2ykA.png)
*Pearl's Ladder of Causation showing the three levels: Association, Intervention, and Counterfactual*

### Directed Acyclic Graphs (DAGs)

DAGs are a fundamental tool in causal inference that provide visual representations of causal relationships among variables. Each node in a DAG represents a variable, and each arrow represents a direct causal effect.

![Basic DAG Example](https://cdn.jsdelivr.net/gh/yubaoliu/ImageHosting@main/images/image-20210522202248070.png)
*A simple example of a Directed Acyclic Graph (DAG) showing causal relationships*

Key elements of DAGs:
- Nodes represent variables
- Directed edges (arrows) represent causal relationships
- No cycles allowed (hence "acyclic")
- Time flows from left to right by convention

### Potential Outcomes Framework

To estimate causal effects, we must define notation that represents the potential results of a variable of interest as a function of a treatment variable. The potential outcomes are the values that the variable of interest would have assumed if the treatment variable had been set at a certain level.

Key components:
- Treatment variable (T): The intervention we're studying
- Outcome variable (Y): The result we're measuring
- Potential outcomes: Y(t=0) and Y(t=1) representing what would happen under each treatment condition

![Potential Outcomes](https://miro.medium.com/v2/resize:fit:720/format:webp/1*mUZofFLSvUFwWhK5xS2YSw.png)
*Visualization of the potential outcomes framework showing observed and counterfactual outcomes*

### The Missing Data Problem

We can never observe both potential outcomes for the same unit simultaneously - we either observe Y(1) or Y(0) for each individual, but never both. This creates the fundamental problem of causal inference.

## Types of Relationships in DAGs

![DAG Relationships](https://causaldiagrams.org/wp-content/uploads/2022/03/10a-Checking-for-confounding.png)
*Different relationship patterns in causal diagrams*

### Common Causal Structures

1. **Confounding**: A variable affects both the treatment and outcome
   
   ![Confounding](https://www.frontiersin.org/files/Articles/534261/fpubh-08-00054-HTML/image_m/fpubh-08-00054-g001.jpg)
   *Confounding structure where variable C affects both X and Y*

2. **Mediation**: Treatment affects an intermediate variable which affects the outcome
   
   ![Mediation](https://www.researchgate.net/publication/339882566/figure/fig1/AS:868353102864384@1584028686276/Example-of-mediator-M-between-exposure-X-and-outcome-Y-with-confounders-C.ppm)
   *Mediation structure where X affects M which affects Y*

3. **Collider**: Two variables both affect a third variable
   
   ![Collider](https://i0.wp.com/epiresearch.org/wp-content/uploads/2018/06/collider.png)
   *Collider structure where both X and Y affect variable C*

## Methods for Causal Discovery and Inference

### Randomized Experiments

The gold standard for establishing causality:
- Random assignment to treatment/control groups
- Eliminates confounding by balancing all factors across groups
- Allows direct comparison of outcomes

![Randomized Experiment](https://cdn.scribbr.com/wp-content/uploads/2023/09/randomized-experiment-1.webp)
*Structure of a randomized experiment design*

### Causal Inference Methods for Observational Data

When randomization isn't possible, several techniques help estimate causal effects:

1. **Matching Methods**: Associate each treated unit with control units having similar characteristics to create a counterbalanced population where treatment is independent of confounders.

   ![Matching](https://www.researchgate.net/publication/332260069/figure/fig1/AS:743883471163392@1554367849375/Matching-methods-for-causal-inference-a-Before-matching-the-distributions-of-the.png)
   *Visualization of matching process to balance treatment and control groups*

2. **Weighting Methods**: Assign weights to units to balance confounder distribution between treatment and control groups, creating a synthetic population where treatment is independent of confounders.

3. **Stratification Methods**: Divide units into homogeneous subgroups based on confounder values, creating a stratified population where treatment is independent of confounders within each stratum.

4. **Regression Discontinuity Design (RDD)**: Exploits discontinuities in variables that determine treatment allocation. Units close to the discontinuity threshold are considered similar, allowing comparison of outcomes just above and below the threshold.

   ![RDD](https://ars.els-cdn.com/content/image/3-s2.0-B9780128167267000032-f03-02-9780128167267.jpg)
   *Example of Regression Discontinuity Design showing treatment assignment at a threshold*

5. **Instrumental Variables**: Variables that influence treatment but not outcomes except through treatment. Must satisfy relevance, exclusion, exchangeability, and monotonicity assumptions.

   ![Instrumental Variables](https://www.frontiersin.org/files/Articles/534261/fpubh-08-00054-HTML/image_m/fpubh-08-00054-g004.jpg)
   *Instrumental variable (Z) affects treatment (X) which affects outcome (Y)*

6. **Difference in Differences (DID)**: Leverages time trends common among treatment and control groups, comparing the difference in outcomes before and after an intervention.

   ![DID](https://miro.medium.com/v2/resize:fit:1400/1*88aJ0cq7kmUOTnGaXtfobQ.png)
   *Difference in Differences methodology comparing two groups over time*

### Causal Discovery Methods

Methods to infer causal structures from observational data:

1. **Constraint-based methods**: Use conditional independence relationships to recover a Markov equivalence class of the underlying causal structure. Requires assumptions like the causal Markov condition and faithfulness.

2. **Functional causal models**: Represent effects as functions of direct causes plus independent noise. The causal direction is identifiable because model assumptions hold only for the true causal direction. Examples include:
   - Linear, Non-Gaussian, Acyclic Model (LiNGAM)
   - Additive noise models
   - Post-nonlinear causal models

## The Backdoor Criterion

The backdoor criterion is a graphical test used to identify which sets of variables need to be controlled for to estimate causal effects from observational data.

![Backdoor Criterion](https://pbs.twimg.com/media/FIOgVLlWYAMGNMf.jpg)
*Illustration of the backdoor path concept in causal diagrams*

A set of variables Z satisfies the backdoor criterion relative to (X,Y) if:
1. No node in Z is a descendant of X
2. Z blocks every path between X and Y that contains an arrow into X

## Applications of Causal Inference

### Healthcare Applications

In healthcare, understanding causality helps determine whether changes like increased staffing actually cause improved outcomes like reduced wait times. Without causal analysis, hospitals might attribute improvements to staffing changes when other factors like decreased patient volume might be responsible.

![Healthcare Causal Example](https://www.researchgate.net/publication/322641889/figure/fig2/AS:667609239240709@1536179777430/A-causal-diagram-solid-arrows-for-the-relation-between-trauma-center-care-and-mortality.png)
*Causal diagram example from healthcare research*

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

![Selection Bias](https://www.researchgate.net/publication/296624368/figure/fig1/AS:613899864489984@1523375323212/Directed-acyclic-graph-for-selection-bias-S-is-the-selection-indicator-A-is-a-treatment.png)
*DAG showing selection bias where S is the selection indicator*

## Challenges and Limitations

Causal Inference faces several challenges:

1. Identification assumptions are often difficult to verify with available data
2. Assumptions can be violated by unobserved factors, creating spurious correlations
3. Generalizing causal effects to other populations or contexts requires additional assumptions
4. Interpreting and communicating results requires nuanced understanding

## Modern Approaches and Tools

### Machine Learning + Causal Inference

Recent advances like the knockoff method allow researchers to perform causal inference using modern ML algorithms. This approach creates artificial features that mimic original ones but don't cause the outcome, allowing statistical comparison of feature importance between real and knockoff features.

![ML and Causality](https://miro.medium.com/v2/resize:fit:1400/1*KbCCPyWALTfMbcvvTjZ8rg.png)
*Integration of machine learning and causal inference approaches*

## Guidelines for Creating DAGs

1. Start with your research question: exposure/treatment (A) and outcome (Y)
2. Add all variables that directly affect either A or Y
3. Add variables that affect any two other variables in the DAG
4. Draw arrows to indicate causal relationships
5. Time should flow from left to right

![DAG Creation Process](https://bookdown.org/jbrophy115/bookdown-clinepi/causal_files/figure-html/unnamed-chunk-3-1.png)
*Example of a step-by-step DAG creation process*

## Recommended Resources for Further Learning

- Causal Inference: What If by Miguel A. Hernán and James M. Robins
- Causal Inference in Statistics: A Primer by Judea Pearl, Madelyn Glymour, and Nicholas P. Jewell
- Causal Inference: The Mixtape by Scott Cunningham
- [DAGitty](http://dagitty.net/) - A web-based tool for creating and analyzing causal diagrams
