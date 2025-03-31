# Assignment 3: Quiz Questions Plan

## Overview
This plan outlines the approach for creating 15 multiple-choice questions testing understanding of causal inference concepts, methodologies, and practical applications. The questions will cover theoretical foundations, implementation techniques, and interpretation challenges across the three datasets used in Assignments 1 and 2 (IHDP, Hotel Bookings, and Lalonde).

## Question Structure Guidelines

### Format Requirements
- Each question must have multiple correct options
- All questions must include detailed explanations for why each option is correct or incorrect
- Questions should vary in difficulty level (basic, intermediate, advanced)

### Topics Distribution
- Theoretical concepts: 5 questions
- Implementation techniques: 5 questions
- Results interpretation: 5 questions

### Dataset Integration
- IHDP dataset: 5 questions
- Hotel Bookings dataset: 5 questions
- Lalonde dataset: 5 questions

## Question Types and Templates

### Theoretical Concept Questions

1. **Causal Framework Understanding**
   - Question about distinctions between Rubin's potential outcomes and Pearl's structural causal models
   - Options covering key aspects of each framework
   - Tests understanding of fundamental causal reasoning approaches

2. **Confounding and Identification**
   - Question about identifying confounders in causal graphs
   - Options with various variable relationships
   - Tests ability to recognize backdoor paths

3. **Causal Assumptions**
   - Question about required assumptions for causal inference
   - Options covering various assumptions (unconfoundedness, SUTVA, etc.)
   - Tests understanding of when causal effects are identifiable

4. **Ladder of Causation**
   - Question about Pearl's three levels of causation
   - Options distinguishing between association, intervention, and counterfactuals
   - Tests understanding of causal hierarchy

5. **Causal Discovery**
   - Question about methods for learning causal structure from data
   - Options covering constraint-based and score-based approaches
   - Tests understanding of automated causal discovery limitations

### Implementation Technique Questions

6. **Matching Methods**
   - Question about propensity score matching implementation
   - Options covering different matching algorithms and distance metrics
   - Tests understanding of practical matching considerations

7. **Estimation Methods Comparison**
   - Question about selecting appropriate estimation methods
   - Options comparing strengths/weaknesses of different approaches
   - Tests ability to choose methods based on data characteristics

8. **Refutation Techniques**
   - Question about validating causal inferences
   - Options covering various refutation approaches
   - Tests understanding of sensitivity analysis

9. **DoWhy Implementation**
   - Question about DoWhy's four-step workflow
   - Options covering modeling, identification, estimation, and refutation
   - Tests practical implementation knowledge

10. **Machine Learning for Causality**
    - Question about using ML for heterogeneous treatment effects
    - Options covering meta-learners and specialized causal algorithms
    - Tests understanding of modern causal ML approaches

### Results Interpretation Questions

11. **IHDP Effect Interpretation**
    - Question about interpreting treatment effects in healthcare context
    - Options covering various aspects of the IHDP treatment effect
    - Tests ability to translate statistical results to policy implications

12. **Hotel Cancellation Interventions**
    - Question about interpreting causal effects on booking cancellations
    - Options for potential business interventions based on causal findings
    - Tests ability to derive actionable insights from causal analysis

13. **Lalonde Program Evaluation**
    - Question about interpreting job training effects
    - Options covering effect heterogeneity and policy implications
    - Tests understanding of social program evaluation

14. **Addressing Selection Bias**
    - Question about recognizing and addressing selection bias
    - Options covering various approaches across datasets
    - Tests ability to handle common observational data challenges

15. **Causal Effect Visualization**
    - Question about effective visualization of treatment effects
    - Options for different visualization approaches
    - Tests ability to communicate causal findings clearly

## Sample Question Format

```
Question X: [Question text]

A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
E) [Option E]

Correct answers: [List correct options]

Explanations:
- Option A: [Detailed explanation of why this option is correct/incorrect]
- Option B: [Detailed explanation of why this option is correct/incorrect]
- Option C: [Detailed explanation of why this option is correct/incorrect]
- Option D: [Detailed explanation of why this option is correct/incorrect]
- Option E: [Detailed explanation of why this option is correct/incorrect]

Difficulty level: [Basic/Intermediate/Advanced]
Dataset referenced: [IHDP/Hotel Bookings/Lalonde]
Topic area: [Theoretical/Implementation/Interpretation]
```

## Sample Question (Fully Developed)

```
Question 1: Which of the following correctly describe the "backdoor criterion" in causal inference?

A) It identifies a set of variables that, when controlled for, block all backdoor paths between treatment and outcome
B) It requires conditioning on all parents of the treatment variable
C) It helps determine if a causal effect is identifiable from observational data
D) It requires that no colliders are conditioned on in backdoor paths
E) It is used exclusively in Pearl's structural causal model framework and has no equivalent in the potential outcomes framework

Correct answers: A, C, D

Explanations:
- Option A: CORRECT. The backdoor criterion specifically identifies variables that must be controlled for to block all backdoor paths between treatment and outcome, eliminating confounding bias.
- Option B: INCORRECT. The backdoor criterion does not necessarily require conditioning on all parents of the treatment. It requires conditioning on variables that block all backdoor paths, which may be a subset of the parents or include other variables.
- Option C: CORRECT. The backdoor criterion is a key tool for determining if and how a causal effect can be identified from observational data by specifying the variables needed to control for confounding.
- Option D: CORRECT. Conditioning on colliders in backdoor paths creates, rather than blocks, spurious associations. The backdoor criterion explicitly requires that no colliders on backdoor paths are conditioned on.
- Option E: INCORRECT. While the backdoor criterion is formulated in terms of causal graphs in Pearl's framework, it has equivalents in the potential outcomes framework, specifically corresponding to the unconfoundedness assumption.

Difficulty level: Intermediate
Dataset referenced: Not dataset-specific
Topic area: Theoretical
```

## Development Process

1. **Research and Question Drafting**
   - Review key concepts from the theoretical material
   - Identify common misconceptions and challenging aspects of causal inference
   - Draft initial questions with multiple answer options

2. **Question Refinement**
   - Ensure questions test understanding rather than memorization
   - Validate that each question has multiple correct answers
   - Incorporate scenario-based questions using the three datasets

3. **Explanation Development**
   - Write comprehensive explanations for each option
   - Include references to theoretical concepts and practical implementations
   - Ensure explanations are educational, even for incorrect options

4. **Quality Assurance**
   - Review questions for clarity and accuracy
   - Verify difficulty levels are appropriately distributed
   - Cross-check explanations with reference materials

5. **Formatting and Submission**
   - Format according to required guidelines
   - Organize questions in a logical progression
   - Prepare final document for submission to Canvas

## Timeline
1. Question topic identification and planning: 1 day
2. Drafting questions and answer options: 2 days
3. Writing detailed explanations: 2 days
4. Review and refinement: 1 day
5. Final formatting and submission preparation: 1 day
