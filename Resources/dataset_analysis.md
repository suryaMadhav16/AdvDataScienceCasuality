# Causal Inference Datasets Analysis

This document provides a detailed analysis of the datasets selected for the causal inference assignments, explaining what they are, why they're valuable for learning, and how they will be used in the project.

## 1. IHDP (Infant Health and Development Program) Dataset

### What It Is
The IHDP dataset is based on a randomized controlled study designed to evaluate the effect of specialized childcare on the cognitive test scores of premature infants. It contains:
- 747 subjects (139 treated, 608 control)
- 25 pre-treatment variables (6 continuous, 19 binary)
- Treatment: specialized childcare and home visits
- Outcome: cognitive test scores

### Why It's Worth Working With
- **Semi-synthetic nature**: While based on a real experiment, the outcomes are simulated, providing ground truth for validation
- **Balanced structure**: Contains a good mix of binary and continuous variables
- **Well-documented**: Extensively used in causal inference literature
- **Moderate size**: Large enough to be meaningful but small enough to be computationally tractable
- **Domain relevance**: Healthcare interventions represent a classic causal inference problem

### How It Will Be Used
This dataset will be the primary focus for Assignment 1, demonstrating the complete causal inference workflow:
- Data preprocessing and exploration
- Causal graph specification
- Identification strategies with the backdoor criterion
- Multiple estimation methods (regression, matching, weighting)
- Refutation techniques to validate results

## 2. Hotel Booking Cancellations Dataset

### What It Is
This dataset contains booking information from two hotels (a resort hotel and a city hotel), with details about bookings, including whether they were canceled. Key features include:
- Booking changes
- Lead time between booking and arrival
- Room type assignment (reserved vs. assigned)
- Special requests information
- Customer demographics

### Why It's Worth Working With
- **Accessible context**: Hotel bookings are universally understood, making the causal questions relatable
- **Business relevance**: Demonstrates causal inference in a business setting with clear ROI implications
- **Multiple potential causal questions**: Allows exploration of various treatments (room assignment, deposit policies, booking changes)
- **Rich feature set**: Contains many potential confounders and mediators
- **DoWhy example available**: Has a well-documented example notebook to reference

### How It Will Be Used
This dataset will be used for the first worked example in Assignment 2:
- Exploring the causal effect of different room assignments on booking cancellations
- Implementing the DoWhy four-step workflow
- Focusing on propensity score methods and refutation
- Demonstrating business-relevant interpretations of causal findings

## 3. Lalonde Jobs Dataset

### What It Is
The Lalonde dataset examines the effect of job training programs on subsequent employment status and earnings. It combines:
- Experimental data from the National Supported Work (NSW) Program
- Observational data from the Panel Study of Income Dynamics (PSID)
- Features include demographic variables (age, education, race, previous earnings)
- Treatment: participation in job training program
- Outcome: employment status and earnings in 1978

### Why It's Worth Working With
- **Historical significance**: A seminal dataset in causal inference literature
- **Real policy relevance**: Addresses an important social policy question
- **Combined experimental/observational**: Allows direct comparison of results from different study designs
- **Well-studied**: Extensive literature with benchmark results for comparison
- **Socially relevant**: Examines an intervention with clear social importance

### How It Will Be Used
This dataset will be used for the second worked example in Assignment 2:
- Demonstrating more advanced causal estimation techniques
- Comparing observational estimates with experimental benchmarks
- Addressing selection bias challenges
- Discussing policy implications of causal findings

## Dataset Difficulty Levels and Learning Progression

| Dataset | Difficulty | Key Learning Focus |
|---------|------------|-------------------|
| IHDP | Easy to Medium | Foundational causal inference concepts, core workflow |
| Hotel Bookings | Medium | Business applications, refutation techniques |
| Lalonde | Medium to Hard | Selection bias, policy evaluation, advanced methods |

This progression allows building knowledge incrementally:
1. Start with the structured IHDP dataset to learn core concepts
2. Move to the Hotel Bookings dataset to see business applications
3. Progress to the Lalonde dataset to tackle more complex methodological challenges

## Integration with Assignment Plan

The datasets have been carefully selected to align with the assignment structure:

1. **Assignment 1**: IHDP dataset will be used to demonstrate the complete causal inference workflow, from model specification to refutation.

2. **Assignment 2**:
   - First example: Hotel Bookings dataset to demonstrate business applications
   - Second example: Lalonde dataset to explore more complex methods and policy evaluation

3. **Assignment 3**: Quiz questions will cover concepts from all three datasets, testing understanding of different causal contexts.

This integration ensures a comprehensive learning experience that progressively builds expertise in causal inference across different domains and methodological challenges.
