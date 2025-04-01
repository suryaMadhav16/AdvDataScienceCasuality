import marimo

__generated_with = "0.10.16"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.image(src="https://imgs.xkcd.com/comics/correlation_2x.png").center()
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    return LinearRegression, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Understanding Causal Inference with IHDP: From Theory to Practice""")
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        toc_md = mo.md("""
        ## Table of Contents

        1. [Introduction to Causal Inference](#introduction)
        2. [Theoretical Foundations](#foundations)
           - [Real-World Applications](#applications)
           - [Key Concepts](#concepts)
           - [Treatment Effects](#effects)
           - [Key Assumptions](#assumptions)
           - [Advanced Causal Inference Methods](#advanced-methods)
           - [Key Terms in Causal Inference](#key-terms)
        3. [The IHDP Dataset](#ihdp-intro)
        4. [Exploratory Data Analysis](#eda)
           - [Dataset Overview](#overview)
           - [Covariate Analysis](#covariates)
        5. [Setting Up for Causal Analysis](#analysis-setup)
           - [Data Preparation](#data-prep)
           - [Formulating the Causal Question](#causal-question)
           - [Propensity Score Analysis](#propensity-analysis)
        6. [Future Implementations](#future)
           - [Machine Learning Methods for Causal Inference](#ml-methods)
        """)
        mo.output.replace(toc_md)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        section_header = mo.md("""## 6. Future Implementations {#future}

        This section will include implementations of various causal inference methods:

        - Basic regression approaches
        - Matching methods
        - Propensity score methods
        - Advanced techniques like doubly robust estimation
        - Machine learning methods for causal inference
        """)
        mo.output.replace(section_header)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        ml_header = mo.md("### 6.1 Machine Learning Methods for Causal Inference {#ml-methods}")

        ml_description = mo.md("""
        Machine learning approaches are increasingly being used to estimate causal effects, especially in high-dimensional settings where traditional methods may not perform well. These methods can help discover complex relationships in the data while maintaining causal interpretability.

        The following machine learning methods will be implemented in future updates:
        """)

        ml_methods = mo.hstack([
            mo.callout(
                mo.md("""
                #### Causal Forests

                An extension of random forests designed specifically for causal inference. Causal forests:
                - Estimate heterogeneous treatment effects
                - Adapt to complex, non-linear relationships
                - Provide valid statistical inference
                - Are implemented in the `grf` R package and `econml` Python package

                Causal forests partition the feature space to find regions with similar treatment effects, making them powerful tools for personalized treatment effect estimation.
                """),
                kind="info"
            ),
            mo.callout(
                mo.md("""
                #### Double/Debiased Machine Learning

                A framework that combines machine learning with econometric techniques to provide robust causal estimates. It addresses bias concerns by:
                - Using sample splitting (cross-fitting)
                - Employing orthogonalization techniques
                - Focusing on a low-dimensional causal parameter of interest
                - Allowing for flexible ML models for nuisance functions

                This approach yields valid inference even when using complex black-box models for estimating propensity scores or outcome models.
                """),
                kind="success"
            )
        ])

        more_ml_methods = mo.hstack([
            mo.callout(
                mo.md("""
                #### Meta-Learners

                A class of algorithms that separate the estimation of treatment effects into multiple stages:
                - **S-learner**: Fits a single model with treatment as a feature
                - **T-learner**: Fits separate models for treated and control groups
                - **X-learner**: Improves on T-learner by incorporating treatment propensities
                - **R-learner**: Directly targets the treatment effect by orthogonalizing outcome and treatment

                These approaches provide flexible frameworks for incorporating any machine learning method into causal inference.
                """),
                kind="warn"
            ),
            mo.callout(
                mo.md("""
                #### Causal Representation Learning

                Cutting-edge approaches that learn representations of data that capture causal structure:
                - Deep latent variable models that separate confounders from predictive features
                - Neural networks designed to learn causal mechanisms
                - Methods that combine domain knowledge with data-driven approaches

                These methods are particularly useful when dealing with unstructured data like images or text, or when explicit causal assumptions are difficult to encode.
                """),
                kind="danger"
            )
        ])

        note = mo.callout(
            mo.md("""
            **Implementation Plan**: Each method will be implemented with the IHDP dataset, compared against the known ground truth effects, and evaluated on its ability to capture both average and heterogeneous treatment effects. We'll provide code examples, visualizations of results, and discussions of strengths and limitations.
            """),
            kind="info"
        )

        mo.output.replace(mo.vstack([ml_header, ml_description, ml_methods, more_ml_methods, note]))
    _()
    return


@app.cell(hide_code=True)
def _(T_train, X_train_scaled, pd):
    # Estimate propensity scores using logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, roc_curve

    # Fit logistic regression model to estimate propensity scores
    propensity_model = LogisticRegression(max_iter=1000, C=1.0)
    propensity_model.fit(X_train_scaled, T_train)

    # Calculate propensity scores (probability of receiving treatment)
    propensity_scores = propensity_model.predict_proba(X_train_scaled)[:, 1]

    # Also estimate propensity scores using a Random Forest for comparison
    rf_propensity_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
    rf_propensity_model.fit(X_train_scaled, T_train)
    rf_propensity_scores = rf_propensity_model.predict_proba(X_train_scaled)[:, 1]

    # Create DataFrame with propensity scores
    ps_df = pd.DataFrame({
        'treatment': T_train,
        'ps_logistic': propensity_scores,
        'ps_rf': rf_propensity_scores
    })

    # Evaluate propensity score models
    logistic_auc = roc_auc_score(T_train, propensity_scores)
    rf_auc = roc_auc_score(T_train, rf_propensity_scores)

    print(f"Logistic Regression AUC: {logistic_auc:.4f}")
    print(f"Random Forest AUC: {rf_auc:.4f}")

    # Return models and scores for future use
    return (
        LogisticRegression,
        RandomForestClassifier,
        logistic_auc,
        propensity_model,
        propensity_scores,
        ps_df,
        rf_auc,
        rf_propensity_model,
        rf_propensity_scores,
        roc_auc_score,
        roc_curve,
    )


@app.cell(hide_code=True)
def _(mo):
    def _():
        section_header = mo.md("""## 1. Introduction to Causal Inference {#introduction}""")
        mo.output.replace(section_header)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("#Correlation does not imply causation"), kind="danger")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        In predictive modeling, we often focus on finding correlations between variables. However, for decision-making, we need to understand the *causal* relationship between actions and outcomes.

        The fundamental problem of causal inference is that we can never observe both potential outcomes for the same unit - we can't simultaneously observe what happens when a person receives a treatment and doesn't receive a treatment.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout("""
    In causal inference, a confounder is a variable that affects both the treatment (or independent variable) and the outcome (or dependent variable), potentially creating a spurious association if not controlled for. For example, when studying the effect of alcohol consumption on lung cancer risk, smokers tend to drink more and smoking is a direct cause of lung cancer, so smoking acts as a confounder that can distort the observed relationship between alcohol and cancer.
    """, kind="info")
    return


@app.cell(hide_code=True)
def _(LinearRegression, mo, np, pd, plt):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate data to illustrate correlation vs causation
    n = 1000

    # Common cause (confounder)
    confounder = np.random.normal(0, 1, n)

    # Treatment influenced by confounder
    treatment = 0.7 * confounder + np.random.normal(0, 0.5, n)

    # Outcome influenced by both treatment and confounder
    outcome = 0.3 * treatment + 0.7 * confounder + np.random.normal(0, 0.5, n)

    # Create a DataFrame
    data = pd.DataFrame({
        'Treatment': treatment,
        'Outcome': outcome,
        'Confounder': confounder
    })

    # Create model for the regression line
    model = LinearRegression()
    model.fit(data[['Treatment']], data['Outcome'])


    fig_1 = plt.figure(figsize=(12, 5))

    # Plot 1: Treatment vs Outcome (shows correlation)
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(data['Treatment'], data['Outcome'], alpha=0.5)

    # Add regression line
    x_range = np.linspace(data['Treatment'].min(), data['Treatment'].max(), 100)
    ax1.plot(x_range, model.predict(x_range.reshape(-1, 1)), 'r-', linewidth=2)
    ax1.set_title('Correlation: Treatment vs Outcome\nCorrelation = {:.2f}'.format(
        np.corrcoef(data['Treatment'], data['Outcome'])[0, 1]))
    ax1.set_xlabel('Treatment')
    ax1.set_ylabel('Outcome')

    # Plot 2: Treatment vs Outcome with confounder as color
    ax2 = plt.subplot(1, 2, 2)
    scatter = ax2.scatter(data['Treatment'], data['Outcome'], c=data['Confounder'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Confounder')
    ax2.set_title('Causal Structure: Treatment, Outcome, and Confounder')
    ax2.set_xlabel('Treatment')
    ax2.set_ylabel('Outcome')

    plt.tight_layout()

    # Return interactive plot for marimo
    mo.mpl.interactive(fig_1)
    return (
        ax1,
        ax2,
        confounder,
        data,
        fig_1,
        model,
        n,
        outcome,
        scatter,
        treatment,
        x_range,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
        <h3>Left Plot</h3>
        Shows the correlation between treatment and outcome (0.78), with a regression line indicating a strong positive relationship. This is what you might see in an observational study without accounting for confounders.
        </div>
        <div>
        <h3>Right Plot</h3>
        The same data points, but colored by the confounder value. This reveals the underlying structure - points with similar confounder values cluster together, showing that the apparent treatment-outcome relationship is largely driven by the confounder.
        </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout("""
    The simulation demonstrates that despite seeing a strong correlation (0.78), the actual causal effect of the treatment on the outcome is weaker (0.3 in the data generation). The rest of the association is due to the confounder creating a spurious correlation - a classic example of "correlation does not imply causation."
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        section_header = mo.md("""## 2. Theoretical Foundations of Causal Inference {#foundations}""")
        mo.output.replace(section_header)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    # Creating application domain columns
    healthcare = mo.md(
    """
    ### Healthcare
    - Evaluating treatment effectiveness
    - Understanding disease progression
    - Personalizing medical decisions
    """
    )

    policy = mo.md(
    """
    ### Policy
    - Program evaluation
    - Social interventions assessment
    - Education policy design
    """
    )

    business = mo.md(
    """
    ### Business
    - Marketing strategy optimization
    - Product feature evaluations
    - Customer retention strategies
    """
    )

    # Real-world applications section with columns
    applications_title = mo.md("### 2.1 Real-World Applications {#applications}\n\nCausal inference is crucial in various domains:")
    applications_columns = mo.hstack([healthcare, policy, business])

    # Key concepts section
    concepts = mo.md(
    """
    ### 2.2 Key Concepts in Causal Inference {#concepts}

    **The Potential Outcomes Framework**

    Developed by Rubin, this framework formalizes causal inference through potential outcomes. For each unit i:
    - Y_i(1): Outcome if unit i receives treatment
    - Y_i(0): Outcome if unit i doesn't receive treatment

    The individual treatment effect is defined as:

    \\[ \\tau_i = Y_i(1) - Y_i(0) \\]

    However, we can only observe one of these outcomes for each unit, which is known as the **fundamental problem of causal inference**.
    """
    )

    # Display all sections
    mo.vstack([applications_title, applications_columns, concepts])
    return (
        applications_columns,
        applications_title,
        business,
        concepts,
        healthcare,
        policy,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### 2.3 Types of Treatment Effects in Causal Inference {#effects}

        ### What are "Treatments" in Causal Inference?

        In causal inference, a **treatment** refers to the intervention or manipulation being studied to determine its causal effect on an outcome of interest. Despite the medical-sounding terminology, treatments extend far beyond healthcare settings:

        - Medical interventions (medications, surgical procedures)
        - Policy changes (minimum wage increases, educational reforms) 
        - Business decisions (pricing strategies, marketing campaigns)
        - Social interventions (training programs, behavioral modifications)

        The **treatment variable** typically represents whether subjects received the intervention, usually coded as a binary variable (1=received treatment, 0=control/placebo), though it can sometimes be categorical for different treatment types or doses.

        ### Why Calculate Treatment Effects?

        Treatment effects measure the causal effect of a treatment on an outcome. We calculate them to:

        1. **Establish causality, not just correlation**: Determine the independent effect of a treatment when other factors are controlled for
        2. **Understand counterfactuals**: Estimate what would have happened to treated units had they not received treatment (and vice versa)
        3. **Quantify impact**: Measure not just whether an intervention worked, but how well it worked and for whom
        4. **Inform decision-making**: Make better decisions about implementing interventions and targeting specific populations

        ### Key Treatment Effect Measures

        **Average Treatment Effect (ATE):**
        The average effect of the treatment across the entire population.

        \\[ ATE = E[Y(1) - Y(0)] \\]

        Where Y(1) represents the potential outcome if treated, and Y(0) represents the potential outcome if not treated. This measures the expected difference in outcomes if everyone in the population received treatment versus if no one did.

        **Conditional Average Treatment Effect (CATE):**
        The average effect of the treatment conditional on specific covariates or characteristics.

        \\[ CATE(X=x) = E[Y(1) - Y(0) | X=x] \\]

        This measures how treatment effects vary across different subgroups defined by characteristics X. CATE helps identify which groups benefit most from treatment, enabling more targeted interventions.

        **Average Treatment Effect on the Treated (ATT/ATET):**
        The average effect among those who actually received the treatment.

        \\[ ATT = E[Y(1) - Y(0) | T=1] \\]

        This answers: "How much did those who received the treatment actually benefit?" It's particularly useful when evaluating programs that were targeted at specific populations or when treatment assignment wasn't random.

        ### Challenges in Estimation

        A fundamental challenge is that we never observe both potential outcomes for the same unit—known as the "fundamental problem of causal inference". Various methods address this challenge, including:

        - Randomized experiments
        - Regression adjustment
        - Matching methods
        - Instrumental variables
        - Inverse probability weighting
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Create the header
    header = mo.md("### 2.4 Key Assumptions in Causal Inference {#assumptions}")

    # Create separate elements for each assumption as callouts with fixed height
    assumption1 = mo.callout(
        mo.md(
            """
            ### 1. Unconfoundedness (Ignorability)
            Treatment assignment is independent of potential outcomes given observed covariates.

            \\[ (Y(0), Y(1)) \\perp T | X \\]
            """
        ),
        kind="info"
    )

    assumption2 = mo.callout(
        mo.md(
            """
            ### 2. Positivity (Overlap)
            Every unit has a non-zero probability of receiving either treatment or control.

            \\[ 0 < P(T=1|X=x) < 1 \\text{ for all } x \\]
            """
        ),
        kind="warn"
    )

    assumption3 = mo.callout(
        mo.md(
            """
            ### 3. SUTVA
            **Stable Unit Treatment Value Assumption**

            * No interference: One unit's treatment doesn't affect another unit's outcome
            * No hidden variations of treatment
            """
        ),
        kind="success"
    )

    # Stack the header and the columns of assumptions
    mo.vstack([
        header,
        mo.hstack([assumption1, assumption2, assumption3])
    ])
    return assumption1, assumption2, assumption3, header


@app.cell(hide_code=True)
def _(mo, np, plt):
    def _():
        # Visualize the overlap assumption
        # Use local variables with unique names to avoid conflicts
        sample_size = 1000  # Instead of reusing 'n'

        # Create two features
        X1 = np.random.normal(0, 1, sample_size)
        X2 = np.random.normal(0, 1, sample_size)

        # Scenario 1: Good overlap
        p_good = 1 / (1 + np.exp(-(0.5 * X1)))
        treatment_good = np.random.binomial(1, p_good, sample_size)

        # Scenario 2: Poor overlap
        p_poor = 1 / (1 + np.exp(-(3 * X1 + 2 * X2)))
        treatment_poor = np.random.binomial(1, p_poor, sample_size)

        # Plot - use a different variable name than 'fig'
        overlap_fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Good overlap
        axes[0].scatter(X1, X2, c=treatment_good, cmap='coolwarm', alpha=0.6)
        axes[0].set_title('Good Overlap')
        axes[0].set_xlabel('X1')
        axes[0].set_ylabel('X2')

        # Poor overlap
        axes[1].scatter(X1, X2, c=treatment_poor, cmap='coolwarm', alpha=0.6)
        axes[1].set_title('Poor Overlap (Positivity Violation)')
        axes[1].set_xlabel('X1')
        axes[1].set_ylabel('X2')

        plt.tight_layout()

        # Use marimo's display method instead of plt.show()
        return mo.mpl.interactive(overlap_fig)


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        # Create a header for the DAG section
        header = mo.md("### 2.5 Interactive Causal Diagram")

        # Create a mermaid diagram for the IHDP causal structure
        diagram = mo.mermaid("""
        graph TB
            subgraph "Covariates"
                BR["Birth Related\nx_0, x_1, x_2, x_5, x_6"]
                M["Mother's Characteristics\nx_3, x_4, x_8, x_13-x_16"]
                P["Pregnancy Behaviors\nx_9, x_10, x_11, x_18, x_19"]
                S["Socioeconomic\nx_17"]
                L["Location\nx_20-x_24"]
            end

            BR --> T["Treatment\nIHDP Intervention"]
            M --> T
            P --> T
            S --> T
            L --> T

            BR --> Y["Outcome\nCognitive Score"]
            M --> Y
            P --> Y
            S --> Y

            T --> Y

            style T fill:#ff9999
            style Y fill:#99ccff
            style BR fill:#f9f9f9
            style M fill:#f9f9f9
            style P fill:#f9f9f9
            style S fill:#f9f9f9
            style L fill:#f9f9f9
        """)

        explanation = mo.md("""
        This directed acyclic graph (DAG) represents the assumed causal structure in the IHDP dataset:

        - **Covariates** (various characteristics) affect both treatment assignment and outcomes
        - **Treatment** (IHDP intervention) affects the outcome
        - The arrows represent causal relationships

        This structure illustrates why we need causal inference methods - the treatment effect is confounded by covariates that affect both treatment assignment and outcomes.
        """)

        # Replace output with the combined elements
        mo.output.replace(mo.vstack([header, diagram, explanation]))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    # Create a two-column explanation
    left_column = mo.md("""
    ### Good Overlap (Left Plot)

    In this scenario, treatment assignment is weakly related to feature X1:
    - Blue points (control) and red points (treatment) are well-mixed throughout
    - Units across the entire covariate space have a reasonable probability of being in either treatment or control group
    - Treatment probability is calculated using a mild logistic function: p=11+exp(-(0.5⋅X1))p = 1/(1 + exp(-(0.5 * X1)))
    - This satisfies the positivity assumption because 0<P(T=1∣X=x)<10 < P(T=1|X=x) < 1 for all values of x
    - Makes reliable causal inference possible because counterfactuals exist for all covariate values
    - Allows for unbiased estimation of treatment effects (ATE, CATE, ATT)
    """).callout(kind="success")

    right_column = mo.md("""
    ### Poor Overlap (Right Plot)

    In this scenario, treatment assignment is strongly determined by X1 and X2:
    - Clear separation between blue points (control) and red points (treatment)
    - Left region has almost exclusively control units
    - Right region has almost exclusively treatment units
    - Treatment probability uses a steep logistic function: `p = 1/(1 + exp(-(3 * X1 + 2 * X2)))`
    - Violates the positivity assumption because for some values of x, P(T=1|X=x) ≈ 0 or P(T=1|X=x) ≈ 1
    - Makes causal inference unreliable in regions without overlap
    - Requires strong modeling assumptions or extrapolation to estimate treatment effects
    - May lead to biased estimates, especially in subgroups with poor representation in one treatment condition
    """).callout(kind="danger")

    # Display the two columns side by side
    mo.hstack([left_column, right_column])
    return left_column, right_column


@app.cell(hide_code=True)
def _(mo):
    def _():
        # Create a header for the advanced causal inference methods section
        header = mo.md("### 2.6 Advanced Causal Inference Methods {#advanced-methods}")

        # Create a description of propensity score methods
        ps_methods = mo.callout(
            mo.md("""
            #### Propensity Score Methods

            Propensity score methods are statistical techniques for reducing selection bias in observational data. They work by balancing treatment groups on confounding factors to increase the validity of causal inference. The propensity score represents the probability of receiving treatment given a set of observed covariates.

            **Key applications include:**
            - **Matching**: Pairing treated and control units with similar propensity scores
            - **Stratification**: Grouping units into strata based on propensity scores
            - **Inverse Probability Weighting**: Weighting observations by the inverse of their propensity score
            - **Covariate adjustment**: Including propensity scores as covariates in regression models

            While propensity score methods can be powerful, they only adjust for observed confounders and may have limitations when the functional form is misspecified.
            """),
            kind="info"
        )

        # Create a description of difference-in-differences design
        did_methods = mo.callout(
            mo.md("""
            #### Difference-in-Differences (DiD) Design

            The Difference-in-Differences design is a quasi-experimental approach that estimates causal effects by comparing the changes in outcomes over time between a treatment group and a control group.

            **This method is particularly useful when:**
            - Random assignment to treatment is not feasible
            - Pre-treatment data is available for both groups
            - The parallel trends assumption holds (both groups would follow the same trend in the absence of treatment)

            DiD isolates the effect of a treatment by removing biases from permanent differences between groups and from shared time trends.
            """),
            kind="warn"
        )

        # Replace output with the combined elements
        mo.output.replace(mo.vstack([header, ps_methods, did_methods]))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        # Create descriptions of more advanced methods
        rdd_methods = mo.callout(
            mo.md("""
            #### Regression Discontinuity Design (RDD)

            Regression discontinuity design is an important quasi-experimental approach that can be implemented when treatment assignment is determined by whether a continuous variable crosses a specific threshold.

            RDD exploits the fact that units just above and just below the cutoff threshold are similar in all respects except for treatment assignment, creating a situation similar to random assignment around the threshold. This design estimates causal treatment effects by comparing outcomes for units near this threshold.

            **Key elements of RDD:**
            - A continuous **running variable** (or assignment variable)
            - A clear **cutoff threshold** that determines treatment
            - **Sharp RDD**: Treatment is deterministically assigned based on the threshold
            - **Fuzzy RDD**: Threshold increases probability of treatment but doesn't determine it completely
            """),
            kind="info"
        )

        # Create a description of synthetic control methods
        scm_methods = mo.callout(
            mo.md("""
            #### Synthetic Control Methods

            Synthetic control methods allow for causal inference when we have as few as one treated unit and many control units observed over time.

            **The approach:**
            - Creates a weighted combination of control units that resembles the treated unit before intervention
            - Uses this "synthetic control" to estimate what would have happened to the treated unit without treatment
            - Is especially useful for policy interventions affecting entire regions or populations

            This method has been described as "the most important development in program evaluation in the last decade" by some researchers and is particularly valuable for case studies with a small number of treated units.
            """),
            kind="warn"
        )

        # Replace output with the combined elements
        mo.output.replace(mo.vstack([rdd_methods, scm_methods]))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        # Create descriptions of more advanced methods
        sensitivity_methods = mo.callout(
            mo.md("""
            #### Sensitivity Analysis for Unobserved Confounding

            Unobserved confounding is a central barrier to drawing causal inferences from observational data. Sensitivity analysis explores how sensitive causal conclusions are to potential unobserved confounding, helping researchers understand the robustness of their findings.

            **Key approaches include:**
            - **Rosenbaum bounds**: Quantifies how strong an unobserved confounder would need to be to invalidate results
            - **E-values**: Measures the minimum strength of association an unmeasured confounder would need to have with both treatment and outcome to explain away an observed association
            - **Simulation-based methods**: Creating plausible scenarios with simulated confounders to test result stability

            While methods like propensity score matching can adjust for observed confounding, sensitivity analysis helps address the "Achilles heel" of most nonexperimental studies - the potential impact of unmeasured confounding.
            """),
            kind="danger"
        )

        # Create a description of heterogeneous treatment effects
        hte_methods = mo.callout(
            mo.md("""
            #### Heterogeneous Treatment Effects

            Treatment effects often vary across different subpopulations, a phenomenon known as treatment effect heterogeneity. Understanding this heterogeneity is crucial for targeting interventions effectively.

            **Methods for estimating heterogeneous effects include:**
            - **Subgroup analysis**: Estimating treatment effects within predefined subgroups
            - **Interaction terms**: Including treatment-covariate interactions in regression models
            - **Causal trees/forests**: Machine learning approaches that adaptively identify subgroups with different treatment effects
            - **Meta-learners**: Two-stage approaches that separate the estimation of outcome and treatment effect models

            Discovering heterogeneous effects allows for personalized interventions and can reveal important insights about treatment mechanisms that might be masked when looking only at average effects.
            """),
            kind="success"
        )

        # Replace output with the combined elements
        mo.output.replace(mo.vstack([sensitivity_methods, hte_methods]))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        # Create a header for the key terms section
        header = mo.md("### 2.7 Key Terms in Causal Inference {#key-terms}")

        # Create a tabular layout of key causal inference terms
        left_terms = mo.md("""
        | Term | Definition |
        |------|------------|
        | **Potential Outcomes Framework** | The formal mathematical framework for causal inference where each unit has potential outcomes under different treatment conditions |
        | **Average Treatment Effect (ATE)** | The expected difference between potential outcomes if the entire population received treatment versus control |
        | **Average Treatment Effect on the Treated (ATT/ATET)** | The average effect for those who actually received the treatment |
        | **Conditional Average Treatment Effect (CATE)** | Treatment effects for specific subgroups defined by covariates |
        | **Unconfoundedness/Ignorability** | The assumption that treatment assignment is independent of potential outcomes given observed covariates |
        | **Positivity/Overlap** | The assumption that every unit has a non-zero probability of receiving each treatment condition |
        | **Stable Unit Treatment Value Assumption (SUTVA)** | The assumption that one unit's treatment doesn't affect another unit's outcome |
        | **Instrumental Variables** | Variables that affect treatment assignment but not outcomes directly |
        """)

        right_terms = mo.md("""
        | Term | Definition |
        |------|------------|
        | **Mediation Analysis** | The study of how treatments affect outcomes through intermediate variables |
        | **Heterogeneous Treatment Effects** | Variation in treatment effects across different subpopulations |
        | **Doubly Robust Estimation** | Methods that remain consistent if either the outcome model or the treatment assignment model is correctly specified |
        | **Selection Bias** | Bias arising when treatment groups differ systematically in ways that affect outcomes |
        | **Confounding Bias** | Bias due to variables that affect both treatment assignment and outcomes |
        | **Common Support/Overlap Region** | The range of propensity scores where both treated and control units exist |
        | **G-methods** | A class of causal inference methods for time-varying treatments (g-formula, marginal structural models) |
        | **Causal Diagram/DAG** | Directed acyclic graphs that visually represent causal relationships between variables |
        """)

        # Add note about the importance of terminology
        note = mo.callout(
            mo.md("""
            Understanding these terms is crucial for effectively applying causal inference methods and correctly interpreting results. Many of these concepts are interrelated and build upon each other to form the foundation of causal reasoning from observational data.
            """),
            kind="info"
        )

        # Replace output with the combined elements
        mo.output.replace(mo.vstack([header, mo.hstack([left_terms, right_terms]), note]))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        section_header = mo.md("""## 3. The IHDP Dataset {#ihdp-intro}""")
        mo.output.replace(section_header)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 3.1 Overview of the IHDP Dataset

        The Infant Health and Development Program (IHDP) was conducted from 1985 to 1988 and was designed to evaluate the effect of educational and family support services along with pediatric follow-up on the development of low birth weight infants.

        The intervention consisted of:
        - Home visits by specialists
        - Child development center attendance
        - Parent group meetings

        For causal inference studies, the dataset has been modified by Jennifer Hill (2011) to create a semi-synthetic version where:
        - Some participants from the treatment group were removed to create selection bias
        - The outcomes were simulated while preserving the relationships with covariates

        This modification allows researchers to know the "ground truth" causal effects, making it an ideal benchmark dataset for causal inference methods.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout("""
    Dataset Context: The Infant Health and Development Program was a randomized controlled intervention designed to evaluate the effect of home visits by specialists on the cognitive development of premature infants.""", kind="info")
    return


@app.cell
def _(pd):
    from sklearn.model_selection import train_test_split
    import urllib.request
    import os

    # Function to download and load the IHDP dataset
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
    return ihdp_data, load_ihdp_data, os, train_test_split, urllib


@app.cell(hide_code=True)
def _(mo):
    def _():
        section_header = mo.md("""## 4. Exploratory Data Analysis {#eda}""")
        mo.output.replace(section_header)
    _()
    return


@app.cell(hide_code=True)
def _(ihdp_data, mo):
    def _():
        m1 = mo.md("### 4.1 IHDP Dataset Preview {#overview}")
        m2 = mo.ui.dataframe(ihdp_data.head())
        mo.output.replace(mo.vstack([m1,m2]))
    _()
    return


@app.cell(hide_code=True)
def _(ihdp_data, mo):
    def _():
        m1 = mo.md("### 4.2 Dataset Information")
        info_md = f"""
        - **Number of samples:** {ihdp_data.shape[0]}
        - **Number of variables:** {ihdp_data.shape[1]}
        - **Treatment assignment rate:** {ihdp_data['treatment'].mean():.2f}
        """
        m3 = mo.md(info_md)
        mo.output.replace(mo.vstack([m1,m3]))

    _()
    return


@app.cell(hide_code=True)
def _(ihdp_data, mo, pd):
    def _():
        m1 = mo.md("### 4.3 Column Types")
        m2 = mo.ui.dataframe(pd.DataFrame({
            'Column': ihdp_data.dtypes.index,
            'Data Type': ihdp_data.dtypes.values
        }))
        mo.output.replace(mo.vstack([m1,m2]))
    _()
    return


@app.cell(hide_code=True)
def _(ihdp_data, mo, plt, sns):
    def _():
        m1 = mo.md("### 4.4 Treatment Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='treatment', data=ihdp_data, ax=ax)
        ax.set_title('Distribution of Treatment Assignment')
        ax.set_xlabel('Treatment (0=Control, 1=Treated)')
        ax.set_ylabel('Count')
        # Use mo.mpl.interactive instead of just passing the figure
        interactive_fig = mo.mpl.interactive(fig)
        mo.output.replace(mo.vstack([m1, interactive_fig]))
    _()
    return


@app.cell(hide_code=True)
def _(ihdp_data, mo, plt, sns):
    # Visualize outcome distributions by treatment
    def _():
        m1 = mo.md("### 4.5 Outcome Distributions by Treatment")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='treatment', y='y_factual', data=ihdp_data, ax=ax)
        ax.set_title('Factual Outcome Distribution by Treatment Group')
        ax.set_xlabel('Treatment (0=Control, 1=Treated)')
        ax.set_ylabel('Outcome')
        # Convert to interactive plot
        interactive_fig = mo.mpl.interactive(fig)
        mo.output.replace(mo.vstack([m1, interactive_fig]))
    _()
    return


@app.cell(hide_code=True)
def _(ihdp_data, mo):
    def _():
        m1 = mo.md("### 4.6 Summary Statistics")
        m2 = mo.ui.dataframe(ihdp_data.describe())
        mo.output.replace(mo.vstack([m1, m2]))
    _()
    return


@app.cell
def _(pd):
    # Define covariate descriptions
    covariate_descriptions = {
        'x_0': "Birth weight",
        'x_1': "Birth order",
        'x_2': "Head circumference at birth",
        'x_3': "Mother's age",
        'x_4': "Mother's education level",
        'x_5': "Child is first born",
        'x_6': "Child is male",
        'x_7': "Twin",
        'x_8': "Mother's race/ethnicity",
        'x_9': "Mother smoked during pregnancy",
        'x_10': "Mother drank alcohol during pregnancy",
        'x_11': "Mother had drugs during pregnancy",
        'x_12': "Neonatal health index",
        'x_13': "Mother is married",
        'x_14': "Mother worked during pregnancy",
        'x_15': "Mother had prenatal care",
        'x_16': "Mother's weight gain during pregnancy",
        'x_17': "Family income",
        'x_18': "Preterm birth",
        'x_19': "Birth complications",
        'x_20': "Site 1",
        'x_21': "Site 2",
        'x_22': "Site 3",
        'x_23': "Site 4",
        'x_24': "Site 5"
    }

    # Create a DataFrame for UI display
    covariates_df = pd.DataFrame({
        'Variable': list(covariate_descriptions.keys()),
        'Description': list(covariate_descriptions.values())
    })
    return covariate_descriptions, covariates_df


@app.cell(hide_code=True)
def _(covariates_df, mo):
    def _():
        # Create markdown header for covariate descriptions
        m1 = mo.md("### 4.7 Covariate Descriptions {#covariates}")

        # Display covariate descriptions as an interactive dataframe
        m2 = mo.ui.dataframe(covariates_df)

        # Replace output with vertically stacked components
        mo.output.replace(mo.vstack([m1, m2]))
    _()
    return


@app.cell(hide_code=True)
def _(ihdp_data, mo):
    def _():
        # Define numerical covariates
        numerical_covs = [f'x_{i}' for i in range(5)] + ['x_12']

        # Create markdown header for numerical statistics
        m1 = mo.md("### 4.8 Numerical Covariate Statistics")

        # Display summary statistics as an interactive dataframe
        m2 = mo.ui.dataframe(ihdp_data[numerical_covs].describe())

        # Replace output with vertically stacked components
        mo.output.replace(mo.vstack([m1, m2]))
    _()
    return


@app.cell(hide_code=True)
def _(covariate_descriptions, ihdp_data, mo):
    def _():
        # Define binary covariates
        binary_covs = [f'x_{i}' for i in range(5, 25) if i != 12]

        # Calculate binary rates
        binary_rates = ihdp_data[binary_covs].mean().reset_index()
        binary_rates.columns = ['Variable', 'Rate']
        binary_rates['Description'] = binary_rates['Variable'].map(covariate_descriptions)

        # Create markdown header for binary rates
        m1 = mo.md("### 4.9 Binary Covariate Rates")

        # Display binary rates as an interactive dataframe
        m2 = mo.ui.dataframe(binary_rates)

        # Replace output with vertically stacked components
        mo.output.replace(mo.vstack([m1, m2]))
    _()
    return


@app.cell
def _(mo):
    # Create dropdown for covariate selection
    available_covs = [f'x_{i}' for i in range(5)] + ['x_12']
    covariate_selector = mo.ui.dropdown(
        options=available_covs,
        value='x_0',
        label="Select covariate to visualize"
    )
    return available_covs, covariate_selector


@app.cell(hide_code=True)
def _(covariate_descriptions, covariate_selector, ihdp_data, mo):
    def _():
        # Visualize distribution of selected covariate
        import matplotlib.pyplot as plt
        import seaborn as sns

        def plot_distribution(selected_cov):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(ihdp_data[selected_cov], ax=ax, kde=True)
            desc = covariate_descriptions.get(selected_cov, "")
            if len(desc) > 30:
                desc = desc[:30] + "..."
            ax.set_title(f"{selected_cov}: {desc}")
            ax.set_xlabel(selected_cov)
            ax.set_ylabel("Count")
            return fig

        # Create header and layout
        m1 = mo.md("### 4.10 Distribution of Key Numerical Covariates")

        # Get description for the selected covariate
        selected_desc = mo.md(f"**Selected variable**: {covariate_selector.value} - {covariate_descriptions.get(covariate_selector.value, '')}")

        # Create plot based on selected covariate
        plot_fig = plot_distribution(covariate_selector.value)
        interactive_plot = mo.mpl.interactive(plot_fig)

        # Replace output with interactive components
        mo.output.replace(mo.vstack([
            m1,
            mo.hstack([covariate_selector]),
            selected_desc,
            interactive_plot
        ]))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        m1 = mo.md("""
        1. Strong positive correlation (0.85) between x0x_0 (child's birth weight) and x1x_1 (child's birth order). This indicates that higher birth order (later-born children) tends to be associated with higher birth weight.
        2. Strong negative correlation (-0.76 and -0.7) between x2x_2 (head circumference) and both birth weight (x0x_0) and birth order (x1x_1). This is somewhat counterintuitive, as we might expect larger babies to have larger head circumferences. This inverse relationship could suggest measurement issues or certain medical conditions in the dataset.
        3. Most demographic variables (x3x_3 - mother's age, x4x_4 - mother's education) show weak correlations with other variables, suggesting independence.
        4. Child's neonatal health index (x12x_12) has mostly weak correlations with other variables, with the strongest being a mild positive correlation (0.13) with mother's age.
        """)
        m2 = mo.callout("For causal inference, these correlations are important because strongly correlated variables can create confounding issues. For example, if treatment assignment is related to birth weight, birth order might inadvertently become a confounder due to its strong correlation with birth weight.", kind="info")
        mo.output.replace(mo.vstack([m1, m2]))

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        section_header = mo.md("""## 5. Setting Up for Causal Analysis {#analysis-setup}""")
        mo.output.replace(section_header)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        subsection_header = mo.md("""### 5.1 Data Preparation {#data-prep}

        > 🔧 **Step 1**: Properly prepare the data for causal analysis

        Before implementing causal inference methods, we need to prepare our data appropriately. This includes:

        1. Splitting the data into training and test sets
        2. Scaling continuous features
        3. Identifying the types of variables (continuous vs. binary)
        4. Handling any missing values (if present)

        This preparation ensures that our causal inference methods will work properly and produce reliable estimates.
        """)
        mo.output.replace(subsection_header)
    _()
    return


@app.cell
def _(ihdp_data, mo, pd, plt, sns, train_test_split):
    # Identify continuous and binary variables
    continuous_vars = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_12']
    binary_vars = [f'x_{i}' for i in range(5, 25) if i != 12]

    # Check for missing values
    missing_values = ihdp_data.isnull().sum()
    print(f"Missing values in dataset: {missing_values.sum()}")

    # Split the data into features, treatment, and outcomes
    X = ihdp_data[[f'x_{i}' for i in range(25)]]
    T = ihdp_data['treatment']
    Y = ihdp_data['y_factual']

    # Also track true potential outcomes for evaluation (not available in real-world scenarios)
    Y0 = ihdp_data['mu_0']
    Y1 = ihdp_data['mu_1']

    # Split into training and test sets (80/20 split)
    X_train, X_test, T_train, T_test, Y_train, Y_test, Y0_train, Y0_test, Y1_train, Y1_test = train_test_split(
        X, T, Y, Y0, Y1, test_size=0.2, random_state=42
    )
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    for var in continuous_vars:
        X_train_scaled[var] = scaler.fit_transform(X_train[[var]])
        X_test_scaled[var] = scaler.transform(X_test[[var]])

    # Print information about the split
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Treatment rate in training set: {T_train.mean():.2f}")
    print(f"Treatment rate in test set: {T_test.mean():.2f}")
    print(f"True ATE in training set: {(Y1_train - Y0_train).mean():.4f}")
    print(f"True ATE in test set: {(Y1_test - Y0_test).mean():.4f}")


    # Set plotting style
    def _():
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("notebook", font_scale=1.2)    

        # Scale continuous features
        # Visualize the data split and scaling effects
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot 1: Training vs Test split by treatment
        train_counts = pd.DataFrame({
            'Set': ['Training'] * 2,
            'Treatment': ['Control', 'Treated'],
            'Count': [(T_train == 0).sum(), (T_train == 1).sum()]
        })

        test_counts = pd.DataFrame({
            'Set': ['Test'] * 2,
            'Treatment': ['Control', 'Treated'],
            'Count': [(T_test == 0).sum(), (T_test == 1).sum()]
        })

        counts_df = pd.concat([train_counts, test_counts])

        sns.barplot(x='Set', y='Count', hue='Treatment', data=counts_df, ax=axes[0])
        axes[0].set_title('Sample Distribution in Training and Test Sets')
        axes[0].set_ylabel('Number of Samples')

        # Plot 2: Effect of scaling on a continuous variable
        sns.histplot(X_train['x_0'], kde=True, label='Before scaling', ax=axes[1], alpha=0.5)
        sns.histplot(X_train_scaled['x_0'], kde=True, label='After scaling', ax=axes[1], alpha=0.5)
        axes[1].set_title('Effect of Scaling on Birth Weight (x_0)')
        axes[1].set_xlabel('Value')
        axes[1].legend()

        plt.tight_layout()

        # Replace plt.show() with mo.mpl.interactive for interactivity
        return mo.mpl.interactive(fig)

    _()
    return (
        StandardScaler,
        T,
        T_test,
        T_train,
        X,
        X_test,
        X_test_scaled,
        X_train,
        X_train_scaled,
        Y,
        Y0,
        Y0_test,
        Y0_train,
        Y1,
        Y1_test,
        Y1_train,
        Y_test,
        Y_train,
        binary_vars,
        continuous_vars,
        missing_values,
        scaler,
        var,
    )


@app.cell(hide_code=True)
def _(mo):
    def _():
        analysis_md = mo.md("""
        **Analysis of Data Preparation:**

        The dataset preparation above accomplishes several important steps for causal inference:

        1. **Splitting the data** into training (80%) and test (20%) sets allows us to evaluate the performance of our causal inference methods on unseen data, which is crucial for assessing their generalizability.

        2. **Standardizing continuous variables** ensures that variables with different scales don't unduly influence our models. This is particularly important for methods that are sensitive to the scale of the input features, such as matching based on distances or regularized regression.

        3. **Preserving the treatment assignment rate** across training and test sets maintains the same level of class imbalance, which is important for methods that are sensitive to treatment prevalence.

        4. **Verifying the absence of missing values** confirms that we don't need to implement imputation strategies, which could introduce additional complexity and potential bias.

        The visualization on the left shows the distribution of treated and control units in both training and test sets, confirming that the treatment assignment rate is similar between the splits. The visualization on the right illustrates the effect of standardization on the birth weight variable, transforming it to have zero mean and unit variance, which makes it more suitable for many statistical and machine learning methods.
        """)

        mo.callout(analysis_md, kind="info")
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        subsection_header = mo.md("""### 5.2 Formulating the Causal Question {#causal-question}

        > 📋 **Step 2**: Define what causal effect you want to estimate

        Causal inference begins with a clear formulation of the causal question. For the IHDP dataset, our primary question is:

        **"What is the effect of specialist home visits (treatment) on the cognitive test scores (outcome) of premature infants?"**

        To formalize this question, we need to define:

        1. **Treatment variable (T)**: Binary indicator for receiving home visits
        2. **Outcome variable (Y)**: Cognitive test scores
        3. **Covariates (X)**: Baseline characteristics that may influence treatment assignment or outcomes
        4. **Target population**: Premature infants with low birth weight
        5. **Causal estimand**: The specific causal quantity we want to estimate
        """)
        mo.output.replace(subsection_header)
    _()
    return


@app.cell
def _(T_train, Y0_train, Y1_train, Y_train, mo, pd, plt):
    def _():
        # Define the causal question components
        treatment_var = 'Treatment (IHDP intervention)'
        outcome_var = 'Cognitive test scores'
        covariates_var = 'Baseline characteristics (25 variables)'

        # Calculate true causal effects (available in this simulated dataset)
        true_ate = (Y1_train - Y0_train).mean()

        # Calculate naive estimate
        naive_ate = Y_train[T_train == 1].mean() - Y_train[T_train == 0].mean()

        # Calculate true ATT (Average Treatment Effect on the Treated)
        true_att = (Y1_train[T_train == 1] - Y0_train[T_train == 1]).mean()

        # Create a DataFrame for visualization
        estimands_df = pd.DataFrame({
            'Estimand': ['ATE (Average Treatment Effect)', 'ATT (Average Treatment Effect on Treated)', 'Naive Difference in Means'],
            'Value': [true_ate, true_att, naive_ate],
            'Description': [
                'Expected effect if everyone received treatment vs. none',
                'Average effect among those who actually received treatment',
                'Simple difference in mean outcomes (biased estimate)'
            ]
        })

        # Visualize causal estimands
        plt.figure(figsize=(10, 6))
        bars = plt.barh(estimands_df['Estimand'], estimands_df['Value'], color=['skyblue', 'lightgreen', 'salmon'])
        plt.title('Causal Estimands in the IHDP Dataset')
        plt.xlabel('Effect Size')

        # Add value labels to the bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                     f'{estimands_df["Value"].iloc[i]:.4f}', 
                     va='center')

        # Add grid lines for readability
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        # Create a table of key causal components
        causal_components = pd.DataFrame({
            'Component': ['Treatment Variable', 'Outcome Variable', 'Covariates', 'Target Population', 'Primary Causal Estimand'],
            'Definition': [treatment_var, outcome_var, covariates_var, 
                        'Premature infants with low birth weight', 'Average Treatment Effect (ATE)']
        })

        # Create the interactive output
        causal_question_layout = mo.vstack([
            mo.md("#### Causal Components in the IHDP Study"),
            mo.ui.table(causal_components),

            mo.md("#### Comparison of Causal Estimands"),
            mo.mpl.interactive(plt.gcf()),

            mo.md("""**Note:** In real-world causal inference, we typically don't know the true causal effects. 
            The IHDP dataset is semi-synthetic, allowing us to know the ground truth for evaluation.""")
        ])
        mo.output.append(causal_question_layout)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        analysis_md = mo.md("""
        **Analysis of the Causal Question Formulation:**

        The causal question has been clearly formulated, which is a crucial first step in any causal inference analysis. Key observations:

        1. **Treatment-Outcome Relationship**: We're interested in the effect of the IHDP intervention (home visits) on cognitive test scores, a well-defined relationship that aligns with policy and educational interventions.

        2. **Causal Estimands**: We've defined multiple estimands of interest, with the Average Treatment Effect (ATE) as our primary focus. The ATE represents the expected change in cognitive scores if the entire population received the treatment versus if none did.

        3. **ATT vs ATE**: The Average Treatment Effect on the Treated (ATT) is very close to the ATE in this dataset (difference of only 0.0041). This suggests that the treatment effect is relatively homogeneous across the population, or that selection into treatment wasn't strongly related to treatment effect heterogeneity.

        4. **Naive Estimate**: The naive difference in means is similar to the true ATE in this dataset. This is somewhat unexpected, as we would typically expect selection bias to create a difference between the naive estimate and the true causal effect. This similarity could be a characteristic of how the semi-synthetic dataset was generated.

        Formulating these specific causal questions allows us to select appropriate methods for estimation and evaluate their performance against known ground truth values in this unique dataset.
        """)

        mo.output.append(mo.callout(analysis_md, kind="info"))

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        subsection_header = mo.md("""### 5.3 Propensity Score Analysis {#propensity-analysis}

        > 🏈 **Step 3**: Analyze propensity scores to check assumptions and prepare for causal methods

        Propensity scores are a key concept in causal inference, representing the probability of receiving treatment given observed covariates. They're useful for:

        1. **Assessing overlap**: Checking the positivity assumption by examining the distribution of propensity scores
        2. **Creating balance**: Helping ensure that treated and control groups are comparable
        3. **Estimation**: Using in various estimation methods like inverse probability weighting, matching, and stratification

        Let's estimate propensity scores for our dataset and analyze their properties.
        """)
        mo.output.replace(subsection_header)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        #### What are Propensity Scores?

        Propensity scores represent the probability that a unit receives the treatment, conditional on observed covariates. Mathematically, the propensity score is defined as:

        $ e(X) = P(T=1|X) $

        Where $T$ is the treatment indicator and $X$ represents the covariates.

        #### Why Are Propensity Scores Important?

        Propensity scores help address the fundamental challenge in causal inference: units are either treated or untreated, never both. By conditioning on the propensity score, we can create balance between treated and control groups, mimicking a randomized experiment.

        Key properties of propensity scores include:

        1. **Balancing score**: Conditioning on the propensity score balances the distribution of covariates between treatment groups
        2. **Dimensionality reduction**: Reduces multiple covariates to a single score
        3. **Identification of areas of common support**: Helps identify regions where causal inference is reliable

        We'll estimate propensity scores using logistic regression and also explore a machine learning approach with random forests.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, plt, ps_df, sns):
    def _():
        # Create a figure for propensity score distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Logistic regression propensity score distributions
        sns.histplot(
            data=ps_df, x='ps_logistic', hue='treatment', 
            bins=30, element="step", common_norm=False,
            ax=axes[0], alpha=0.7
        )
        axes[0].set_title('Propensity Score Distribution (Logistic Regression)')
        axes[0].set_xlabel('Propensity Score')
        axes[0].set_ylabel('Count')

        # Plot 2: Random forest propensity score distributions
        sns.histplot(
            data=ps_df, x='ps_rf', hue='treatment', 
            bins=30, element="step", common_norm=False,
            ax=axes[1], alpha=0.7
        )
        axes[1].set_title('Propensity Score Distribution (Random Forest)')
        axes[1].set_xlabel('Propensity Score')
        axes[1].set_ylabel('Count')

        plt.tight_layout()

        # Return the interactive plot
        plot = mo.mpl.interactive(fig)
        header = mo.md("#### Propensity Score Distributions")
        mo.output.replace(mo.vstack([header, plot]))

    _()
    return


@app.cell(hide_code=True)
def _(T_train, mo, pd, propensity_scores, rf_propensity_scores):
    def _():
        # Check for overlap/positivity assumption
        # Calculate min and max propensity scores for treated and control groups
        ps_stats = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest'],
            'Min (Treated)': [propensity_scores[T_train == 1].min(), rf_propensity_scores[T_train == 1].min()],
            'Max (Treated)': [propensity_scores[T_train == 1].max(), rf_propensity_scores[T_train == 1].max()],
            'Min (Control)': [propensity_scores[T_train == 0].min(), rf_propensity_scores[T_train == 0].min()],
            'Max (Control)': [propensity_scores[T_train == 0].max(), rf_propensity_scores[T_train == 0].max()]
        })

        # Calculate common support region
        ps_stats['Common Support Min'] = ps_stats[['Min (Treated)', 'Min (Control)']].max(axis=1)
        ps_stats['Common Support Max'] = ps_stats[['Max (Treated)', 'Max (Control)']].min(axis=1)

        # Calculate percentage of units in common support
        in_support_logistic = ((propensity_scores >= ps_stats.loc[0, 'Common Support Min']) & 
                              (propensity_scores <= ps_stats.loc[0, 'Common Support Max'])).mean() * 100
        in_support_rf = ((rf_propensity_scores >= ps_stats.loc[1, 'Common Support Min']) & 
                        (rf_propensity_scores <= ps_stats.loc[1, 'Common Support Max'])).mean() * 100

        ps_stats['Units in Common Support (%)'] = [in_support_logistic, in_support_rf]

        # Count extreme propensity scores (< 0.1 or > 0.9)
        extreme_ps_logistic = ((propensity_scores < 0.1) | (propensity_scores > 0.9)).mean() * 100
        extreme_ps_rf = ((rf_propensity_scores < 0.1) | (rf_propensity_scores > 0.9)).mean() * 100

        ps_stats['Extreme PS (%)'] = [extreme_ps_logistic, extreme_ps_rf]

        # Create markdown output
        header = mo.md("#### Overlap and Common Support Analysis")

        explanation = mo.md("""
        To satisfy the positivity assumption for causal inference, we need sufficient overlap in propensity scores between treated and control groups. A good overlap indicates that units with similar characteristics have a chance of being in either treatment group.

        The **common support region** is the range of propensity scores where both treated and control units exist. Ideally, we want most units to fall within this region. Units outside this region may be problematic for causal inference.

        **Extreme propensity scores** (close to 0 or 1) indicate units that are very likely to be in one group only, which can cause issues for some causal inference methods.
        """)

        table = mo.ui.table(ps_stats.round(4))

        # Assessment of positivity assumption
        assessment = mo.callout(
            mo.md("""
            **Assessment of Positivity Assumption:**  

            - For the logistic regression model, {:.1f}% of units are within the common support region.  
            - The random forest model shows {:.1f}% of units in common support.  
            - Extreme propensity scores affect {:.1f}% (logistic) and {:.1f}% (random forest) of units.  

            The {} model provides better overlap. Overall, the positivity assumption appears to be {} satisfied, which {} for reliable causal inference using propensity score methods.
            """.format(
                in_support_logistic, 
                in_support_rf,
                extreme_ps_logistic,
                extreme_ps_rf,
                "logistic regression" if in_support_logistic > in_support_rf else "random forest",
                "reasonably well" if max(in_support_logistic, in_support_rf) > 80 else "partially",
                "is promising" if max(in_support_logistic, in_support_rf) > 80 else "raises some concerns"
            )),
            kind="info"
        )

        mo.output.replace(mo.vstack([header, explanation, table, assessment]))
    _()
    return


@app.cell(hide_code=True)
def _(
    T_train,
    X_train_scaled,
    mo,
    np,
    pd,
    plt,
    propensity_scores,
    rf_propensity_scores,
    roc_auc_score,
    roc_curve,
):
    def _():
        # Create a figure for ROC curves
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate ROC curves
        fpr_lr, tpr_lr, _ = roc_curve(T_train, propensity_scores)
        fpr_rf, tpr_rf, _ = roc_curve(T_train, rf_propensity_scores)

        # Calculate AUC scores
        auc_lr = roc_auc_score(T_train, propensity_scores)
        auc_rf = roc_auc_score(T_train, rf_propensity_scores)

        # Plot ROC curves
        ax.plot(fpr_lr, tpr_lr, lw=2, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
        ax.plot(fpr_rf, tpr_rf, lw=2, label=f'Random Forest (AUC = {auc_rf:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves for Propensity Score Models')
        ax.legend(loc='lower right')

        # Calculate standardized mean differences for covariates
        def calc_smd(var, treatment):
            """Calculate standardized mean difference for a variable"""
            treated_mean = var[treatment == 1].mean()
            control_mean = var[treatment == 0].mean()
            treated_var = var[treatment == 1].var()
            control_var = var[treatment == 0].var()
            pooled_std = np.sqrt((treated_var + control_var) / 2)
            # Handle zero standard deviation
            if pooled_std == 0:
                return 0
            return (treated_mean - control_mean) / pooled_std

        # Calculate SMD for each variable
        smd_values = []
        for col in X_train_scaled.columns:
            smd = calc_smd(X_train_scaled[col], T_train)
            smd_values.append({'Variable': col, 'SMD': smd})

        smd_df = pd.DataFrame(smd_values)

        # Sort by absolute SMD
        smd_df['Abs_SMD'] = smd_df['SMD'].abs()
        smd_df = smd_df.sort_values('Abs_SMD', ascending=False)

        # Create two-part layout
        roc_plot = mo.mpl.interactive(fig)
        roc_header = mo.md("#### Propensity Score Model Evaluation")
        roc_explanation = mo.md("""
        The ROC curves and AUC scores show how well our propensity score models discriminate between treated and control units. Higher AUC indicates better discrimination. 

        The Random Forest model typically achieves higher AUC, but this doesn't necessarily make it better for propensity score estimation. In fact, for propensity score analysis, we often prefer models that achieve good covariate balance rather than maximizing predictive performance.
        """)

        # Show balance table for top 10 most imbalanced covariates
        balance_header = mo.md("#### Covariate Balance Assessment")
        balance_explanation = mo.md("""
        The table below shows the standardized mean differences (SMD) for the most imbalanced covariates. SMD measures the difference in means between treated and control groups in standard deviation units.

        - **SMD > 0.1**: Indicates meaningful imbalance
        - **SMD > 0.25**: Indicates substantial imbalance

        Effective propensity score methods should reduce these imbalances when we condition on the propensity score.
        """)

        balance_table = mo.ui.table(smd_df.head(10).round(4))

        # Combine all elements
        mo.output.replace(mo.vstack([
            roc_header, 
            roc_plot,
            roc_explanation,
            balance_header,
            balance_explanation,
            balance_table
        ]))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Summary of Propensity Score Analysis:**

    1. We've estimated propensity scores using both logistic regression and random forest models.
    2. The distributions show some separation between treated and control groups, which is expected in observational data.
    3. The common support analysis confirms that most units fall within regions where causal inference is reliable.
    4. The covariate balance assessment identifies which variables contribute most to selection bias.

    These propensity scores will be used in subsequent sections for implementing various causal inference methods including:
    - Inverse Probability Weighting (IPW)
    - Propensity Score Matching
    - Propensity Score Stratification
    - Doubly Robust methods

    Each method leverages propensity scores differently to estimate causal effects while accounting for confounding.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        section_header = mo.md("""## 6. Implementing Causal Inference Methods {#methods}

        In this section, we'll implement and evaluate various causal inference methods on the IHDP dataset. We'll start with simple methods, then move to propensity score-based approaches, and finally explore advanced machine learning methods. For each method, we'll:

        1. Explain the methodology and key assumptions
        2. Implement the method on our training data
        3. Evaluate its performance against the known ground truth
        4. Discuss strengths, weaknesses, and practical considerations
        """)
        mo.output.replace(section_header)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        section_header = mo.md("""### 6.1 Simple Causal Inference Methods {#simple-methods}

        > 🔍 **Step 1**: Start with simple methods before moving to more complex approaches

        We'll begin with straightforward approaches that form the foundation of causal inference. These methods are easy to implement and interpret, making them excellent starting points for causal analysis.
        """)
        mo.output.replace(section_header)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    # Create a description of naive mean difference approach
    naive_description = mo.callout(
        mo.md(r"""
        #### Naive Mean Difference

        The simplest approach to estimating causal effects is to compare the average outcomes between treated and control groups:

        \[
        \hat{ATE}_{naive} = \frac{1}{n_1}\sum_{i:T_i=1}Y_i - \frac{1}{n_0}\sum_{i:T_i=0}Y_i
        \]

        where \(n_1\) is the number of treated units and \(n_0\) is the number of control units.

        **Key Assumption**: Treatment is randomly assigned (no confounding).

        **Limitations**: In observational studies, this estimate is often biased due to confounding factors that affect both treatment assignment and outcomes.
        """),
        kind="info"
    )

    # Create a description of regression adjustment approach
    regression_description = mo.callout(
        mo.md(r"""
        #### Regression Adjustment

        This method controls for confounding by including covariates in a regression model:

        \[
        Y_i = \alpha + \tau T_i + \beta X_i + \epsilon_i
        \]

        The coefficient \(\tau\) of the treatment variable \(T\) provides an estimate of the ATE.

        **Key Assumption**: The regression model is correctly specified (includes all confounders and captures their relationships with the outcome).

        **Advantages**: Simple to implement, interpretable, can handle continuous and binary covariates.

        **Limitations**: Relies on strong assumptions about the functional form of the relationship between covariates and outcomes.
        """),
        kind="warn"
    )

    # Create a description of stratification approach
    stratification_description = mo.callout(
        mo.md(r"""
        #### Stratification/Subclassification

        This method divides the data into subgroups (strata) based on important covariates, estimates treatment effects within each stratum, and takes a weighted average:

        \[
        \hat{ATE}_{strat} = \sum_{s=1}^{S} w_s (\bar{Y}_{s,1} - \bar{Y}_{s,0})
        \]

        where \(\bar{Y}_{s,1}\) is the average outcome for treated units in stratum \(s\), \(\bar{Y}_{s,0}\) is the average for control units, and \(w_s\) is the proportion of units in stratum \(s\).

        **Key Assumption**: Within each stratum, treatment is effectively randomly assigned.

        **Advantages**: Intuitive, handles non-linear relationships, allows examination of effect heterogeneity.

        **Limitations**: Can only stratify on a few variables before encountering sparsity issues.
        """),
        kind="success"
    )

    # Display all method descriptions together
    mo.vstack([naive_description, regression_description, stratification_description])
    return (
        naive_description,
        regression_description,
        stratification_description,
    )


@app.cell(hide_code=True)
def _(
    LinearRegression,
    T_train,
    X_train_scaled,
    Y0_train,
    Y1_train,
    Y_train,
    mo,
    pd,
    plt,
):
    def _():
        # 1. Naive Mean Difference
        def naive_estimator(T, Y):
            """Calculate naive mean difference between treated and control outcomes"""
            treated_mean = Y[T == 1].mean()
            control_mean = Y[T == 0].mean()
            ate = treated_mean - control_mean
            return ate

        # Calculate naive ATE
        naive_ate = naive_estimator(T_train, Y_train)

        # Get true ATE for comparison
        true_ate = (Y1_train - Y0_train).mean()    

        def regression_adjustment(X, T, Y):
            """Estimate ATE using regression adjustment"""
            # Create a copy of X to avoid modifying the original
            X_with_treatment = X.copy()
            # Add treatment as a feature
            X_with_treatment['treatment'] = T

            # Fit linear regression model
            model = LinearRegression()
            model.fit(X_with_treatment, Y)

            # Extract treatment coefficient
            treatment_idx = X_with_treatment.columns.get_loc('treatment')
            ate = model.coef_[treatment_idx]

            return ate, model

        # Calculate regression-adjusted ATE
        reg_ate, reg_model = regression_adjustment(X_train_scaled, T_train, Y_train)

        # 3. Stratification on an important covariate
        def stratification(X, T, Y, stratify_var, n_strata=5):
            """Estimate ATE by stratifying on a variable"""
            # Create a copy of the data with relevant variables
            data = pd.DataFrame({
                'T': T,
                'Y': Y,
                'strat_var': X[stratify_var]
            })

            # Create equal-sized strata based on the stratification variable
            data['stratum'] = pd.qcut(data['strat_var'], n_strata, labels=False)

            # Calculate treatment effect within each stratum
            strata_effects = []
            strata_sizes = []

            for s in range(n_strata):
                stratum_data = data[data['stratum'] == s]
                # Only calculate if we have both treated and control units
                if (stratum_data['T'] == 1).sum() > 0 and (stratum_data['T'] == 0).sum() > 0:
                    stratum_treated = stratum_data[stratum_data['T'] == 1]['Y'].mean()
                    stratum_control = stratum_data[stratum_data['T'] == 0]['Y'].mean()
                    stratum_effect = stratum_treated - stratum_control
                    stratum_size = len(stratum_data)

                    strata_effects.append(stratum_effect)
                    strata_sizes.append(stratum_size)

            # Calculate weighted average (weighted by stratum size)
            total_size = sum(strata_sizes)
            weights = [size / total_size for size in strata_sizes]
            weighted_ate = sum(effect * weight for effect, weight in zip(strata_effects, weights))

            return weighted_ate, strata_effects, weights

        # Choose birth weight (x_0) as stratification variable
        strat_ate, strat_effects, strat_weights = stratification(X_train_scaled, T_train, Y_train, 'x_0')

        # Compile results
        methods = ['Naive Mean Difference', 'Regression Adjustment', 'Stratification']
        estimates = [naive_ate, reg_ate, strat_ate]
        biases = [est - true_ate for est in estimates]
        abs_biases = [abs(bias) for bias in biases]

        # Print results
        print("Simple Methods Results:")
        print(f"True ATE: {true_ate:.4f}")
        print("-" * 50)
        for method, estimate, bias in zip(methods, estimates, biases):
            print(f"{method}: ATE = {estimate:.4f}, Bias = {bias:.4f}")

        # Return all relevant objects for use in visualization
            # Create figure for comparing estimates from simple methods
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Bar chart comparing ATE estimates
        bars = axes[0].bar(methods, estimates, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0].axhline(y=true_ate, color='red', linestyle='--', label=f'True ATE = {true_ate:.4f}')
        axes[0].set_title('ATE Estimates from Simple Methods')
        axes[0].set_ylabel('ATE Estimate')
        axes[0].legend()

        # Add value labels to the bars
        for bar in bars:
            height = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.05,
                f'{height:.4f}',
                ha='center', va='bottom', 
                rotation=0
            )

        # Plot 2: Bar chart for bias
        biases_df = pd.DataFrame({
            'Method': methods,
            'Absolute Bias': abs_biases
        }).sort_values('Absolute Bias')

        bars = axes[1].barh(biases_df['Method'], biases_df['Absolute Bias'], color=['lightgreen', 'skyblue', 'salmon'])
        axes[1].set_title('Absolute Bias of Simple Methods')
        axes[1].set_xlabel('Absolute Bias')

        # Add value labels to the bars
        for bar in bars:
            width = bar.get_width()
            axes[1].text(
                width + 0.005, 
                bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center'
            )

        plt.tight_layout()

        # Create visualization for stratification results
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Plot stratum-specific effects
        strata_indices = list(range(len(strat_effects)))
        ax2.bar(strata_indices, strat_effects, alpha=0.7)
        ax2.axhline(y=true_ate, color='red', linestyle='--', label=f'True ATE = {true_ate:.4f}')

        # Set labels and title
        ax2.set_title('Treatment Effects by Stratum')
        ax2.set_xlabel('Stratum (Birth Weight Quintile)')
        ax2.set_ylabel('Treatment Effect')
        ax2.set_xticks(strata_indices)
        ax2.set_xticklabels([f'Q{i+1}\n({w:.2f})' for i, w in enumerate(strat_weights)])
        ax2.legend()

        # Create layout containing both visualizations
        header = mo.md("#### Simple Methods Comparison")
        methods_comparison = mo.mpl.interactive(fig)

        strat_header = mo.md("#### Heterogeneous Effects by Birth Weight Stratum")
        strat_note = mo.md("*Note: Numbers in parentheses show the weight of each stratum in the overall estimate.*")
        strat_effects_plot = mo.mpl.interactive(fig2)

        methods_analysis = mo.callout(
            mo.md(f"""
            **Analysis of Simple Methods:**

            1. **Naive Mean Difference**: The naive estimate has a bias of {biases[0]:.4f}, which is {abs_biases[0]:.4f} in absolute terms. This is relatively {'small' if abs_biases[0] < 0.1 else 'moderate' if abs_biases[0] < 0.5 else 'large'}, suggesting that selection bias in this dataset may not be very strong.

            2. **Regression Adjustment**: This method has a bias of {biases[1]:.4f} (absolute: {abs_biases[1]:.4f}), {'improving upon' if abs_biases[1] < abs_biases[0] else 'performing worse than'} the naive estimator. This {'improvement' if abs_biases[1] < abs_biases[0] else 'decline'} in performance suggests that the linear model {'adequately' if abs_biases[1] < abs_biases[0] else 'inadequately'} captures the relationship between covariates and outcomes.

            3. **Stratification**: This approach has a bias of {biases[2]:.4f} (absolute: {abs_biases[2]:.4f}), {'outperforming' if abs_biases[2] < min(abs_biases[0], abs_biases[1]) else 'underperforming compared to'} the other methods. Stratification by birth weight reveals some heterogeneity in treatment effects across strata, which is valuable information for targeting interventions.

            Overall, the {'Regression Adjustment' if abs_biases[1] == min(abs_biases) else 'Naive Mean Difference' if abs_biases[0] == min(abs_biases) else 'Stratification'} method performs best in terms of bias reduction for this dataset. However, all methods show relatively small bias, suggesting that the selection mechanism in this semi-synthetic dataset may not induce strong confounding.
            """),
            kind="info"
        )

        # Combine all elements
        mo.output.replace(mo.vstack([
            header,
            methods_comparison,
            methods_analysis,
            strat_header,
            strat_effects_plot,
            strat_note
        ]))

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        section_header = mo.md("""### 6.2 Propensity Score Methods {#ps-methods}

        > 🎯 **Step 2**: Apply propensity score-based methods to adjust for confounding

        Building on the propensity scores we estimated earlier, we'll now implement methods that use these scores to create balance between treated and control groups.
        """)
        mo.output.replace(section_header)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    # Create a description of IPW approach
    ipw_description = mo.callout(
        mo.md(r"""
        #### Inverse Probability Weighting (IPW)

        IPW creates a pseudo-population where the confounding influence is eliminated by weighting each observation by the inverse of its probability of receiving the treatment it actually received:

        \[
        \hat{ATE}_{IPW} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i) Y_i}{1-e(X_i)} \right)
        \]

        where \(e(X_i)\) is the propensity score for unit \(i\).

        **Key Advantages**:
        - Uses all data points
        - Simple to implement
        - Intuitive connection to survey sampling

        **Limitations**:
        - Sensitive to extreme propensity scores (near 0 or 1)
        - Can have high variance
        - Requires well-specified propensity model
        """),
        kind="info"
    )

    # Create a description of matching approach
    matching_description = mo.callout(
        mo.md(r"""
        #### Propensity Score Matching

        Matching pairs treated units with control units that have similar propensity scores. The average difference in outcomes between matched pairs provides an estimate of the ATE:

        \[
        \hat{ATE}_{match} = \frac{1}{n_1} \sum_{i:T_i=1} (Y_i - Y_{j(i)})
        \]

        where \(j(i)\) is the index of the control unit matched to treated unit \(i\).

        **Key Advantages**:
        - Intuitive and easy to explain
        - Can be combined with exact matching on key variables
        - Preserves the original outcome variable scale

        **Limitations**:
        - Discards units that cannot be matched
        - Choice of matching algorithm and caliper can affect results
        - Matches may not be perfect, leaving some residual confounding
        """),
        kind="warn"
    )

    # Create a description of stratification approach using propensity scores
    ps_stratification_description = mo.callout(
        mo.md(r"""
        #### Propensity Score Stratification

        This method divides the data into strata based on propensity scores, estimates treatment effects within each stratum, and computes a weighted average:

        \[
        \hat{ATE}_{strat} = \sum_{s=1}^{S} w_s (\bar{Y}_{s,1} - \bar{Y}_{s,0})
        \]

        where \(w_s\) is the proportion of units in stratum \(s\), and \(\bar{Y}_{s,1}\) and \(\bar{Y}_{s,0}\) are the average outcomes for treated and control units in that stratum.

        **Key Advantages**:
        - Uses all data points
        - Examines effect heterogeneity across propensity score strata
        - Usually reduces ~90% of confounding bias with just 5 strata

        **Limitations**:
        - Less precise than matching for estimating average effects
        - Choice of strata boundaries can affect results
        - May not fully eliminate confounding within strata
        """),
        kind="success"
    )

    # Display all method descriptions together
    mo.vstack([ipw_description, matching_description, ps_stratification_description])
    return (
        ipw_description,
        matching_description,
        ps_stratification_description,
    )


@app.cell(hide_code=True)
def _(mo):
    def _():
        section_header = mo.md("""### 6.3 Advanced Machine Learning Methods {#ml-methods}

        > 🚀 **Step 3**: Leverage machine learning techniques for improved causal inference

        Finally, we'll explore advanced methods that combine machine learning with causal inference principles to estimate treatment effects more accurately.
        """)
        mo.output.replace(section_header)
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    # Create description of meta-learners
    s_learner_description = mo.callout(
        mo.md("""
        #### S-Learner (Single-Model Learner)

        The S-Learner uses a single model to estimate treatment effects by including the treatment as a feature:

        1. Train a model $\hat{μ}(x, t)$ to predict outcome using both features $X$ and treatment $T$
        2. Estimate treatment effect as $\hat{τ}(x) = \hat{μ}(x, 1) - \hat{μ}(x, 0)$

        **Key Advantages**:
        - Simple to implement
        - Can use any regression model
        - Works well when treatment effect is relatively constant

        **Limitations**:
        - May not capture treatment effect heterogeneity well
        - Can be biased when treatment assignment is highly imbalanced
        - Selection bias can influence the learned model
        """),
        kind="info"
    )

    t_learner_description = mo.callout(
        mo.md("""
        #### T-Learner (Two-Model Learner)

        The T-Learner trains separate models for treated and control groups:

        1. Train model $\hat{μ}_1(x)$ using only treated units ($T=1$)
        2. Train model $\hat{μ}_0(x)$ using only control units ($T=0$)
        3. Estimate treatment effect as $\hat{τ}(x) = \hat{μ}_1(x) - \hat{μ}_0(x)$

        **Key Advantages**:
        - Captures heterogeneity better than S-Learner
        - Can use different model types for each group
        - No implicit regularization of treatment effect

        **Limitations**:
        - May perform poorly in regions with limited data from one group
        - Requires sufficient samples in both treatment groups
        - Can be inefficient when treatment effects are simple
        """),
        kind="warn"
    )

    x_learner_description = mo.callout(
        mo.md("""
        #### X-Learner

        The X-Learner extends T-Learner by incorporating imputed treatment effects and propensity scores:

        1. Train outcome models $\hat{μ}_1(x)$ and $\hat{μ}_0(x)$ as in T-Learner
        2. Impute treatment effects for each unit:
           - For treated: $\hat{D}_i = Y_i(1) - \hat{μ}_0(X_i)$
           - For control: $\hat{D}_i = \hat{μ}_1(X_i) - Y_i(0)$
        3. Train models $\hat{τ}_1(x)$ and $\hat{τ}_0(x)$ to predict these imputed effects
        4. Combine using propensity scores: $\hat{τ}(x) = g(x)\hat{τ}_0(x) + (1-g(x))\hat{τ}_1(x)$

        **Key Advantages**:
        - Works well with strong confounding
        - Particularly effective with imbalanced treatment groups
        - Often more accurate than S- and T-Learners

        **Limitations**:
        - More complex to implement
        - Requires accurate propensity score estimation
        - Relies on accurate counterfactual predictions
        """),
        kind="success"
    )

    causal_forest_description = mo.callout(
        mo.md("""
        #### Causal Forests

        Causal Forests adapt random forests to estimate heterogeneous treatment effects:

        1. Build a forest of causal trees that split the data to maximize treatment effect heterogeneity
        2. Each tree is trained on a random subset of data and features
        3. For prediction, find similar observations through the forest structure
        4. Estimate treatment effect as the average difference in outcomes among similar observations

        **Key Advantages**:
        - Automatically discovers treatment effect heterogeneity
        - Provides valid statistical inference
        - Handles high-dimensional features well
        - Robust to the presence of many irrelevant features

        **Limitations**:
        - Computationally intensive
        - May require large sample sizes for reliable estimation
        - Less interpretable than simpler models
        """),
        kind="danger"
    )

    # Display all method descriptions together
    mo.vstack([s_learner_description, t_learner_description, x_learner_description, causal_forest_description])
    return (
        causal_forest_description,
        s_learner_description,
        t_learner_description,
        x_learner_description,
    )


if __name__ == "__main__":
    app.run()
