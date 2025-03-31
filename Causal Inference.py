import marimo

__generated_with = "0.10.16"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.image(src="https://imgs.xkcd.com/comics/correlation_2x.png").center()
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Understanding Causal Inference with IHDP: From Theory to Practice""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout("Correlation does not imply causation", kind="danger")
    mo.nav_menu({
        "/overview": "Overview"
    })
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
def _(mo):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression


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


    fig = plt.figure(figsize=(12, 5))

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
    mo.mpl.interactive(fig)
    return (
        LinearRegression,
        ax1,
        ax2,
        confounder,
        data,
        fig,
        model,
        n,
        np,
        outcome,
        pd,
        plt,
        scatter,
        sns,
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
    applications_title = mo.md("## Real-World Applications\n\nCausal inference is crucial in various domains:")
    applications_columns = mo.hstack([healthcare, policy, business])

    # Key concepts section
    concepts = mo.md(
    """
    ## Key Concepts in Causal Inference

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
        ## Types of Treatment Effects in Causal Inference

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
    header = mo.md("## Key Assumptions in Causal Inference")

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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
