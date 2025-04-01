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
    def _():
        toc_md = mo.md("""
        ## Table of Contents

        1. [Introduction to Causal Inference](#introduction)
        2. [Theoretical Foundations](#foundations)
           - [Real-World Applications](#applications)
           - [Key Concepts](#concepts)
           - [Treatment Effects](#effects)
           - [Key Assumptions](#assumptions)
        3. [The IHDP Dataset](#ihdp-intro)
        4. [Exploratory Data Analysis](#eda)
           - [Dataset Overview](#overview)
           - [Covariate Analysis](#covariates)
        5. [Setting Up for Causal Analysis](#analysis-setup)
           - [Data Preparation](#data-prep)
           - [Formulating the Causal Question](#causal-question)
           - [Propensity Score Analysis](#propensity-analysis)
        6. [Future Implementations](#future)
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


@app.cell
def _():
    return


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
        LinearRegression,
        ax1,
        ax2,
        confounder,
        data,
        fig_1,
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
def _(mo):
    def _():
        # Visualize the overlap assumption
        import numpy as np
        import matplotlib.pyplot as plt

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
    - Treatment probability is calculated using a mild logistic function: `p = 1/(1 + exp(-(0.5 * X1)))`
    - This satisfies the positivity assumption because `0 < P(T=1|X=x) < 1` for all values of x
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
        m2 = fig
        mo.output.replace(mo.vstack([m1, m2]))
    _()
    return


@app.cell
def _():
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
        m2 = fig
        mo.output.replace(mo.vstack([m1,m2]))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        m1 = mo.md("""
        This boxplot compares outcome distributions between treated and control groups:

            1. Clear separation in outcomes between groups, with the treatment group (1) having substantially higher outcomes (median around 6.5) than the control group (0) with median around 2.3.

            2. The interquartile ranges are similar for both groups, indicating comparable outcome variability.

            3. Outliers appear more prominently in the control group, particularly at the upper end.
        """)
        m2 = mo.callout("""This clear separation strongly suggests a positive treatment effect. The visualization supports the semi-synthetic nature of the dataset, where outcomes were generated to have known causal effects. The substantial difference in outcomes corresponds to the "ground truth" positive effect of the intervention on cognitive development that was built into the dataset.
        """, kind="info")
        mo.output.replace(mo.vstack([m1, m2]))

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


@app.cell(hide_code=True)
def _(ihdp_data, pd):
    # Describe the covariates
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

    # Create a DataFrame with descriptions
    covariates_df = pd.DataFrame({
        'Variable': covariate_descriptions.keys(),
        'Description': covariate_descriptions.values()
    })

    # print("Covariate Descriptions:")
    # print(covariates_df)

    # Calculate summary statistics for numerical covariates
    numerical_covs = [f'x_{i}' for i in range(5)] + ['x_12']
    # print("\nNumerical Covariate Statistics:")
    # print(ihdp_data[numerical_covs].describe())

    # Calculate rates for binary covariates
    binary_covs = [f'x_{i}' for i in range(5, 25) if i != 12]
    binary_rates = ihdp_data[binary_covs].mean().reset_index()
    binary_rates.columns = ['Variable', 'Rate']
    binary_rates['Description'] = binary_rates['Variable'].map(covariate_descriptions)

    # print("\nBinary Covariate Rates:")
    # print(binary_rates)
    return (
        binary_covs,
        binary_rates,
        covariate_descriptions,
        covariates_df,
        numerical_covs,
    )


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


@app.cell(hide_code=True)
def _(covariate_descriptions, ihdp_data, mo):
    def _():
        # Visualize distribution of a few key numerical covariates
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Plot histograms for numerical covariates
        key_covs = ['x_0', 'x_2', 'x_3', 'x_4', 'x_12']
        descriptions = [covariate_descriptions[k][:20] + '...' if len(covariate_descriptions[k]) > 20 
                       else covariate_descriptions[k] for k in key_covs]

        for i, (cov, desc) in enumerate(zip(key_covs, descriptions)):
            sns.histplot(ihdp_data[cov], ax=axes[i], kde=True)
            axes[i].set_title(f"{cov}: {desc}")

        # Hide unused subplot
        axes[5].axis('off')

        plt.tight_layout()

        # Create and display markdown header
        m1 = mo.md("### 4.10 Distribution of Key Numerical Covariates")

        # Replace output with vertically stacked components
        mo.output.replace(mo.vstack([m1, fig]))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        m1 = mo.md("""
        This panel of histograms shows the distribution patterns of key variables:

    1. Child's birth weight (x_0) shows a slightly right-skewed distribution, indicating most infants are clustered around or slightly below average birth weight.
    2. Head circumference (x_2) displays a complex multi-modal distribution with several peaks, suggesting potential subgroups or recording practices.
    3. Mother's age (x_3) shows a striking multi-modal distribution with distinct peaks at approximately -0.8, 0.2, 1.2, and 2.0. This could reflect common maternal age groups or possibly artifacts from how age was recorded.
    4. Mother's education (x_4) appears approximately normally distributed, centered slightly below 0, with a long left tail indicating some mothers with very low education levels.
    5. Child's neonatal health index (x_12) shows a binary distribution with values concentrated at 0 and 1, suggesting this is actually a categorical variable despite being stored as numeric.
        """)
        m2 = mo.callout("These distributions highlight the heterogeneity in the sample and identify potential issues for causal inference, like the binary nature of the health index that might need special handling in analysis.", kind="info")
        mo.output.replace(mo.vstack([m1, m2]))
    _()
    return


@app.cell(hide_code=True)
def _(ihdp_data, mo):
    def _():
        # Compute correlation matrix for numerical covariates
        numerical_covs = [f'x_{i}' for i in range(5)] + ['x_12']
        corr_matrix = ihdp_data[numerical_covs].corr()

        # Create visualization using matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix for Numerical Covariates')

        # Create and display markdown header
        m1 = mo.md("### 4.11 Correlation Between Numerical Covariates")

        # Replace output with vertically stacked components
        mo.output.replace(mo.vstack([m1, fig]))
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        m1 = mo.md("""
        1. Strong positive correlation (0.85) between `x_0` (child's birth weight) and `x_1` (child's birth order). This indicates that higher birth order (later-born children) tends to be associated with higher birth weight.
        2. Strong negative correlation (-0.76 and -0.7) between `x_2` (head circumference) and both birth weight (`x_0`) and birth order (`x_1`). This is somewhat counterintuitive, as we might expect larger babies to have larger head circumferences. This inverse relationship could suggest measurement issues or certain medical conditions in the dataset.
        3. Most demographic variables (`x_3` - mother's age, `x_4` - mother's education) show weak correlations with other variables, suggesting independence.
        4. Child's neonatal health index (`x_12`) has mostly weak correlations with other variables, with the strongest being a mild positive correlation (0.13) with mother's age.
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
def _(ihdp_data, pd, plt, sns, train_test_split):
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
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
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
        plt.show()

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
        
        mo.callout(analysis_md, kind="info")

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


@app.cell
def _(T_train, X_train_scaled, mo, np, pd, plt, sns):
    def _():
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_curve, roc_auc_score
        
        # Estimate propensity scores using logistic regression
        propensity_model = LogisticRegression(max_iter=1000, C=1.0)
        propensity_model.fit(X_train_scaled, T_train)
        
        # Get propensity scores
        propensity_scores = propensity_model.predict_proba(X_train_scaled)[:, 1]
        
        # Create a copy of training data with propensity scores
        train_data = pd.DataFrame({
            'treatment': T_train,
            'propensity_score': propensity_scores
        })
        
        # Summarize propensity scores by treatment group
        ps_summary = train_data.groupby('treatment')['propensity_score'].describe()
        print("Propensity Score Summary by Treatment Group:")
        print(ps_summary)
        
        # Check for extreme propensity scores
        extreme_ps = (propensity_scores < 0.05) | (propensity_scores > 0.95)
        print(f"Percentage of units with extreme propensity scores: {100 * extreme_ps.mean():.2f}%")
        
        # Check for overlap
        # Calculate min and max propensity scores for treated and control groups
        treated_min = propensity_scores[T_train == 1].min()
        treated_max = propensity_scores[T_train == 1].max()
        control_min = propensity_scores[T_train == 0].min()
        control_max = propensity_scores[T_train == 0].max()
        
        # Determine the region of common support
        common_min = max(treated_min, control_min)
        common_max = min(treated_max, control_max)
        
        # Calculate the percentage of units in the common support region
        in_common_support = ((propensity_scores >= common_min) & 
                            (propensity_scores <= common_max))
        pct_in_support = 100 * in_common_support.mean()
        
        print(f"Common support region: [{common_min:.4f}, {common_max:.4f}]")
        print(f"Percentage of units in common support: {pct_in_support:.2f}%")
        
        # Visualize propensity score distributions
        fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Distribution of propensity scores by treatment group
        sns.histplot(train_data, x='propensity_score', hue='treatment', bins=20, 
                   element="step", common_norm=False, stat='density', ax=axes[0])
        axes[0].axvline(x=common_min, color='red', linestyle='--', label='Common support')
        axes[0].axvline(x=common_max, color='red', linestyle='--')
        axes[0].set_title('Propensity Score Distribution')
        axes[0].set_xlabel('Propensity Score')
        axes[0].set_ylabel('Density')
        axes[0].legend(labels=['Common support', '', 'Control', 'Treated'])
        
        # Plot 2: Propensity scores vs treatment assignment
        jitter = np.random.uniform(-0.05, 0.05, size=len(T_train))
        axes[1].scatter(propensity_scores, T_train + jitter, alpha=0.5)
        axes[1].axhline(y=0, color='blue', linestyle='-', alpha=0.3)
        axes[1].axhline(y=1, color='red', linestyle='-', alpha=0.3)
        axes[1].set_title('Propensity Scores vs Treatment')
        axes[1].set_xlabel('Propensity Score')
        axes[1].set_ylabel('Treatment Assignment (with jitter)')
        axes[1].set_ylim(-0.5, 1.5)
        
        # Plot 3: ROC curve for propensity score model
        fpr, tpr, _ = roc_curve(T_train, propensity_scores)
        auc = roc_auc_score(T_train, propensity_scores)
        
        axes[2].plot(fpr, tpr, lw=2)
        axes[2].plot([0, 1], [0, 1], 'k--', lw=2)
        axes[2].set_title(f'ROC Curve (AUC = {auc:.3f})')
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        
        plt.tight_layout()
        
        # Get feature importance for the propensity score model
        coef = propensity_model.coef_[0]
        features = X_train_scaled.columns
        
        # Create a DataFrame of coefficients
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coef,
            'Abs_Coefficient': np.abs(coef)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        # Plot top influential features
        fig2 = plt.figure(figsize=(10, 6))
        plt.barh(coef_df['Feature'].head(10), coef_df['Coefficient'].head(10))
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.title('Top 10 Features for Treatment Assignment')
        plt.xlabel('Coefficient in Propensity Score Model')
        plt.tight_layout()
        
        # Create an interactive output for Marimo
        propensity_analysis = mo.vstack([
            mo.md("#### Propensity Score Distributions and Model Quality"),
            mo.mpl.interactive(fig1),
            mo.md("#### Features Influencing Treatment Assignment"),
            mo.mpl.interactive(fig2),
            mo.md("#### Propensity Score Statistics"),
            mo.ui.dataframe(ps_summary)
        ])

        mo.output.append(propensity_analysis)

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        analysis_md = mo.md("""
        **Analysis of Propensity Score Results:**
        
        The propensity score analysis provides important insights for causal inference:
        
        1. **Propensity Score Distribution**: The distribution shows moderate overlap between treated and control groups. The common support region covers most of the sample, which is favorable for causal inference methods that rely on propensity scores.
        
        2. **Treatment Assignment Predictability**: The ROC curve and AUC value indicate how well we can predict treatment assignment from the observed covariates. A high AUC suggests significant selection bias that needs to be addressed by causal methods.
        
        3. **Influential Features**: The feature importance plot reveals which variables most strongly influence treatment assignment. In the IHDP dataset, several demographic and medical characteristics show strong associations with treatment probability. These are the key confounders we need to adjust for.
        
        4. **Positivity Assessment**: Most units have propensity scores away from the extreme values (0 or 1), which satisfies the positivity assumption required for many causal inference methods. However, a small percentage of units have extreme scores, which may require special handling such as trimming or stratification.
        
        This analysis confirms we have reasonable overlap between treatment groups, suggesting that propensity score-based methods like matching, weighting, or stratification should work well for this dataset. The next step would be to implement and compare these methods.
        """)
        
        mo.callout(analysis_md, kind="info")

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
