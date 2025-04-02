# Marimo Syntax and Coding Patterns

This guide covers the syntax, coding patterns, and best practices for writing effective marimo notebooks.

## Basic Structure of a Marimo Notebook

A marimo notebook is composed of Python cells that are executed reactively based on dependencies. Here's a basic structure:

```python
# Import marimo - typically done in the first cell
import marimo as mo

# Define variables, functions, or load data
data = [1, 2, 3, 4, 5]

# Create interactive elements
slider = mo.ui.slider(1, 10, value=5)

# Use variables and interactive elements in calculations
result = sum(data) * slider.value

# Display output with markdown
mo.md(f"The result is: **{result}**")
```

Unlike Jupyter, marimo doesn't require special syntax or magic commands. It's pure Python, with the marimo library providing additional functionality.

## Importing Marimo

Always start your notebook by importing the marimo library, typically aliased as `mo`:

```python
import marimo as mo
```

This import provides access to all marimo functionality, including UI elements, markdown, layout components, and more.

## Reactive Execution: Understanding Cell Dependencies

Marimo's core feature is reactive execution. Cells are executed based on the variables they define and reference, not their order in the notebook.

As described in the marimo documentation: "The order of cells on the page has no bearing on the order cells are executed in: execution order is determined by the variables cells define and the variables they read" ([docs.marimo.io](https://docs.marimo.io/getting_started/key_concepts/)).

How it works:
1. Marimo statically analyzes each cell to determine dependencies
2. When a cell is run, all cells that reference its variables are automatically run
3. When a cell is deleted, its variables are removed from memory

## Creating and Using UI Elements

Marimo provides a rich set of UI elements via the `mo.ui` module:

```python
# Create a slider with min=1, max=10, default=5
slider = mo.ui.slider(1, 10, value=5)

# Create a dropdown menu
dropdown = mo.ui.dropdown(
    options=["Option 1", "Option 2", "Option 3"],
    value="Option 1",
    label="Select an option"
)

# Create a text input field
text_input = mo.ui.text(placeholder="Enter text here")

# Create a button
button = mo.ui.button(label="Click me")

# Create a checkbox
checkbox = mo.ui.checkbox(label="Check me", value=False)
```

To use these elements, simply reference their `value` attribute:

```python
# Display the selected value
mo.md(f"You selected: {slider.value}")

# Use in calculations
result = slider.value * 2
```

When a UI element is interacted with, all cells referencing it are automatically re-run.

## Working with Markdown

Marimo provides rich markdown capabilities with the `mo.md` function:

```python
# Simple markdown
mo.md("# Title\nThis is a paragraph with **bold** and *italic* text.")

# Markdown with interpolated Python values
name = "World"
mo.md(f"# Hello, {name}!\nThe current value is {slider.value}.")

# Embed UI elements directly in markdown
mo.md(f"""
# Interactive Example
Adjust the slider: {slider}

Selected option: {dropdown.value}
""")
```

For other objects like plots, wrap them in `mo.as_html()`:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

mo.md(f"""
# Plot Example
Here's a plot:

{mo.as_html(fig)}
""")
```

## Layout Components

Organize your outputs with layout components:

```python
# Horizontal stack
mo.hstack([
    mo.md("Left content"),
    mo.md("Right content")
])

# Vertical stack
mo.vstack([
    mo.md("Top content"),
    mo.md("Bottom content")
])

# Accordion
mo.accordion({
    "Section 1": mo.md("Content for section 1"),
    "Section 2": mo.md("Content for section 2")
})

# Tabs
mo.ui.tabs({
    "Tab 1": mo.md("Content for tab 1"),
    "Tab 2": mo.md("Content for tab 2")
})

# Sidebar
mo.sidebar([
    mo.md("# Controls"),
    slider,
    dropdown
])
```

## Best Practices

The marimo documentation provides several best practices for writing effective notebooks ([docs.marimo.io/guides/best_practices/](https://docs.marimo.io/guides/best_practices/)):

### 1. Use Global Variables Sparingly

Keep the number of global variables minimal to avoid name collisions:

```python
# Bad practice: Many global variables
x = 1
y = 2
result = x + y

# Better practice: Encapsulate in functions
def calculate_sum(a, b):
    return a + b

result = calculate_sum(1, 2)
```

### 2. Use Descriptive Variable Names

```python
# Not descriptive
s = mo.ui.slider(1, 10)

# More descriptive
temperature_celsius = mo.ui.slider(0, 100, value=25, label="Temperature (°C)")
```

### 3. Avoid Splitting Declarations and Mutations

As noted in the marimo documentation, don't split variable declarations and mutations across cells:

```python
# Don't do this in separate cells:
data = [1, 2, 3]
# (in another cell)
data.append(4)  # Mutation in a different cell

# Instead, do this in a single cell:
data = [1, 2, 3]
data.append(4)

# Or create a new variable:
data = [1, 2, 3]
# (in another cell)
expanded_data = data + [4]  # Creating new variable instead of mutating
```

### 4. Write Idempotent Cells

Cells should produce the same output when given the same inputs:

```python
# Not idempotent (depends on external state)
counter += 1
mo.md(f"Counter: {counter}")

# Idempotent (depends only on its inputs)
def increment(value):
    return value + 1

new_counter = increment(counter)
mo.md(f"Counter: {new_counter}")
```

### 5. Use Functions for Code Organization

Encapsulate logic in functions to reduce global namespace pollution:

```python
def process_data(data, parameter):
    # Complex processing logic
    result = data * parameter
    return result

# Use the function
output = process_data(data, slider.value)
```

## Example Data Analysis Workflow

Here's a typical pattern for data analysis in marimo:

```python
# Cell 1: Import libraries
import marimo as mo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Load data
def load_data():
    return pd.read_csv("data.csv")

data = load_data()

# Cell 3: Create interactive filters
date_range = mo.ui.date_range(
    value=(data['date'].min(), data['date'].max()),
    label="Date range"
)

category_filter = mo.ui.multiselect(
    options=data['category'].unique().tolist(),
    value=data['category'].unique().tolist(),
    label="Categories"
)

# Cell 4: Filter data based on user selections
def filter_data(df, dates, categories):
    start_date, end_date = dates
    return df[
        (df['date'] >= start_date) &
        (df['date'] <= end_date) &
        (df['category'].isin(categories))
    ]

filtered_data = filter_data(data, date_range.value, category_filter.value)

# Cell 5: Create visualizations
def create_plot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='category', y='value', ax=ax)
    return fig

plot = create_plot(filtered_data)

# Cell 6: Display results
mo.vstack([
    mo.md(f"### Filtered Data ({len(filtered_data)} rows)"),
    mo.as_html(plot),
    mo.md("#### Summary Statistics"),
    filtered_data.describe()
])
```

## Conclusion

Marimo's syntax and coding patterns leverage standard Python while adding powerful reactive capabilities. By following the best practices outlined in this guide, you can create effective, reproducible, and interactive notebooks that take full advantage of marimo's unique features.

Remember:
- Use pure Python with the marimo library
- Keep global variables minimal and descriptive
- Organize code with functions
- Understand how reactive execution works
- Leverage UI elements and markdown for interactivity
- Follow best practices for mutations and state management

---

*References:*
- [Marimo Documentation: Key Concepts](https://docs.marimo.io/getting_started/key_concepts/)
- [Marimo Documentation: Best Practices](https://docs.marimo.io/guides/best_practices/)
- [Marimo Documentation: Interactive Elements](https://docs.marimo.io/guides/interactivity.html)
- [GitHub: marimo-team/marimo](https://github.com/marimo-team/marimo)
