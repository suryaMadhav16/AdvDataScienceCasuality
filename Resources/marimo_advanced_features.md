# Marimo Advanced Features

This document explores the advanced features of marimo notebooks, including interactive visualizations, SQL integration, performance optimization, and deployment options.

## Interactive Visualization Capabilities

Marimo provides powerful interactive visualization capabilities, extending beyond standard plotting libraries with reactive features.

### Reactive Plots with Altair

Marimo integrates with Altair to create interactive, selectable plots where frontend selections are automatically available as pandas DataFrames in Python:

```python
import marimo as mo
import altair as alt
import pandas as pd

# Sample data
data = pd.DataFrame({
    'x': range(100),
    'y': [i**2 for i in range(100)],
    'category': ['A' if i < 50 else 'B' for i in range(100)]
})

# Create an Altair chart
chart = alt.Chart(data).mark_point().encode(
    x='x',
    y='y',
    color='category'
)

# Make it reactive with mo.ui.altair_chart
reactive_chart = mo.ui.altair_chart(chart)

# Display the chart and the selected data
mo.vstack([
    mo.md("## Interactive Scatter Plot"),
    reactive_chart,
    mo.md("### Selected Points:"),
    reactive_chart.value  # This contains the selected data as a DataFrame
])
```

As explained in the marimo documentation: "Use `mo.ui.altair_chart` to easily create interactive, selectable plots: selections you make on the frontend are automatically made available as Pandas dataframes in Python."

### Reactive Plots with Plotly

Similar functionality is available for Plotly:

```python
import marimo as mo
import plotly.express as px
import pandas as pd

# Sample data
data = px.data.iris()

# Create a Plotly scatter plot
fig = px.scatter(
    data, 
    x="sepal_width", 
    y="sepal_length",
    color="species"
)

# Make it reactive
reactive_plot = mo.ui.plotly(fig)

# Display the plot and selections
mo.vstack([
    mo.md("## Iris Dataset Explorer"),
    reactive_plot,
    mo.md("### Selected Points:"),
    reactive_plot.value  # Contains selected points
])
```

Note that marimo can render any Plotly plot, but `mo.ui.plotly` only supports reactive selections for scatter plots, treemaps charts, and sunbursts charts.

## SQL Integration

Marimo has built-in support for SQL, enabling seamless querying of databases and DataFrames directly within notebooks.

### Querying DataFrames with SQL

```python
import marimo as mo
import pandas as pd

# Sample data
data = pd.DataFrame({
    'id': range(1, 101),
    'name': [f'Item {i}' for i in range(1, 101)],
    'category': ['A' if i < 30 else 'B' if i < 70 else 'C' for i in range(1, 101)],
    'value': [i * 10 for i in range(1, 101)]
})

# Create a SQL cell in marimo to query this DataFrame
# In the UI, you can create a SQL cell by clicking the SQL button
# Here's what the SQL query would look like:
"""
SELECT 
    category,
    COUNT(*) as count,
    AVG(value) as avg_value,
    MAX(value) as max_value
FROM data
GROUP BY category
ORDER BY count DESC
"""

# The result would be returned as a DataFrame
```

### Connecting to Databases

Marimo supports various database connections, including MotherDuck (cloud-based DuckDB):

```python
import marimo as mo
import duckdb

# Connect to MotherDuck
duckdb.sql("ATTACH IF NOT EXISTS 'md:my_db'")

# Query a table (this would be done in a SQL cell)
"""
SELECT * FROM my_table
LIMIT 10
"""

# Process results with Python
# result_df contains the query results as a DataFrame
processed_data = result_df['value'].mean()
mo.md(f"Average value: {processed_data}")
```

As described in the marimo documentation: "marimo's reactive execution model extends into SQL queries, so changes to your SQL will automatically trigger downstream computations for dependent cells."

## Performance Optimization for Large Datasets

Marimo provides several tools for optimizing performance when working with expensive computations or large datasets.

### Conditional Execution with mo.stop

Control execution of expensive operations with `mo.stop`:

```python
import marimo as mo

# Create a button to trigger computation
run_button = mo.ui.button(label="Run expensive calculation")

# Use mo.stop to prevent execution until button is clicked
mo.stop(not run_button.value)

# This code only runs when button is clicked
result = expensive_calculation()
mo.md(f"Result: {result}")
```

### In-Memory Caching

Cache function results to avoid redundant calculations:

```python
import marimo as mo
import time

@mo.cache
def expensive_function(param1, param2):
    """This function's results are cached based on its arguments."""
    time.sleep(2)  # Simulating expensive computation
    return param1 * param2

# First call will compute and cache
result1 = expensive_function(10, 20)

# Second call with same args will use cached result
result2 = expensive_function(10, 20)  # Returns instantly

# Different args will compute and cache separately
result3 = expensive_function(5, 5)  # Will compute again
```

### Disk Caching with Persistence

For very expensive calculations that you want to persist across notebook restarts:

```python
import marimo as mo
import time

@mo.persistent_cache(name="my_expensive_calculation")
def expensive_calculation(data, params):
    """Results cached to disk for persistence across notebook restarts."""
    time.sleep(5)  # Simulating very expensive computation
    return data * params

# First run: will compute and save to disk
result = expensive_calculation(large_dataset, 0.5)

# On notebook restart: will load from disk instead of recomputing
```

### Lazy Loading UI Elements

Optimize performance by lazy loading UI elements that are expensive to compute:

```python
import marimo as mo
import pandas as pd

# Large dataset
data = pd.read_csv("large_dataset.csv")

# Lazy load the table - only renders when visible
mo.lazy(mo.ui.table(data))

# For expensive calculations, use a function
def expensive_component():
    # Only calculated when component becomes visible
    processed_data = expensive_processing(data)
    return mo.ui.table(processed_data)

# Use in an accordion
accordion = mo.accordion({
    "Raw Data": mo.lazy(mo.ui.table(data)),
    "Processed Data": mo.lazy(expensive_component)
})
```

## Deployment Options

Marimo offers versatile deployment options for sharing your notebooks.

### Running as Web Apps

Every marimo notebook can be run as a web app with Python code hidden:

```bash
marimo run my_notebook.py
```

Additional options include:
- `--include-code` to include code in the app
- `--base-url` to deploy at a specific URL path

### WebAssembly and Static HTML

Export notebooks as WebAssembly-powered HTML:

```bash
marimo export my_notebook.py
```

This creates a standalone HTML file that can be hosted on static sites like GitHub Pages without requiring a Python backend.

### Programmatic Deployment

For more complex applications, marimo notebooks can be integrated into ASGI applications:

```python
from fastapi import FastAPI
from marimo.server import asyncio as marimo_asyncio

app = FastAPI()

# Mount a marimo app at /my-app
app.mount(
    "/my-app",
    marimo_asyncio.AppServer("path/to/my_notebook.py", include_code=False)
)
```

### Authentication and Security

For secure deployments, marimo provides authentication options:

```bash
marimo run my_notebook.py --auth=basic
```

This requires username/password authentication to access the app.

## Integration with External Tools

### Module Autoreloading

Marimo supports automatic reloading of external Python modules, making it easier to organize large projects:

```python
# In marimo notebook
import marimo as mo

# Enable module autoreloading
mo.autoreload(True)

# Import your module
import my_custom_module

# Now when you change my_custom_module.py, changes are automatically available
result = my_custom_module.calculate()
```

### Package Management

Marimo can track and manage package dependencies directly in the notebook:

```python
# Add a package to the notebook's requirements
mo.require("pandas==2.0.0")

# Or use sandbox mode when opening a notebook
# marimo edit my_notebook.py --sandbox
```

## Advanced Layout Techniques

Create sophisticated layouts by nesting components:

```python
import marimo as mo

# Create a complex layout with sidebar, tabs, and nested components
page = mo.hstack([
    # Sidebar
    mo.sidebar([
        mo.md("## Controls"),
        mo.ui.slider(1, 10, value=5, label="Parameter 1"),
        mo.ui.dropdown(options=["A", "B", "C"], label="Category")
    ]),
    
    # Main content area with tabs
    mo.vstack([
        mo.md("# Dashboard"),
        mo.ui.tabs({
            "Overview": mo.vstack([
                mo.md("## Summary Statistics"),
                mo.ui.dataframe(summary_df)
            ]),
            "Details": mo.vstack([
                mo.md("## Detailed Analysis"),
                # Nested tabs for different visualizations
                mo.ui.tabs({
                    "Chart 1": plot1,
                    "Chart 2": plot2,
                    "Chart 3": plot3
                })
            ]),
            "Data": mo.ui.dataframe(full_data)
        })
    ])
])
```

## Conclusion

Marimo's advanced features provide a powerful toolkit for creating sophisticated, interactive, and performant notebooks. By leveraging reactive visualizations, SQL integration, performance optimization, and flexible deployment options, you can build complex data applications that go far beyond traditional notebook capabilities.

These features make marimo particularly well-suited for:
- Building interactive dashboards and data applications
- Working with large datasets
- Creating reproducible research workflows
- Sharing analysis with non-technical stakeholders
- Deploying notebooks as standalone applications

As the marimo ecosystem continues to evolve, we can expect even more powerful features and integrations to emerge, further enhancing its capabilities for data science, research, and education.

---

*References:*
- [Marimo Documentation: Plotting](https://docs.marimo.io/guides/working_with_data/plotting/)
- [Marimo Documentation: SQL](https://docs.marimo.io/guides/working_with_data/sql/)
- [Marimo Documentation: Expensive Notebooks](https://docs.marimo.io/guides/expensive_notebooks/)
- [Marimo Documentation: Deployment](https://docs.marimo.io/guides/deploying/)
- [Marimo Documentation: MotherDuck Integration](https://docs.marimo.io/integrations/motherduck/)
