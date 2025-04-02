# Advanced Data Science Causality

This repository contains interactive Marimo notebooks for Advanced Data Science Causality course.

## GitHub Pages Deployment

This project's interactive Marimo notebooks are deployed as WebAssembly-powered applications on GitHub Pages, allowing you to run and interact with the notebooks directly in your browser without any Python installation required.

### Access the Deployed Notebooks

View the deployed notebooks at: `https://[your-github-username].github.io/AdvDataScienceCasuality/`

## About Marimo

[Marimo](https://marimo.io) is a reactive notebook for Python that enables:

- Interactive data exploration with reactive execution
- Clean Python files for version control
- WebAssembly-powered deployment of notebooks as web applications

## Local Development

To run these notebooks locally:

1. Install marimo:
   ```
   pip install marimo
   ```

2. Run the notebook:
   ```
   marimo edit notebook.py
   ```

## Building for Deployment

To rebuild the WebAssembly version:

```bash
marimo export html-wasm notebook.py -o build --mode run
```

This will generate a self-contained HTML file with the necessary WebAssembly assets.
