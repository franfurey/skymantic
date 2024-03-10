# Skymantics Repository

This repository contains examples and resources related to the concepts I'm exploring during my interview at Skymantics.

## Usage Instructions for MACOS operating systems.

1. Clone this repository:
    ```
    git clone <REPO_URL>
    ```

2. Create a virtual environment (venv):
    ```
    python -m venv venv
    ```    

3. Activate the virtual environment:
    ```
    source venv/bin/activate
    ```

4. Update pip:
    ```
    pip install --upgrade pip
    ```

5. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

6. Create a Jupyter Notebook kernel to use the virtual environment's Python interpreter:
    ```
    python -m ipykernel install --user --name skymantics --display-name "Skymantics"
    ```

## Concepts to Explore

### KDE (Kernel Density Estimation)

**What is it?** KDE is a non-parametric technique for estimating the probability density function of a dataset.

**Resources:**
- [Scikit-learn documentation on KDE](https://scikit-learn.org/stable/modules/density.html)

### F Regression

**What is it?** F Regression refers to the use of F-statistics in hypothesis testing to select significant variables during regression modeling.

**Resources:**
- [Scikit-learn documentation on feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)

### Mutual Information Regression

**What is it?** Mutual Information Regression measures the mutual dependence between two variables, useful for feature selection.

**Resources:**
- [Scikit-learn guide on feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#mutual-info)

### K Means

**What is it?** K Means is a clustering algorithm that divides a dataset into k groups based on the similarity of data features.

**Resources:**
- [Scikit-learn guide on K Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)

### Markov Models

**What is it?** Markov models are stochastic models useful for modeling sequences of events where the conditional independence property is maintained.
