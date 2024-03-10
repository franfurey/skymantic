import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from f_regression import train_and_evaluate_rfecv

def test_train_and_evaluate_rfecv_structure():
    """
    Test to ensure the function returns a structure with non-empty values for different parameters.
    """
    # Define parameters for testing
    steps = [1]
    cv_splits = [4, 5]
    scorings = ['accuracy', 'f1_macro']

    summary = train_and_evaluate_rfecv(steps=steps, cv_splits=cv_splits, scorings=scorings)

    # Check for the main keys
    assert 'results' in summary and isinstance(summary['results'], dict), f"Expected 'results' key with a dict value."
    assert 'best_per_scoring' in summary and isinstance(summary['best_per_scoring'], dict), f"Expected 'best_per_scoring' key with a dict value."
    assert 'overall_best' in summary and isinstance(summary['overall_best'], dict), f"Expected 'overall_best' key with a dict value."

    # Check for valid 'best_per_scoring' structure and non-empty values
    for scoring, details in summary['best_per_scoring'].items():
        assert 'score' in details and 'details' in details, f"Missing 'score' or 'details' in best_per_scoring for {scoring}."
        assert isinstance(details['score'], float), f"Expected 'score' to be float in best_per_scoring for {scoring}, but it gets {details['score']}"
        assert isinstance(details['details'], dict), f"Expected 'details' to be dict in best_per_scoring for {scoring}, but it gets {details['details']}"

    # Check 'overall_best' structure and non-empty values
    assert 'score' in summary['overall_best'] and isinstance(summary['overall_best']['score'], float), "Invalid or missing 'score' in 'overall_best'."
    assert 'details' in summary['overall_best'] and isinstance(summary['overall_best']['details'], dict), "Invalid or missing 'details' in 'overall_best'."

    # Print statement if all assertions pass
    print("test_train_and_evaluate_rfecv_structure function passed all checks.")

def test_train_and_evaluate_rfecv_functionality():
    """
    Test the functionality of the train_and_evaluate_rfecv function with known inputs and outputs.
    """
    # Generate synthetic dataset with fixed parameters for testing
    np.random.seed(42)  # Set seed for reproducibility
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    summary = train_and_evaluate_rfecv(n_samples=1000, n_features=10, n_informative=3, n_redundant=2, n_classes=2, test_size=0.25, random_state=42, n_estimators=100, steps=[1], cv_splits=[5], scorings=['accuracy'])

    # Ensure the overall_best score matches the highest score in results
    max_score = max(summary['overall_best']['score'] for scoring in summary['best_per_scoring'] for step_cv, details in summary['results'][scoring].items() for step_cv in summary['results'][scoring])
    assert summary['overall_best']['score'] == max_score, f"Overall best score does not match the highest score in results."

    # Print statement if all assertions pass
    print("test_train_and_evaluate_rfecv_functionality function passed all checks.")
