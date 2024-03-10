from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import numpy as np

def train_and_evaluate_rfecv(n_samples: int = 1000, n_features: int = 10, n_informative: int = 3, n_redundant: int = 2, n_classes: int = 2, test_size: float = 0.25, random_state: int = 42, n_estimators: int = 100, steps: list = [1], cv_splits: list = [5], scorings: list = ['accuracy']):
    """
    Trains a classification model and selects the most important features through RFECV,
    iterating over different configurations and scoring metrics. Returns a structured summary of the results.
    """
    # Generate synthetic dataset based on the specified parameters.
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_classes=n_classes, random_state=random_state)
    # Split dataset into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    results = {}
    best_per_scoring = {}
    overall_best = {'score': -np.inf, 'details': None}

    for scoring in scorings:  # Iterate over different scoring metrics.
        results[scoring] = {}
        best_per_scoring[scoring] = {'score': -np.inf, 'details': None}
        
        for step in steps:  # Iterate over different steps (number of features to remove at each iteration).
            for cv in cv_splits:  # Iterate over different numbers of cross-validation splits.
                # Initialize and train the RandomForestClassifier.
                forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                # Initialize and fit RFECV with current configuration.
                rfecv = RFECV(estimator=forest, step=step, cv=StratifiedKFold(cv), scoring=scoring, min_features_to_select=1)
                rfecv.fit(X_train, y_train)
                
                # Obtain the highest mean test score for the current configuration.
                score = rfecv.cv_results_['mean_test_score'].max()
                # Store configuration details.
                config_details = {
                    'step': step,
                    'cv': cv,
                    'optimal_features': rfecv.n_features_,
                    'support': rfecv.support_,
                    'ranking': rfecv.ranking_,
                    'mean_test_score': rfecv.cv_results_['mean_test_score']
                }
                
                # Record the details of the current configuration under the current scoring metric.
                results[scoring][(step, cv)] = config_details
                
                # Update best model per scoring metric if the current score is higher.
                if score > best_per_scoring[scoring]['score']:
                    best_per_scoring[scoring] = {'score': score, 'details': config_details}
                
                # Update the overall best model if the current score is the highest encountered so far.
                if score > overall_best['score']:
                    overall_best = {'score': score, 'details': config_details}
    
    # Return a structured summary of all results, best results per scoring metric, and the overall best result.
    return {
        'results': results,
        'best_per_scoring': best_per_scoring,
        'overall_best': overall_best
    }

# Example usage demonstrating how to call the function with specified parameters.
if __name__ == '__main__':
    steps = [1, 2]  # Different steps to try.
    cv_splits = [4, 5]  # Different numbers of cross-validation splits to try.
    scorings = ['accuracy', 'f1_macro']  # Different scoring metrics to try.
    summary = train_and_evaluate_rfecv(steps=steps, cv_splits=cv_splits, scorings=scorings)
    print(summary)
