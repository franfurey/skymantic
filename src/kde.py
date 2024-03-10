# Import all the necessary libraries
import logging
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Generate synthetic data
np.random.seed(0)  # For reproducibility
X = np.concatenate((np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)))[:, np.newaxis]

# Set points to evaluate the KDEs
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

# True density of the data to calculate MISE
true_dens = (0.5 * norm(0, 1).pdf(X_plot[:, 0]) + 0.5 * norm(5, 1).pdf(X_plot[:, 0]))

# Function to calculate MISE
def calc_mise(true_density: np.array, kde_density: np.array) -> float:
    """Calculate the Mean Integrated Squared Error (MISE) for KDE estimates.
    
    Parameters:
        true_density (np.array): The true density function of the data.
        kde_density (np.array): The KDE density estimate.
    
    Returns:
        float: The MISE value.
    """
    try:
        mise = np.mean((true_density - kde_density) ** 2)
        return mise
    except Exception as e:
        logger.error(f"Error calculating MISE: {e}")
        return float('inf')

# Function to display KDE with different bandwidths
def plot_kde_bandwidth_and_kernel_comparison(X: np.array, X_plot: np.array, true_dens: np.array, bandwidths: List[float], kernels: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculate and return MISE values for different bandwidths and kernels.
    
    Parameters:
        X (np.array): The input data for KDE.
        X_plot (np.array): The x values to plot the KDE.
        true_dens (np.array): The true density function for comparison.
        bandwidths (List[float]): The list of bandwidths to use for KDE.
        kernels (List[str]): The list of kernels to use for KDE.
    
    Returns:
        Dict[str, Dict[str, float]]: A summary of results with the best performance per kernel and the overall best performance.
    """
    results = {}
    best_per_kernel = {}
    overall_best = {'kernel': None, 'bandwidth': None, 'mise': float('inf')}

    for kernel in kernels:
        best_per_kernel[kernel] = {'bandwidth': None, 'mise': float('inf')}
        for bandwidth in bandwidths:
            try:
                kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
                kde.fit(X)
                log_dens = kde.score_samples(X_plot)
                kde_density = np.exp(log_dens)
                mise = calc_mise(true_density=true_dens, kde_density=kde_density)
            except Exception as e:
                logger.error(f"Error in KDE calculation with kernel={kernel} and bandwidth={bandwidth}: {e}")
                mise = float('inf')

            # Update the results dictionary
            if kernel not in results:
                results[kernel] = {}
            results[kernel][bandwidth] = mise

            # Check for best performance per kernel
            if mise < best_per_kernel[kernel]['mise']:
                best_per_kernel[kernel] = {'bandwidth': bandwidth, 'mise': mise}

            # Check for overall best performance
            if mise < overall_best['mise']:
                overall_best = {'kernel': kernel, 'bandwidth': bandwidth, 'mise': mise}

    return {
        'results': results,
        'best_per_kernel': best_per_kernel,
        'overall_best': overall_best
    }


if __name__ == '__main__':
    # Generates bandwidths from 0.1 to 5.0, inclusive, with a step of 0.1
    bandwidths = np.arange(0.1, 5.1, 0.1).tolist()
    kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']  # Example kernels
    summary = plot_kde_bandwidth_and_kernel_comparison(X=X, X_plot=X_plot, true_dens=true_dens, bandwidths=bandwidths, kernels=kernels)
    print(summary)