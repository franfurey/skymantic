import numpy as np
from scipy.stats import norm
from kde import calc_mise, plot_kde_bandwidth_and_kernel_comparison

def test_calc_mise():
    """Test the calc_mise function with a known input and output."""
    true_dens = np.array([0.2, 0.5, 0.2])
    kde_dens = np.array([0.1, 0.4, 0.3])
    x_plot = np.array([1, 2, 3])
    expected_mise = np.mean((true_dens - kde_dens) ** 2)
    assert calc_mise(true_density=true_dens, kde_density=kde_dens) == expected_mise, f"The expected MISE value is: + {expected_mise}, instead got: {calc_mise(true_density=true_dens, kde_density=kde_dens, x_plot=x_plot)}"
    print(f"calc_mise function is working correctly, the MISE value is: {expected_mise}")

def test_plot_kde_bandwidth_and_kernel_comparison_structure():
    """Test to ensure the function returns a structure with non-empty values for different kernels and bandwidths."""
    X = np.random.normal(0, 1, 10)[:, np.newaxis]
    X_plot = np.linspace(-5, 5, 100)[:, np.newaxis]
    true_dens = norm(0, 1).pdf(X_plot[:, 0])
    bandwidths = np.arange(0.1, 5.1, 0.1).tolist()  # Example bandwidths
    kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']  # Example kernels
    summary = plot_kde_bandwidth_and_kernel_comparison(X=X, X_plot=X_plot, true_dens=true_dens, bandwidths=bandwidths, kernels=kernels)

    # Check for the main keys
    assert 'results' in summary and isinstance(summary['results'], dict), f"Expected 'results' key with a dict value."
    assert 'best_per_kernel' in summary and isinstance(summary['best_per_kernel'], dict), f"Expected 'best_per_kernel' key with a dict value."
    assert 'overall_best' in summary and isinstance(summary['overall_best'], dict), f"Expected 'overall_best' key with a dict value."

    # Check for non-empty responses for each kernel in results
    for kernel in kernels:
        assert kernel in summary['results'], f"Kernel {kernel} is missing in the results."
        assert all(bandwidth in summary['results'][kernel] for bandwidth in bandwidths), f"Some bandwidths are missing for kernel {kernel}."
        assert all(isinstance(summary['results'][kernel][bandwidth], float) for bandwidth in bandwidths), f"Expected float values for kernel {kernel} results."

    # Check for valid 'best_per_kernel' structure and non-empty values
    for kernel, details in summary['best_per_kernel'].items():
        assert 'bandwidth' in details and 'mise' in details, f"Missing 'bandwidth' or 'mise' in best_per_kernel for {kernel}."
        assert isinstance(details['bandwidth'], float), f"Expected 'bandwidth' to be float in best_per_kernel for {kernel}, but it gets {details['bandwidth']}"
        assert isinstance(details['mise'], float), f"Expected 'mise' to be float in best_per_kernel for {kernel}, but it gets {details['mise']}"

    # Check 'overall_best' structure and non-empty values
    assert 'kernel' in summary['overall_best'] and summary['overall_best']['kernel'] in kernels, "Invalid or missing 'kernel' in 'overall_best'."
    assert 'bandwidth' in summary['overall_best'] and isinstance(summary['overall_best']['bandwidth'], float), "Invalid or missing 'bandwidth' in 'overall_best'."
    assert 'mise' in summary['overall_best'] and isinstance(summary['overall_best']['mise'], float), "Invalid or missing 'mise' in 'overall_best'."

    # Print statement if all assertions pass
    print("test_plot_kde_bandwidth_and_kernel_comparison_structure function passed all checks.")
