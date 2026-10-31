import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit


# Task 2.1: Mathematical Formulation
def pupillary_response_function(t, latency, n, tau, amplitude):
    """
    Generates the Pupillary Response Function (PRF).
    """
    prf = np.zeros_like(t)
    valid_indices = t >= latency

    t_val = t[valid_indices] - latency

    # Avoid numerical instability with very large powers by clipping
    t_val_clipped = np.clip(t_val, 0, 10)
    prf_raw = (t_val_clipped ** n) * np.exp(-t_val_clipped / tau)

    prf[valid_indices] = amplitude * prf_raw

    # Normalize
    area = np.sum(np.abs(prf))
    if area > 0:
        prf = (prf / area) * np.abs(amplitude) * np.sign(amplitude)

    return prf


# Task 2.2: Synthetic Stimulus & Data Generation
def generate_synthetic_plr_data():
    """ Simulates step light stimulus and noisy biological observation. """
    fs = 100
    t = np.arange(0, 10, 1 / fs)

    stimulus = np.zeros_like(t)
    stimulus[(t >= 1.0) & (t <= 3.0)] = 1.0

    # True physiological parameters
    true_params = {'latency': 0.25, 'n': 10.0, 'tau': 0.05, 'amplitude': -2.0, 'baseline': 7.0}

    prf_kernel = pupillary_response_function(t, true_params['latency'], true_params['n'], true_params['tau'],
                                             true_params['amplitude'])
    pupil_change = convolve(stimulus, prf_kernel, mode='full')[:len(t)]

    true_signal = true_params['baseline'] + pupil_change
    noisy_pupil = true_signal + np.random.normal(0, 0.05, len(t))

    return t, stimulus, noisy_pupil, true_signal, true_params


# Task 2.3: Numerical Optimization & Fitting
def fit_physiological_model(t, stimulus, noisy_data):
    """
    Solves the inverse problem: Extracts hidden physiological parameters from noisy data.
    """

    # 1. Define the forward model closure so curve_fit can use the known stimulus
    def forward_model(t_func, latency, n, tau, amplitude, baseline):
        kernel = pupillary_response_function(t_func, latency, n, tau, amplitude)
        pupil_response = convolve(stimulus, kernel, mode='full')[:len(t_func)]
        return baseline + pupil_response

    # 2. Initial Guesses (p0) and Biological Bounds
    # Parameters: [latency, n, tau, amplitude, baseline]
    p0 = [0.2, 8.0, 0.1, -1.0, 6.0]

    # Bounds prevent the optimizer from testing biologically impossible values (e.g., negative latency)
    bounds = (
        [0.1, 5.0, 0.01, -5.0, 4.0],  # Lower bounds
        [0.5, 15.0, 0.2, -0.1, 9.0]  # Upper bounds
    )

    # 3. Non-linear Least Squares Optimization
    print("Starting numerical optimization (Curve Fitting)...")
    optimized_params, covariance = curve_fit(forward_model, t, noisy_data, p0=p0, bounds=bounds)

    # 4. Generate the fitted curve using the discovered parameters
    fitted_curve = forward_model(t, *optimized_params)

    return optimized_params, fitted_curve


# Execution and visualization block
if __name__ == "__main__":
    # Generate data
    t, stim, noisy_pupil, true_signal, true_params = generate_synthetic_plr_data()

    # Run Optimizer
    fitted_params, fitted_curve = fit_physiological_model(t, stim, noisy_pupil)

    # Print results comparison
    print("\n--- Optimization Results ---")
    param_names = ['Latency (s)', 'Shape (n)', 'Time Const (tau)', 'Amplitude', 'Baseline (mm)']
    true_vals = list(true_params.values())
    for i, name in enumerate(param_names):
        print(f"{name:<18}: True = {true_vals[i]:>6.2f} | Fitted = {fitted_params[i]:>6.2f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, noisy_pupil, color='lightgray', label='Noisy Sensor Data', zorder=1)
    plt.plot(t, true_signal, color='red', lw=2, linestyle='--', label='True Physiology', zorder=2)
    plt.plot(t, fitted_curve, color='blue', lw=2, label='Fitted Mathematical Model', zorder=3)

    plt.title("Numerical Optimization of Pupillary Light Reflex (PLR)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pupil Diameter (mm)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()