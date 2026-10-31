# MPI-Cactus-Temporal-models-of-light-evoked-physiological-responses-
# Convolution-Based Temporal Modeling of the Pupillary Light Reflex (PLR)
**Extracting Hidden Physiological Dynamics via Non-Linear Optimization**

This repository contains a mathematical and computational framework for modeling the human nervous system's dynamic response to light. Rather than relying on purely descriptive statistics, this project treats the Pupillary Light Reflex (PLR) as a **Linear Time-Invariant (LTI) System**, utilizing convolution-based response models and numerical optimization to solve the inverse problem of parameter extraction.

This work directly aligns with computational approaches in translational and circadian neuroscience, aiming to capture transient, sustained, and persistent components of physiological time-series data.

---

## 1. The Forward Model: Physiological Convolution
The biological system processes an incoming light stimulus $s(t)$ and produces a mechanical output (pupil constriction) $y(t)$. We model this biological transformation mathematically as a convolution of the stimulus with a **Pupillary Response Function (PRF)** kernel, denoted as $h(t)$:

$$ y(t) = (s * h)(t) + \epsilon(t) = \int_{0}^{t} s(\tau) h(t - \tau) d\tau + \epsilon(t) $$

Where $\epsilon(t)$ represents the inherent biological baseline and measurement noise from the eye-tracker.

### The Physiological Kernel (Gamma Distribution)
To accurately represent the neuromuscular delay and the asymmetric nature of the parasympathetic response (fast constriction, slower recovery), the kernel $h(t)$ is defined using a modified Gamma distribution:

$$ h(t) = 
\begin{cases} 
A \cdot (t - t_d)^n e^{-\frac{t - t_d}{\tau}} & \text{for } t \ge t_d \\
0 & \text{for } t < t_d 
\end{cases} 
$$

*   $t_d$ (**Latency**): The physiological delay (e.g., signal travel time from retina to midbrain to sphincter pupillae).
*   $A$ (**Amplitude**): The maximum capacity for constriction.
*   $n$ (**Shape Parameter**): Controls the steepness of the initial transient response.
*   $\tau$ (**Time Constant**): Governs the exponential decay/recovery phase.

---

## 2. Solving the Inverse Problem (Numerical Optimization)
In a real-world experimental setting, the physiological parameters $\theta = \{t_d, n, \tau, A, \text{baseline}\}$ are hidden. We only observe the noisy macroscopic output $y_{obs}(t)$. 

To extract these hidden biological features, the system frames the challenge as a **Non-linear Least Squares Optimization Problem**. We seek the optimal parameter set $\hat{\theta}$ that minimizes the residual sum of squares (RSS) between the observed data and our LTI forward model:

$$ \hat{\theta} = \arg\min_{\theta} \sum_{i=1}^{N} \left[ y_{obs}(t_i) - \Big( (s * h_{\theta})(t_i) + \text{baseline} \Big) \right]^2 $$

Using the `scipy.optimize.curve_fit` implementation (Trust-Region Reflective algorithm), the model enforces strict biological boundary conditions (e.g., $0.1 \le t_d \le 0.5$). 

This rigorous mathematical fitting effectively reconstructs the true underlying physiology, isolating the signal from heavy environmental and sensor noise without using destructive moving-average smoothing.

<div align="center">
  <!-- TODO: Save your optimization plot as plr_optimization.png in a 'docs' folder -->
  <img src="docs/plr_optimization.png" alt="Optimization Results" width="700"/>
  <p><i>The optimizer successfully converges, reconstructing the true physiological kernel from heavily noised observations.</i></p>
</div>

---

## 3. Extensibility to Multimodal Data
While this initial repository focuses on pupillometry, the convolution-based LTI framework is highly modular. The underlying mathematics can be seamlessly adapted to other light-evoked responses:
    
1.  **Facial EMG:** By modifying the kernel $h(t)$ to reflect different neuromuscular time constants, the same optimization loop can extract muscle activation dynamics.
2.  **Computer-Vision Action Units (AUs):** Shared latent components (like common latencies or integrated mEDI daylight doses) can be jointly optimized across multimodal time-series.
    
This structured approach ensures parameters are not only mathematically identifiable but also biologically interpretable across different participants and lighting conditions.
