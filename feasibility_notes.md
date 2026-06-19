# Feasibility Study Design Brief: Edge-Conveyor Physical Object Grading

This document outlines the technical parameters, mathematical constraints, and hardware latency budgets for the automated physical object grading system assessment.

---

## 1. Operational Parameters & Latency Budget
High-throughput sorting conveyors require real-time processing of fast-moving items.

* **Conveyor Line Throughput**: 30 to 60 objects per second.
* **Total Latency Budget per Object**: $\le 16.6\text{ ms}$ (for 60 FPS conveyor capture).
* **Backbone Forward Pass Target**: $\le 5.0\text{ ms}$ (allowing remaining time for image acquisition, decoding, preprocessing, and mechanical ejector signaling).
* **Hardware Profile**: Low-power edge processors (e.g., Raspberry Pi 4/5, Jetson Nano, Intel NUC CPU) deployed directly on the conveyor frame.

---

## 2. Mathematical Formulations

### A. Data Quality: Sharpness Metric
To detect defects (micro-scratches, discoloration), images must be blur-free. We calculate the variance of the Laplacian convolution to determine edge crispness:

$$\Delta = \text{Var}(\nabla^2 I)$$

Where:
* $I$ is the grayscale representation of the input image.
* $\nabla^2$ is the Laplacian operator, implemented as a 2D convolution with a $3 \times 3$ edge-detecting kernel:
  $$K = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
* A low variance $\Delta$ indicates a lack of high-frequency edge gradients, signaling motion blur or poor camera focus.

### B. Expected Calibration Error (ECE)
In conveyor routing, borderline confidence predictions must be sent to manual audit lines. The model's confidence must reflect empirical accuracy. ECE partitions prediction confidences into $M$ equally spaced bins:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where:
* $N$ is the total number of validation samples.
* $B_m$ is the set of indices of samples whose prediction confidence falls in the $m$-th bin interval.
* $\text{acc}(B_m)$ is the average classification accuracy in bin $B_m$.
* $\text{conf}(B_m)$ is the average prediction confidence in bin $B_m$.

---

## 3. Architecture Feasibility Matrix

We evaluate two lightweight vision backbones to establish baseline feasibility for edge deployment:

| Metric | MobileNet-v3 Small | ShuffleNet-v2 x0.5 | Target Constraints |
| :--- | :--- | :--- | :--- |
| **Parameters** | 2.54 Million | 1.36 Million | $< 5.0\text{ Million}$ (Memory-bound edge) |
| **MACs (FLOPs)**| ~56 Million | ~43 Million | $< 100\text{ Million}$ |
| **Target Latency**| $< 8.0\text{ ms}$ | $< 5.0\text{ ms}$ | $\le 10.0\text{ ms}$ (CPU) |
| **Key Advantage**| Squeeze-and-Excitation attention | Channel Shuffle grouping | Real-time Edge Compatibility |

---

## 4. Assessment Implementation Checklist
- [x] Configure telemetry logging routed to stdout (prevent stream interlacing).
- [x] Implement convolutional Laplacian sharpness metrics (device-aware).
- [x] Set up baseline Mobilenet-v3 and ShuffleNet-v2 model heads for 3-class grading.
- [x] Wrap forward passes in a thread-safe `torch.no_grad()` context with GPU warmup passes.
- [x] Calculate Expected Calibration Error (ECE) across confidence bins.
- [x] Export an executive decision-grade ASCII text report.
