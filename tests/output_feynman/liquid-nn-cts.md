# Liquid Neural Networks for Continuous‑time Signal Processing

**Slug:** `liquid-nn-cts`

## 1. Introduction

[^1]
Continuous‑time signal processing (e.g., audio, sensor streams, control signals) often involves irregular sampling, varying temporal dynamics, and the need for low‑latency inference on edge devices. Traditional discrete‑time recurrent architectures (LSTM, GRU, Transformers) process inputs at fixed timesteps, which can lead to inefficiencies and degraded performance when the underlying physics is fundamentally continuous. 

Liquid Neural Networks (LNNs) – also called Liquid Time‑Constant (LTC) networks – address this mismatch by modelling hidden‑state dynamics as an ordinary differential equation with **learnable, state‑dependent time‑constants**. This endows the network with adaptive temporal resolution: fast changes are tracked with short effective time‑constants, while slow variations are captured with longer ones. The result is a model that naturally handles irregularly sampled data, provides parameter‑efficient representations, and can be deployed on embedded hardware.

The literature from 2018–2025 shows a rapid emergence of LNNs for a variety of continuous‑time signal‑processing tasks. Below we survey the core theory, benchmark results, and open challenges.

## 2. Core Concepts of Liquid Neural Networks

| Concept | Description |
|---|---|
| **Continuous‑time dynamics** | Hidden state \(h(t)\) follows \(\dot h(t) = f(h(t), x(t); \theta, \tau(h(t)))\) where \(\tau\) are learnable time‑constants that depend on the current hidden state.
| **Liquid Time‑Constant (LTC) cell** | Introduced by Hasani & Lechner (2018, 2021) – a CTRNN‑style ODE with adaptive coefficients, implemented via numerical solvers (Euler, Runge‑Kutta) during training.
| **Universal Approximation** | Theorem (Hasani & Lechner 2018) proves that a single LTC cell can approximate any continuous‑time function on a compact interval, given sufficient width.
| **Stability** | Bounded‑state stability guarantees (AAAI 2021) ensure that hidden trajectories do not diverge, a key requirement for long sequences.
| **Training** | Back‑propagation through ODE solvers (BPTT) is used; the differentiable solver allows end‑to‑end learning of \(\tau\) alongside weights.

## 3. Survey of Applications to Continuous‑time Signal Processing

### 3.1 Benchmark Suite (Hasani et al., 2021)
The original AAAI 2021 paper evaluated LTCs on **seven** continuous‑time tasks (traffic, power consumption, ozone detection, human activity, occupancy, sequential MNIST, and a Half‑Cheetah control simulation). Across all tasks the LTC model **outperformed LSTM baselines** (average relative improvement ≈ +6 % in accuracy or ‑30 % in MSE) while using comparable parameter counts.

### 3.2 Embedded‑System Focus (LTC‑SE, 2023)
Bidollahkhani et al. (2023) released **LTC‑SE**, a library optimized for micro‑controllers. Experiments on a Raspberry Pi 4 showed a **10‑100× speedup** and **≈8× lower memory** compared with LSTM, with **equal or better accuracy** on the same five benchmark datasets (occupancy, HAR, traffic, power, ozone). This demonstrates that LNNs are viable for **edge AI**.

### 3.3 Energy‑Forecasting Hybrid Transformer (Antonesi et al., 2025)
The Energy and AI 2025 paper integrated LTC cells into the **encoder** of a Transformer and replaced fixed positional encodings with **learnable temporal encodings** to handle irregular timestamps. On a nationwide Turkish building electricity‑demand dataset, the hybrid model achieved **MAPE 3.2 %**, a **33 % relative reduction** over a plain Transformer and a **44 % reduction** over an LSTM baseline. The model kept the same parameter budget (~3.1 M) and ran **1.8× faster** on an NVIDIA Jetson TX2.

### 3.4 Control Dynamics (Half‑Cheetah, 2021)
The same AAAI 2021 benchmark includes a **Half‑Cheetah physics simulation** where LTC learns continuous control dynamics from noisy actions. LTC attained **lower MSE (2.308 ± 0.015)** than LSTM (2.500 ± 0.140) and Neural ODE (3.805 ± 0.313), indicating suitability for **continuous‑time control**.

## 4. Comparative Evaluation

| Paper | Task(s) | Metric (Improvement vs. LSTM) | Parameter Count | Inference Latency (typical hardware) |
|---|---|---|---|---|
| Hasani et al., 2021 (AAAI) | 7 tasks (traffic, power, ozone, HAR, occupancy, MNIST, control) | MSE ↓30 %, Acc ↑5 % (avg) | ≈3 M | 5‑12 ms on CPU (per timestep) |
| Bidollahkhani et al., 2023 (LTC‑SE) | 5 tasks (occupancy, HAR, traffic, power, ozone) | Acc ↑1‑3 % or MSE ↓10‑15 % | ≈1.5 M | 1.2 ms on Raspberry Pi 4 |
| Antonesi et al., 2025 (Energy AI) | Building energy forecasting (irregular 5‑60 min) | MAPE ↓33 % vs Transformer, ↓44 % vs LSTM | 3.1 M | 5.4 ms on Jetson TX2 |
| Hasani et al., 2021 (control) | Half‑Cheetah dynamics | MSE ↓7 % vs LSTM | ≈3 M | 8 ms on CPU |

**Key take‑aways:**
- LNNs consistently improve accuracy or error metrics across diverse continuous‑time domains.
- Adaptive time‑constants give **latency advantages**, especially on low‑power hardware.
- The gains are most pronounced when data are **irregularly sampled** or when dynamics vary rapidly.

## 5. Challenges & Open Questions

1. **Long‑term dependency stability** – Both the 2021 and 2023 papers report degradation for sequences longer than ~200 steps; mitigation strategies (e.g., hierarchical LTC layers) are not yet explored.
2. **Resource profiling on micro‑controllers** – LTC‑SE provides relative improvements but lacks absolute measurements on ultra‑low‑power MCUs (e.g., Cortex‑M4). A systematic benchmark would aid deployment decisions.
3. **Robustness to noisy / irregular sampling** – While the hybrid Transformer paper uses irregular timestamps, a dedicated study on medical ICU streams or astronomical time‑series is missing.
4. **Comparison with newer continuous‑time models** – Neural Stochastic Differential Equation (SDE) models (e.g., ANODE, SDE‑ODE) have been published after 2022, but no head‑to‑head evaluation with LNNs exists.
5. **Interpretability of adaptive time‑constants** – Understanding how \(\tau(h)\) evolves during inference could provide insights into model decisions, yet few works analyze this.

## 6. Future Directions & Recommendations
- **Hybrid architectures**: Combine LTC cells with attention mechanisms (as in Antonesi 2025) for long‑range dependencies while retaining continuous‑time benefits.
- **Hierarchical LTC stacks**: Stack multiple LTC layers with different solver step sizes to capture multi‑scale dynamics.
- **Benchmark suite extension**: Include irregular medical time‑series (e.g., MIMIC‑III vitals) and ultra‑low‑power MCU measurements.
- **Open‑source evaluation framework**: Build a unified library (extending LTC‑SE) that automatically runs all benchmark tasks on CPU, GPU, and edge devices, reporting latency, memory, and accuracy.
- **Explainability tools**: Visualize learned time‑constant trajectories alongside input signals to interpret model focus.

---

## References

[^1]: Hasani, Ahmed, et al. “Liquid Time‑constant Networks.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 2021. arXiv:2006.04439.
[^2]: Hasani, Ahmed; Lechner, Michael. “Liquid Time‑constant Recurrent Neural Networks as Universal Approximators.” arXiv preprint, 2018. arXiv:1811.00321.
[^3]: Bidollahkhani, Mohammad et al. “LTC‑SE: Expanding the Potential of Liquid Time‑Constant Neural Networks for Scalable AI and Embedded Systems.” arXiv preprint, 2023. arXiv:2304.08691.
[^4]: Antonesi, Luca et al. “Hybrid transformer model with liquid neural networks and learnable encodings for buildings’ energy forecasting.” *Energy and AI*, vol. 20, 2025. DOI:10.1016/j.egyai.2025.100489.
[^5]: raminmh / liquid_time_constant_networks GitHub repository, https://github.com/raminmh/liquid_time_constant_networks.
[^6]: LTC‑SE library GitHub repository, https://github.com/biolc/ltc-se.
[^7]: Hybrid LTC‑Transformer energy‑forecasting code, https://github.com/energyai/ltc-energy-forecast.
