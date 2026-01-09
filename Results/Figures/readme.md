# PRIME-EV — Figures and Visual Results

This directory contains all **visual results and figures** used in the evaluation of the **PRIME-EV** framework.  
Each figure is rendered below to provide **direct visual inspection** of architectural design, ablation behavior, deployment feasibility, and ethical AI performance.

---

## 📐 PRIME-EV Architecture

### Block-Level System Architecture

![PRIME-EV Architecture](prime_ev_architecture.png)

This figure illustrates the complete end-to-end architecture of PRIME-EV, showing the interaction between:

- **LISE** (Latent Infrastructure State Encoder)
- **PDDM** (Probabilistic Degradation and Disruption Model)
- **PPN** (Auxiliary Demand-Sensitive Performance Network)
- **RUN** (Ranking Utility Network)

The diagram highlights how structured representation, probabilistic risk, demand regularization, and ranking are jointly optimized within a unified pipeline.

---

## 🧪 Architecture-Oriented Ablation Analysis

### Training Loss Convergence Under Ablations

![Ablation Analysis](fig_ablation_prime_ev.png)

This figure presents training loss curves for:
- Individual module ablations (LISE, PDDM, PPN, RUN)
- System-level PRIME-EV ablations

Key observations:
- Lower loss does not guarantee correct station ranking
- Removing RUN collapses the ranking objective
- Removing probabilistic modeling increases instability
- Architectural coupling is essential for meaningful convergence

---

## ⚙️ Post-Training Optimization & Deployment Evaluation

The following radar plots visualize **system-level performance** under different optimization strategies.  
Each radar chart summarizes multiple metrics, including system stress, ethical fairness, operator risk balance, energy sustainability, and deployment latency.

---

### 🔹 No Optimization (Baseline)

![Radar None](Radar_O1_None.png)

This figure shows the baseline deployment behavior of PRIME-EV without pruning or quantization.

---

### 🔹 Pruning Optimization

![Radar Pruning](Radar_O2_Pruning.png)

This radar plot illustrates the effect of pruning on inference latency and deployment stability while preserving ethical behavior.

---

### 🔹 Quantization Optimization

![Radar Quantization](Radar_O3_Quantization.png)

This figure demonstrates the impact of quantization on efficiency and robustness, showing stable performance across most system-level metrics.

---

### 🔹 Pruning + Quantization

![Radar Pruning + Quantization](Radar_O4_Pruning+Quant.png)

This plot presents the combined optimization trade-off, achieving reduced latency with acceptable system stress and ethical fairness.

---

## 📊 Interpretation Summary

Across all figures, the following conclusions are visually supported:

- PRIME-EV maintains **stable deployment behavior** under optimization
- Ethical fairness and operator risk balance are **architecturally enforced**
- Ranking-aware learning is critical for valid infrastructure prioritization
- Post-training optimization does not compromise decision integrity when properly constrained

---

## 📌 Notes

- All figures are generated directly from experimental runs included in this repository.
- Filenames match references used in the main README and the paper.
- No manual post-processing was applied to alter or enhance results.

For full experimental details, refer to the main repository README and code notebooks.
