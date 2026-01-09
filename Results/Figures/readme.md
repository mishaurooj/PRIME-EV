# PRIME-EV — Figures

This directory contains all **figures and visual artifacts** used in the evaluation and analysis of the **PRIME-EV** framework.  
Each figure corresponds directly to results reported in the accompanying paper and supports architectural analysis, ablation studies, deployment evaluation, and ethical AI assessment.

---

## 📐 Architecture Visualization

### `prime_ev_architecture.png`

This figure illustrates the **block-level architecture of the PRIME-EV framework**, showing the end-to-end integration of all learning modules:

- **LISE** — Latent Infrastructure State Encoder  
- **PDDM** — Probabilistic Degradation and Disruption Model  
- **PPN** — Auxiliary Demand-Sensitive Performance Network  
- **RUN** — Ranking Utility Network  

The diagram highlights how structured infrastructure representations, probabilistic risk estimates, demand regularization, and ranking utilities are jointly optimized within a unified pipeline.

---

## 🧪 Ablation Study Visualization

### `fig_ablation_prime_ev.png`

This figure presents the **training loss convergence curves** for:

- Individual module ablations (LISE, PDDM, PPN, RUN)
- System-level PRIME-EV ablations

It demonstrates that:
- Lower loss does not necessarily imply correct decision-making
- Removing ranking or probabilistic components can collapse utility ordering
- Architectural coupling is essential for stable and meaningful optimization

---

## ⚙️ Post-Training Optimization & Deployment Analysis

The following radar plots visualize **system-level performance** under different post-training optimization strategies.  
Each radar chart summarizes multiple deployment and ethical metrics, including system stress, fairness, risk balance, energy sustainability, and inference latency.

### `Radar_O1_None.png`
**Baseline (No Optimization)**  
Shows the reference deployment behavior of the full PRIME-EV model without pruning or quantization.

### `Radar_O2_Pruning.png`
**Pruning Optimization**  
Illustrates the impact of model pruning on deployment latency, system stress, and ethical fairness.

### `Radar_O3_Quantization.png`
**Quantization Optimization**  
Demonstrates the effect of weight quantization on efficiency and stability while preserving decision quality.

### `Radar_O4_Pruning+Quant.png`
**Pruning + Quantization**  
Shows the combined optimization trade-off, achieving reduced latency while maintaining acceptable ethical and stability metrics.

---

## 📊 Purpose of Figures

Collectively, these figures support the following claims:

- PRIME-EV maintains **deployment stability** under optimization
- Ethical behavior and fairness are **architecturally enforced**
- Ranking-aware learning is critical for valid infrastructure prioritization
- Post-training optimization does not compromise system integrity when properly constrained

These visualizations are intended to provide **transparent, reproducible, and reviewer-friendly evidence** of the framework’s behavior beyond predictive accuracy.

---

## 📌 Notes

- All figures are generated directly from experimental runs included in the repository.
- Filenames match references used in the paper and the main README.
- No post-processing or manual adjustment was applied to alter results.

For full experimental details, refer to the main repository README and the associated code notebooks.

