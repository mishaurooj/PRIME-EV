# PRIME-EV

**PRIME-EV: Probabilistic Representation–Integrated Modeling for EV Charging Infrastructure Placement and Recommendation**

Official research repository accompanying the PRIME-EV framework, providing complete source code, datasets, trained model weights, ablation studies, optimization analysis, and ethical AI evaluation.

Repository: https://github.com/mishaurooj/PRIME-EV

---

## Overview

Electric vehicle (EV) charging infrastructure planning must operate under uncertainty, heterogeneous demand, infrastructure degradation, and ethical constraints. Most existing approaches decouple representation learning, risk modeling, demand handling, and station ranking, leading to unstable or biased decisions.

PRIME-EV introduces a unified probabilistic learning framework that jointly integrates structured infrastructure representation, probabilistic degradation modeling, demand-sensitive regularization, ranking-aware utility learning, and deployment-aware ethical AI constraints within a single optimization pipeline.

---

## PRIME-EV Architecture

![PRIME-EV Architecture](Results/Figures/prime_ev_architecture.png)

The framework consists of four tightly coupled modules:
- **LISE**: Latent Infrastructure State Encoder
- **PDDM**: Probabilistic Degradation and Disruption Model
- **PPN**: Auxiliary Demand-Sensitive Performance Network
- **RUN**: Ranking Utility Network

---

## Key Contributions

- Structured latent infrastructure encoding capturing cross-attribute interactions.
- Explicit probabilistic modeling of operational risk and uncertainty.
- Auxiliary demand-aware regularization without inference contamination.
- Ranking-aware learning integrated directly into training.
- Unified end-to-end optimization with deployment and ethical AI constraints.

---

## Comparison with Prior EV Charging Frameworks

| Model Category | DT | SR | PR | DR | ER | UO |
|---------------|----|----|----|----|----|----|
| Prior Works (2016–2025) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Partial DT Models | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Structured-only Models | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Ranking-only Models | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| **PRIME-EV** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

---

## Ablation Analysis (25+ Models)

![Ablation Analysis](Results/Figures/fig_ablation_prime_ev.png)

Extensive ablations demonstrate that reducing loss alone does not guarantee valid station prioritization. Removing key modules degrades ranking stability, deployment robustness, or ethical behavior even when predictive loss decreases.

---

## Optimization, Deployment & Ethical Evaluation (480+ Configurations)

![Radar None](Results/Figures/Radar_O1_None.png)
![Radar Pruning](Results/Figures/Radar_O2_Pruning.png)
![Radar Quantization](Results/Figures/Radar_O3_Quantization.png)
![Radar Pruning+Quant](Results/Figures/Radar_O4_Pruning+Quant.png)

PRIME-EV is evaluated across more than 480 model–optimization–ethics configurations, including pruning and quantization. The framework achieves the lowest system stress, balanced ethical fairness, and stable sub-millisecond inference latency.

---

## Repository Structure

```
PRIME-EV/
├── Code/
│   ├── PRIME_EV_FULL.ipynb
│   └── optimization_ethics.ipynb
├── Dataset/
│   └── ev_charging_stations.csv
├── Results/
│   └── Figures/
├── Trained-model-weights/
├── prime-evblock.pptx
├── LICENSE
└── README.md
```

---

## How to Run

### Install Dependencies
```
pip install torch numpy pandas scikit-learn matplotlib scipy
```

### Train PRIME-EV and Ablations
```
jupyter notebook Code/PRIME_EV_FULL.ipynb
```

### Run Optimization and Ethical Analysis
```
jupyter notebook Code/optimization_ethics.ipynb
```

---

## License

This project is released under the Apache 2.0 License.

---

## Citation

```
@article{prime_ev,
  title   = {Probabilistic Representation--Integrated Modeling for EV Charging Infrastructure Placement and Recommendation},
  author  = {Khan, Ajmal and Khan, Misha Urooj and Kaleem, Zeeshan},
  journal = {IEEE Transactions},
  year    = {2026}
}
```

---

## Contact

Ajmal Khan — Sultan Qaboos University  
Misha Urooj Khan — CERN  
Zeeshan Kaleem — KFUPM
