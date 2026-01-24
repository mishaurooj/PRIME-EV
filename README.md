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

- Structured latent infrastructure encoding capturing cross-attribute interactions  
- Explicit probabilistic modeling of operational risk and uncertainty  
- Auxiliary demand-aware regularization without inference contamination  
- Ranking-aware learning integrated directly into training  
- Unified end-to-end optimization with deployment and ethical AI constraints  

---

## 🌍 Dataset Description

<div style="background:#f0f9ff;border-left:6px solid #0284c7;padding:18px;border-radius:8px">

<b>Global EV Charging Stations Dataset (Kaggle)</b><br><br>

This work uses the public Global EV Charging Stations Dataset containing more than <b>5,000 charging station records</b> worldwide.
The dataset includes spatial, technical, operational, and sustainability-related attributes that support large-scale EV infrastructure analysis.

<ul>
<li><b>Location:</b> Geographic coordinates and spatial indicators</li>
<li><b>Charger Types:</b> AC Level 1, AC Level 2, DC Fast Chargers</li>
<li><b>Capacity:</b> Charging power (kW)</li>
<li><b>Operator:</b> Ownership and management details</li>
<li><b>Availability:</b> Public access and station status</li>
<li><b>Connectors:</b> Supported charging interfaces</li>
<li><b>Installation Year:</b> Infrastructure age</li>
<li><b>Renewable Usage:</b> Sustainability indicators</li>
<li><b>User Feedback:</b> Ratings and reviews</li>
<li><b>Maintenance:</b> Parking capacity and service frequency</li>
</ul>

Dataset link:  
https://www.kaggle.com/datasets/vivekattri/global-ev-charging-stations-dataset

</div>

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

## 📊 Detailed Experimental Tables

<!-- GitHub-safe CSS -->
<style>
.table-container { overflow-x:auto; margin:20px 0; }
table.primeev { border-collapse:collapse; width:100%; font-size:14px; text-align:center; }
table.primeev th, table.primeev td { border:1px solid #ccc; padding:6px; }
table.primeev th { background:#f2f2f2; }
.best { background:#d8f3dc; font-weight:bold; }
.worst { background:#ffe5e5; }
.note { font-size:13px; color:#555; margin-top:6px; }
</style>


### Architecture-Oriented Ablation Summary

<div class="table-container">
<table class="primeev">
<tr><th>Module</th><th>Variant</th><th>Loss</th><th>MSE</th><th>Interpretation</th></tr>
<tr><td>LISE</td><td>No CNN</td><td class="worst">0.7601</td><td class="worst">0.0668</td><td>Loss of interactions</td></tr>
<tr><td>PDDM</td><td>Deterministic</td><td>0.7099</td><td>0.0166</td><td>Overconfident risk</td></tr>
<tr><td>RUN</td><td>Pointwise</td><td class="best">0.0089</td><td>0.0087</td><td>Ranking collapse</td></tr>
</table>
<div class="note">Green = best, Red = worst</div>
</div>

---

## Ablation Analysis (25+ Models)

![Ablation Analysis](Results/Figures/fig_ablation_prime_ev.png)

---

## Optimization, Deployment & Ethical Evaluation (480+ Configurations)

![Radar None](Results/Figures/Radar_O1_None.png)  
![Radar Pruning](Results/Figures/Radar_O2_Pruning.png)  
![Radar Quantization](Results/Figures/Radar_O3_Quantization.png)  
![Radar Pruning+Quant](Results/Figures/Radar_O4_Pruning+Quant.png)

---

## How to Run

```bash
pip install torch numpy pandas scikit-learn matplotlib scipy
jupyter notebook Code/PRIME_EV_FULL.ipynb
```

---

## License

Apache 2.0 License

---

## Citation

```bibtex
@article{prime_ev,
  title   = {Probabilistic Representation--Integrated Modeling for EV Charging Infrastructure Placement and Recommendation},
  author  = {Khan, Misha Urooj, Suleman Ahmad and Kaleem, Zeeshan},
  journal = {IEEE Transactions},
  year    = {2026}
}
```

---

## Contact

Ajmal Khan — Sultan Qaboos University  
Misha Urooj Khan — CERN  
Zeeshan Kaleem — KFUPM
