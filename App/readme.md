# PRIME-EV Interactive Decision Dashboard

PRIME-EV is an interactive Streamlit dashboard for EV charging station prioritization. It lets you enter a station ID such as `EVS00001` or `EVS00050`, then runs station-level risk assessment, utility ranking, deployment impact analysis, baseline comparison, similarity search, map visualization, and top-priority station screening.

## Dashboard Preview

### 1. Executive Overview
![Executive overview](p1.png)

### 2. Station Profile and Diagnostic Radar
![Station profile and radar](p2.png)

### 3. Baseline Comparison
![Baseline comparison](p3.png)

### 4. Risk versus Utility Planning View
![Risk utility planning view](p4.png)

### 5. Geographic Station View and Similar Stations
![Map and similar stations](p5.png)

### 6. Top PRIME-EV Priority Stations
![Top priority stations](p6.png)

### 7. Raw Dataset and Downloadable Results
![Raw results](p7.png)

## Repository Structure

```text
PRIME-EV/
├── App/
│   ├── Code/
│   │   └── prime_ev_dashboard.py
│   ├── p1.png
│   ├── p2.png
│   ├── p3.png
│   ├── p4.png
│   ├── p5.png
│   ├── p6.png
│   ├── p7.png
│   ├── prime_ev_filtered_station_results.csv
│   └── readme.md
├── Dataset/
│   └── ev_charging_stations-dataset.csv
├── Results/
├── Trained-model-weights/
└── README.md
```

## Required Dataset

Place the EV charging station dataset at:

```text
D:\other\prime-ev\Dataset\ev_charging_stations-dataset.csv
```

The dashboard also supports manual CSV upload from the sidebar.

Expected dataset size used in the dashboard:

```text
5,000 EV charging stations
Station IDs: EVS00001 to EVS05000
```

Required columns include:

```text
Station ID
Address
Charger Type
Station Operator
Connector Types
Availability
Cost (USD/kWh)
Distance to City (km)
Usage Stats (avg users/day)
Charging Capacity (kW)
Installation Year
Reviews (Rating)
Parking Spots
Renewable Energy Source
Maintenance Frequency
Latitude
Longitude
```

## Environment Setup

### Option 1: Conda Environment

```bash
conda create -n prime-ev python=3.10 -y
conda activate prime-ev
```

Install packages:

```bash
pip install streamlit pandas numpy scikit-learn plotly pydeck
```

### Option 2: Requirements File

Create `requirements.txt` with:

```text
streamlit
pandas
numpy
scikit-learn
plotly
pydeck
```

Then run:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

Go to the application folder:

```bash
cd /d D:\other\prime-ev\App\Code
```

Start Streamlit:

```bash
streamlit run prime_ev_dashboard.py
```

Open the local URL shown in the terminal:

```text
http://localhost:8501
```

## How to Use

1. Start the dashboard.
2. Upload the EV charging station CSV from the sidebar if the file is not auto-loaded.
3. Enter a station ID, for example:

```text
EVS00001
EVS00050
EVS02775
```

4. Adjust the Top-K slider to control how many priority stations are displayed.
5. Use the tabs to inspect:
   - Station Analysis
   - Model and Baselines
   - Planning Visuals
   - Top Stations
   - Raw Data

## Main Dashboard Outputs

### Station-Level Outputs

The dashboard computes:

```text
PRIME-EV utility score
PRIME-EV risk score
Deployment impact score
Demand-capacity fit
Cost-accessibility fit
Network rank
```

Example output for `EVS00050`:

```text
Network rank: 1,070 of 5,000
Utility score: 56.1/100
Risk score: 39.8/100
Deployment impact: 59.4/100
Demand-capacity fit: 78.8/100
```

### Baseline Methods

The app compares PRIME-EV with practical prioritization baselines:

```text
DistanceOnly
QualityGreen
Gradient-Boosted Ranker
AHP
Random Forest
MultiObjective
DemandCapacity
CostOnly
```

These comparisons help verify whether PRIME-EV gives a balanced decision rather than ranking stations using a single factor.

### Planning Visuals

The dashboard provides:

```text
Risk vs utility scatter plot
Geographic station map
Similar-station table
Top priority station table
Top station utility chart
Raw filtered dataset
```

## Export Results

Use the **Download filtered station results** button in the Raw Data tab to export the processed station table.

The exported file contains the original station attributes plus PRIME-EV outputs:

```text
PRIME_UtilityScore_100
PRIME_RiskScore_100
PRIME_DeploymentImpact_100
NetworkRank
```


## Notes

- The dashboard is designed for decision support and planning analysis.
- PRIME-EV scores are normalized to a 0 to 100 range for interpretability.
- The map requires valid `Latitude` and `Longitude` columns.
- If a station ID is not found, check the format. Use five digits after `EVS`, such as `EVS00001`.

## Citation Statement

If you use this dashboard in a paper, describe it as:

```text
The PRIME-EV dashboard provides station-level decision support by integrating utility ranking, risk estimation, deployment impact scoring, baseline comparison, geospatial inspection, and similar-station retrieval in an interactive Streamlit interface.
```

