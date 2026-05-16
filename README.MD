# Engineering Data Systems Pipeline: SOL-04
**Project Title:** Automated Torsional Rigidity Analysis of EV Drivetrain Shafts under Severe Terrain Conditions  
**Student Name:** Ram Anthony E. Pampliega  
**Course & Section:** Computer Programming (BSME - 1C)  
**Academic Year:** 2026  
**Engineering Pillar:** Pillar 3 - Solid Mechanics & Stress Analysis  
**Institution:** Department of Mechanical Engineering, Technological University of the Philippines – Manila  

---

## I. Project Overview
This project implements a robust, automated Python-based data pipeline designed to analyze torsional rigidity and fatigue-loading profiles in Electric Vehicle (EV) drivetrain shafts. By isolating extreme operational boundary conditions from large-scale predictive maintenance telemetry using an engineering-driven threshold filter, the system calculates critical mechanical stress metrics and tracks cumulative system degradation over time.

---

## II. System Architecture
The software is developed utilizing **Object-Oriented Programming (OOP)** patterns to ensure modularity, loose coupling, and enterprise scalability:

1. **Data Ingestion Module (`module_1_ingestion`)**: Handles raw CSV telemetry loading using secure `try-except` blocks to eliminate runtime crashes and sandboxes a copy into `data/dataset_original.csv`.
2. **Data Pipeline & Structural Filter Module (`module_2_cleaning`)**: Enforces structural integrity by dropping null records and duplicates, then applies an 80th-percentile quantile threshold to isolate severe operational environments.
3. **Statistical Analysis Module (`module_3_analysis`)**: Utilizes vectorized arrays via **NumPy** and **SciPy** to calculate high-level descriptive and distribution statistics.
4. **Data Visualization Module (`module_4_visualization`)**: Generates 6 comprehensive static engineering plots (using Matplotlib/Seaborn) and 2 dynamic cumulative animations (using Plotly) to model real-time structural performance.

---

## III. Summary of Visual Assets

### Static Evaluation Figures
* **Static 1 (Correlation Heatmap):** Quantifies cross-variable dependencies between `Motor_Torque`, `RPM`, `Vibration`, and `Temperature` to uncover hidden mechanical couplings.
* **Static 2 (Linear Regression Plot):** Charts torque against rotational speed (`Motor_RPM`) with a fitted regression trendline to show drivetrain loading characteristics.
* **Static 3 (Violin Plot):** Fulfills the comparative analysis requirement by mapping torque density distributions across categorized smooth versus rough terrains.
* **Static 4 (Boxplot):** Isolates extreme transient torsional stress spikes out past the whiskers to identify operational anomalies and fatigue outliers.
* **Static 5 (Hexagonal JointPlot):** Models the two-dimensional density concentration of thermal-mechanical stress blending `Motor_Temperature` and `Motor_Torque`.
* **Static 6 (Time-Series Line Chart):** Illustrates raw, high-frequency temporal torque fluctuations over a localized sample space.

### Cumulative Animation Interfaces
* **Animation 1 (Cumulative Histogram):** Employs iterative data-frame row-accumulation to show the **distribution shift and growth** of torque across operational cycles without translating-line glitches.
* **Animation 2 (Moving Violin Plot):** Displays a dynamic, frame-by-frame structural shape analysis of data variance, tracking how load densities adapt over time under severe terrain exposure.

---

## IV. How to Run
Ensure you have Python 3.9+ installed on your local machine before executing the following commands:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/codekomori0-lgtm/EDS_TUPM-25-0703_Pampliega
   cd EDS_TUPM-25-0703_Pampliega