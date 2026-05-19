# Engineering Data Systems Pipeline: SOL-04
**Project Title:** Automated Torsional Rigidity Analysis of EV Drivetrain Shafts under Severe Terrain Conditions  
**Student Name:** Ram Anthony E. Pampliega  
**Course & Section:** Computer Programming (BSME - 1C)  
**Academic Year:** 2026  
**Engineering Pillar:** Pillar 3 - Solid Mechanics & Stress Analysis  
**Institution:** Department of Mechanical Engineering, Technological University of the Philippines – Manila  

---

## I. Project Overview
This project implements a robust, automated Python-based data pipeline designed to analyze torsional rigidity and fatigue-loading profiles in Electric Vehicle (EV) drivetrain shafts. 

Rather than analyzing smoothed, generic operational averages, this software utilizes a **Unique Dual-Stage Mathematical Filter** to dynamically isolate extreme operational boundary conditions from large-scale predictive maintenance telemetry. By applying strict engineering-driven thresholds, the system calculates critical mechanical stress metrics and tracks cumulative system degradation exclusively under severe physical stress.

---

## II. The Unique Dual-Stage Structural Filter
The core analytical strength of this pipeline lies in its bespoke data conditioning engine, which mathematically isolates drivetrain stress using two distinct parameters:

1. **The IQR Anomaly Detection Matrix:** The system automatically calculates the Interquartile Range (IQR) of the `Motor_Torque` channel to establish rigid upper and lower operational bounds. This identifies and isolates transient torsional shock spikes (anomalies) that fall outside standard mechanical tolerances.
2. **The 80th-Percentile Terrain Stratification:** To evaluate the drivetrain purely under extreme conditions, the pipeline dynamically calculates the 80th-percentile floor of the `Route_Roughness` sensor data. The database is cleanly severed—filtering out normal, smooth-terrain driving entirely—leaving only the top 20% most severe environmental data to run through the statistical interpretation modules.

---

## III. System Architecture
The software is developed utilizing strict **Object-Oriented Programming (OOP)** patterns to ensure modularity, loose coupling, and enterprise scalability. The class `EDS_TorsionalPipeline` contains four distinct execution stages:

1. **Stage 1: Data Ingestion (`module_1_ingestion`)**: Handles raw CSV telemetry loading using secure `try-except` blocks to eliminate runtime crashes and sandboxes a protective memory matrix.
2. **Stage 2: Data Pipeline & Cleaning (`module_2_cleaning`)**: Enforces structural integrity by dropping null records and duplicates, standardizing data types, and executing the dual-stage IQR and 80th-percentile filtering logic.
3. **Stage 3: Statistical Analysis (`module_3_analysis`)**: Utilizes vectorized arrays via **NumPy** and **SciPy** to calculate high-level descriptive mathematical moments and derivative engineering physics (e.g., fatigue damage, mechanical power conversion, torsional shock).
4. **Stage 4: Data Synthesis & Visualization (`module_4_plotting_engine`)**: Generates manuscript-ready data tables, 3D response surfaces, and uses a native JavaScript writer to compile zero-dependency, interactive HTML5 video dashboards (bypassing FFmpeg).

---

## IV. Summary of Visual Assets

The Module 4 compilation engine automatically exports a highly technical, 7-part graphics portfolio directly to the `/outputs/` directory:

### High-Resolution Static Figures (2D & 3D)
* **Figure 1 (Torsional Load PDF):** A probability density function mapping the statistical asymmetry and concentration of physical torque loads.
* **Figure 2 (3D Trisurf Mesh):** A Delaunay triangulated geometric landscape mapping raw, non-linear fatigue allocations directly across speed and torque coordinates.
* **Figure 3 (Speed Decoupling Curve):** A scatter matrix paired with an active regression line mapping the electro-mechanical operating envelope.
* **Figure 4 (3D Fatigue Response Surface):** A mathematically smoothed 3D mesh evaluating cumulative Palmgren-Miner fatigue damage against vibration and torque.
* **Figure 5 (3D Power Topology Map):** A multi-variable contour surface calculating mechanical power dissipation (kW) across the entire operational RPM limit.

### Interactive HTML5 Dashboards (Native JS Playback)
* **Figure 6 (Multi-Channel Stripchart):** A fully interactive, decelerated dual-axis simulation tracking live torque spikes alongside casing acceleration (g-force) over a downsampled evaluation index.
* **Figure 7 (Hysteresis Envelope Loop):** An interactive phase-space tracker utilizing an accumulating plot line to visualize the precise operational lag and physical hysteresis of the drivetrain under severe terrain transitions.

---

## V. How to Run
Ensure you have Python 3.9+ installed on your local machine before executing the following commands. No external video encoders are required.

**1. Clone the Repository:**
```bash
git clone [https://github.com/codekomori0-lgtm/EDS_TUPM-25-0703_Pampliega](https://github.com/codekomori0-lgtm/EDS_TUPM-25-0703_Pampliega)
cd EDS_TUPM-25-0703_Pampliega