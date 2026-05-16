import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
import os
from scipy.stats import skew

class EDS_TorsionalPipeline:
    def __init__(self, raw_path):
        self.raw_path = raw_path
        self.df = None
        self.df_cleaned = None
        # Ensure GitHub structure folders exist
        for folder in ['data', 'outputs']:
            if not os.path.exists(folder): os.makedirs(folder)

    def module_1_ingestion(self):
        """Mandatory Module: Data Ingestion with Robust Error Handling."""
        try:
            self.df = pd.read_csv(self.raw_path)
            # Save original to the required folder structure
            self.df.to_csv('data/dataset_original.csv', index=False)
            print("[1/4] Data Ingestion Successful.")
        except FileNotFoundError:
            print("Error: The raw data file was not found.")
        except Exception as e:
            print(f"Ingestion Error: {e}")

    def module_2_cleaning(self):
        """Module 2: Automated Cleaning & Severe Terrain Threshold Filtering."""
        try:
            self.df.dropna(inplace=True)
            self.df.drop_duplicates(inplace=True)
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
            
            # UNIQUE FILTER SWITCH: Top 20% harshest road conditions
            roughness_threshold = self.df['Route_Roughness'].quantile(0.80)
            self.df_cleaned = self.df[self.df['Route_Roughness'] >= roughness_threshold].copy()
            
            # Essential: Sort by timestamp so your cumulative animations still grow chronologically
            self.df_cleaned.sort_values(by='Timestamp', inplace=True)
            
            self.df_cleaned.to_csv('data/dataset_cleaned.csv', index=False)
            print(f"[2/4] Severe Terrain Filtering Complete (Threshold: {roughness_threshold:.2f}).")
        except Exception as e:
            print(f"Cleaning Error: {e}")

    def module_3_analysis(self):
        """Mandatory Module: Engineering Data Analytics using NumPy."""
        try:
            # Use NumPy for mandatory metrics [cite: 448]
            torque = self.df_cleaned['Motor_Torque'].values
            
            results = {
                "Mean": np.mean(torque),
                "Median": np.median(torque),
                "StdDev": np.std(torque),
                "Variance": np.var(torque),
                "Skewness": skew(torque)
            }
            
            # Comparative Analysis: High vs Low RPM Torque Performance [cite: 458]
            mid_rpm = np.median(self.df_cleaned['Motor_RPM'])
            high_rpm_data = self.df_cleaned[self.df_cleaned['Motor_RPM'] > mid_rpm]['Motor_Torque']
            low_rpm_data = self.df_cleaned[self.df_cleaned['Motor_RPM'] <= mid_rpm]['Motor_Torque']
            
            print("\n--- Engineering Metrics ---")
            for key, val in results.items():
                print(f"{key}: {val:.4f}")
            print(f"Comparative Mean (High RPM): {np.mean(high_rpm_data):.4f}")
            print(f"Comparative Mean (Low RPM): {np.mean(low_rpm_data):.4f}")
            
            return results
        except Exception as e:
            print(f"Analysis Error: {e}")

    def module_4_visualization(self):
        """Module 4: 6 Static and 2 Cumulative Animated Graphs [cite: 63-64]."""
        try:
            sns.set_theme(style="whitegrid")
            
            # --- 6 STATIC PLOTS ---
            # 1. Heatmap: Correlation Analysis [cite: 59]
            plt.figure(figsize=(8,6))
            corr = self.df_cleaned[['Motor_Torque', 'Motor_RPM', 'Motor_Vibration', 'Motor_Temperature']].corr()
            sns.heatmap(corr, annot=True, cmap='RdBu_r', fmt=".2f")
            plt.title('Mechanical Parameter Correlation Matrix')
            plt.savefig('outputs/static_1_heatmap.png')

            # 2. Scatter with Regression: Torque vs RPM
            plt.figure(figsize=(8,5))
            sns.regplot(x='Motor_RPM', y='Motor_Torque', data=self.df_cleaned.sample(500), 
                        scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
            plt.title('Torsional Load vs. Rotational Speed Trend')
            plt.savefig('outputs/static_2_regplot.png')

            # 3. Violin Plot: Comparative Analysis (Road Conditions) [cite: 60]
            plt.figure(figsize=(8,5))
            self.df_cleaned['Road_Quality'] = pd.cut(self.df_cleaned['Route_Roughness'], bins=2, labels=['Smooth', 'Rough'])
            sns.violinplot(x='Road_Quality', y='Motor_Torque', data=self.df_cleaned)
            plt.title('Torque Variance: Smooth vs. Rough Terrain')
            plt.savefig('outputs/static_3_violin_comparison.png')

            # 4. Boxplot: Outlier Detection [cite: 58]
            plt.figure(figsize=(8,5))
            sns.boxplot(x=self.df_cleaned['Motor_Torque'], color='lightgreen')
            plt.title('Identification of Torsional Stress Outliers')
            plt.savefig('outputs/static_4_boxplot.png')

            # 5. JointPlot: Thermal-Mechanical stress density
            g = sns.jointplot(x='Motor_Temperature', y='Motor_Torque', data=self.df_cleaned, kind="hex")
            plt.savefig('outputs/static_5_jointplot.png')

            # 6. Time-Series: Static Trend
            plt.figure(figsize=(10,4))
            plt.plot(self.df_cleaned['Timestamp'].head(100), self.df_cleaned['Motor_Torque'].head(100))
            plt.title('Temporal Torque Fluctuations (Sample)')
            plt.savefig('outputs/static_6_timeseries.png')

            # --- 2 FIXED ANIMATIONS (Cumulative Growth Logic) ---
            sample = self.df_cleaned.head(100).copy()
            frames_data = []
            for i in range(1, len(sample) + 1):
                temp_df = sample.iloc[:i].copy()
                temp_df['frame'] = i
                frames_data.append(temp_df)
            animated_df = pd.concat(frames_data)

            # Animation 1: Growing Histogram (Distribution Shift)
            fig1 = px.histogram(animated_df, x="Motor_Torque", animation_frame="frame",
                                range_x=[0, self.df_cleaned['Motor_Torque'].max() + 20],
                                range_y=[0, 30], nbins=20, title="Cumulative Torsional Load Accumulation")
            fig1.write_html("outputs/animation_1_histogram.html")

            # Animation 2: Moving Violin (Dynamic Shape Analysis)
            fig2 = px.violin(animated_df, y="Motor_Torque", x="Road_Quality", animation_frame="frame",
                             color="Road_Quality", range_y=[0, self.df_cleaned['Motor_Torque'].max() + 50],
                             box=True, title="Dynamic Torque Shape Analysis")
            fig2.write_html("outputs/animation_2_violin.html")
        except Exception as e:
            print(f"Visualization Error: {e}")

# Run Pipeline
if __name__ == "__main__":
    # Point this to your original source file
    pipeline = EDS_TorsionalPipeline('data/data_dataset_original.csv')
    pipeline.module_1_ingestion()
    pipeline.module_2_cleaning()
    pipeline.module_3_analysis()
    pipeline.module_4_visualization()