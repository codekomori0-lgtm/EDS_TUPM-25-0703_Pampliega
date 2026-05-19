import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import skew


class EDS_TorsionalPipeline:
    def __init__(self, filepath="data/dataset_original.csv"):
        self.raw_filepath = filepath
        self.df = None
        self.df_normal_terrain = None
        self.df_cleaned = None
        
        # Comprehensive Data Registries for Publication
        self.conditioning_metrics = {}
        self.torque_moments = {}
        self.comparative_metrics = {}
        self.advanced_analytics = {}
        
        # ANSI Terminal Color Configuration
        self.C_CYAN = "\033[36m"
        self.C_GREEN = "\033[32m"
        self.C_YELLOW = "\033[33m"
        self.C_RED = "\033[31m"
        self.C_BOLD = "\033[1m"
        self.C_RESET = "\033[0m"
        
        # Ensure targeted directories exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

    # ==========================================================================
    # MODULE 1: DATA INGESTION (Gathering Stage)
    # ==========================================================================
    def module_1_ingestion(self):
        print(f"\n{self.C_BOLD}{self.C_CYAN}┌────────────────────────────────────────────────────────────────┐")
        print("│  [STAGE 1/4] >>> INITIALIZING DATA INGESTION (Gathering)       │")
        print(f"└────────────────────────────────────────────────────────────────┘{self.C_RESET}")
        print(f" {self.C_BOLD}[INFO]{self.C_RESET} Spawning secure telemetry database link...")
        
        try:
            if not os.path.exists(self.raw_filepath):
                raise FileNotFoundError(f"The raw file '{self.raw_filepath}' could not be located.")
            
            self.df = pd.read_csv(self.raw_filepath)
            print(f"\n {self.C_GREEN}STATUS: Telemetry core matrix mapped successfully into system memory.{self.C_RESET}")
            print(f"  └─ Total Logged Database Volume : {len(self.df):,} rows Ingested")
            print(f"{self.C_CYAN}──────────────────────────────────────────────────────────────────{self.C_RESET}")
            return True
        except Exception as e:
            print(f"\n {self.C_RED}CRITICAL STAGE 1 INGESTION FAULT:{self.C_RESET} {e}")
            return False

    # ==========================================================================
    # MODULE 2: DATA PIPELINE & CLEANING (Refining Stage)
    # ==========================================================================
    def module_2_cleaning(self):
        print(f"\n{self.C_BOLD}{self.C_CYAN}┌────────────────────────────────────────────────────────────────┐")
        print("│  [STAGE 2/4] >>> EXECUTING DATA PIPELINE & CLEANING (Refining) │")
        print(f"└────────────────────────────────────────────────────────────────┘{self.C_RESET}")
        print(f" {self.C_BOLD}[INFO]{self.C_RESET} Launching multi-channel structural scanning vectors...")
        
        try:
            initial_count = len(self.df)
            self.df.dropna(inplace=True)
            self.df.drop_duplicates(inplace=True)

            col_mapping = {
                'Timestamp': 'Timestamp', 'Motor_Torque': 'Motor_Torque',
                'Motor_RPM': 'Motor_RPM', 'Motor_Vibration': 'Vibration_Amplitude',
                'Motor_Temperature': 'Motor_Temperature', 'Route_Roughness': 'Route_Roughness'
            }
            active_mapping = {k: v for k, v in col_mapping.items() if k in self.df.columns}
            self.df.rename(columns=active_mapping, inplace=True)

            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
            for col in ['Motor_Torque', 'Motor_RPM', 'Vibration_Amplitude', 'Motor_Temperature', 'Route_Roughness']:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            q1 = self.df['Motor_Torque'].quantile(0.25)
            q3 = self.df['Motor_Torque'].quantile(0.75)
            iqr = q3 - iqr if 'iqr' in locals() else (q3 - q1)
            upper_bound = q3 + 1.5 * iqr
            lower_bound = q1 - 1.5 * iqr
            outliers_count = len(self.df[(self.df['Motor_Torque'] > upper_bound) | (self.df['Motor_Torque'] < lower_bound)])
            
            roughness_threshold = self.df['Route_Roughness'].quantile(0.80)
            
            self.df_normal_terrain = self.df[self.df['Route_Roughness'] < roughness_threshold].copy()
            self.df_cleaned = self.df[self.df['Route_Roughness'] >= roughness_threshold].copy()
            self.df_cleaned.sort_values(by='Timestamp', inplace=True)
            
            self.conditioning_metrics['Raw_Count'] = initial_count
            self.conditioning_metrics['Threshold_Floor'] = roughness_threshold
            self.conditioning_metrics['Filtered_Count'] = len(self.df_cleaned)
            
            self.torque_moments['Q1'] = q1
            self.torque_moments['Q3'] = q3
            self.torque_moments['Lower_Bound'] = lower_bound
            self.torque_moments['Upper_Bound'] = upper_bound
            self.torque_moments['Outliers_Count'] = outliers_count

            self.comparative_metrics['Normal_Count'] = len(self.df_normal_terrain)
            self.comparative_metrics['Severe_Count'] = len(self.df_cleaned)
            self.comparative_metrics['Normal_Mean_Torque'] = self.df_normal_terrain['Motor_Torque'].mean()
            self.comparative_metrics['Severe_Mean_Torque'] = self.df_cleaned['Motor_Torque'].mean()
            self.comparative_metrics['Normal_Mean_Vib'] = self.df_normal_terrain['Vibration_Amplitude'].mean()
            self.comparative_metrics['Severe_Mean_Vib'] = self.df_cleaned['Vibration_Amplitude'].mean()

            cleaned_path = 'data/dataset_cleaned.csv'
            self.df_cleaned.to_csv(cleaned_path, index=False)
            
            print(f"\n {self.C_BOLD}{self.C_YELLOW}┌──────────────────────────────────────────────────────────────┐")
            print(" │       DISTRIBUTION SCANS: STATISTICAL OUTLIER DETECTION      │")
            print(" ├──────────────────────────────────────────────────────────────┤")
            print(f" │  Evaluation Standard : Interquartile Range (IQR Rule)        │")
            print(f" │  Lower Stress Boundary: {lower_bound:<12.2f} Nm                      │")
            print(f" │  Upper Stress Boundary: {upper_bound:<12.2f} Nm                      │")
            print(f" │  Flagged Anomalies    : {outliers_count:<12,} rows                    │")
            print(" └──────────────────────────────────────────────────────────────┘")
            
            print(f"\n {self.C_GREEN}STATUS: Data conditioning and matrix stratification complete.{self.C_RESET}")
            print(f"{self.C_CYAN}──────────────────────────────────────────────────────────────────{self.C_RESET}")
            return True
        except Exception as e:
            print(f"\n {self.C_RED}CRITICAL STAGE 2 FILTERING FAULT:{self.C_RESET} {e}")
            return False
        
    # ==========================================================================
    # MODULE 3: STATISTICAL ANALYSIS & DERIVATIVE PARAMETERS (Interpretation Stage)
    # ==========================================================================
    def module_3_analysis(self):
        print(f"\n{self.C_BOLD}{self.C_CYAN}┌────────────────────────────────────────────────────────────────┐")
        print("│  [STAGE 3/4] >>> RUNNING STATISTICAL INTERPRETATION (Analysis) │")
        print(f"└────────────────────────────────────────────────────────────────┘{self.C_RESET}")
        print(f" {self.C_BOLD}[INFO]{self.C_RESET} Computing high-differentiability mathematical moments...")
        
        try:
            torque_array = self.df_cleaned['Motor_Torque'].to_numpy()
            
            self.torque_moments['Mean'] = np.mean(torque_array)
            self.torque_moments['Median'] = np.median(torque_array)
            self.torque_moments['Std_Dev'] = np.std(torque_array)
            self.torque_moments['Variance'] = np.var(torque_array)
            self.torque_moments['Skewness'] = skew(torque_array)
            
            self.comparative_metrics['RPM_Corr'] = self.df_cleaned['Motor_RPM'].corr(self.df_cleaned['Motor_Torque'])
            self.comparative_metrics['Temp_Corr'] = self.df_cleaned['Motor_Temperature'].corr(self.df_cleaned['Motor_Torque'])
            
            self.df_cleaned['Fatigue_Damage'] = (self.df_cleaned['Motor_Torque'] / 400.0) ** 3
            self.advanced_analytics['Total_Fatigue_Sum'] = self.df_cleaned['Fatigue_Damage'].sum()
            
            self.df_cleaned['Mechanical_Power_kW'] = (2 * np.pi * self.df_cleaned['Motor_Torque'] * self.df_cleaned['Motor_RPM']) / 60000.0
            self.advanced_analytics['Max_Power_kW'] = self.df_cleaned['Mechanical_Power_kW'].max()
            self.advanced_analytics['Mean_Power_kW'] = self.df_cleaned['Mechanical_Power_kW'].mean()
            
            self.df_cleaned['Torsional_Shock_Factor'] = np.abs(self.df_cleaned['Motor_Torque'].diff().fillna(0))
            self.advanced_analytics['Max_Shock_Factor'] = self.df_cleaned['Torsional_Shock_Factor'].max()

            print(f"\n {self.C_BOLD}{self.C_YELLOW}┌──────────────────────────────────────────────────────────────┐")
            print(" │      CALCULATED VECTOR MOMENTS (ISOLATED SEVERE STRATA)      │")
            print(" ├──────────────────────────────────────────────────────────────┤")
            print(f" │  Mean Load Value    (mu_T)     :  {self.torque_moments['Mean']:<12.4f} Nm            │")
            print(f" │  Median Load Value  (M_T)      :  {self.torque_moments['Median']:<12.4f} Nm            │")
            print(f" │  Standard Deviation (sigma_T)  :  {self.torque_moments['Std_Dev']:<12.4f} Nm            │")
            print(f" │  Sample Variance    (sigma_T²) :  {self.torque_moments['Variance']:<12.4f} Nm²           │")
            print(f" │  Distribution Skew  (S_T)      :  {self.torque_moments['Skewness']:<12.4f}               │")
            print(" └──────────────────────────────────────────────────────────────┘")

            print(f"\n {self.C_RESET}{self.C_GREEN}STATUS: Vector mathematical interpretations completed smoothly.{self.C_RESET}")
            print(f"{self.C_CYAN}──────────────────────────────────────────────────────────────────{self.C_RESET}")
            return True
        except Exception as e:
            print(f"\n {self.C_RED}CRITICAL STAGE 3 MATHEMATICAL FAULT:{self.C_RESET} {e}")
            return False

    # ==========================================================================
    # MODULE 4: DATA SYNTHESIS & VISUALIZATION (Compilation Stage)
    # ==========================================================================
    def module_4_plotting_engine(self):
        print(f"\n{self.C_BOLD}{self.C_CYAN}┌────────────────────────────────────────────────────────────────┐")
        print("│  [STAGE 4/4] >>> DATA SYNTHESIS & VISUALIZATION (Compilation)  │")
        print(f"└────────────────────────────────────────────────────────────────┘{self.C_RESET}")
        
        try:
            # --- MANUSCRIPT TEXT TABLES I, II, III ---
            print(f"\n{self.C_BOLD}TABLE I: MATHEMATICAL STRATIFICATION AND FILTERING BOUNDARIES{self.C_RESET}")
            print(f"{self.C_BOLD}{self.C_YELLOW}┌───────────┬──────────────────────────────────────────────┬──────────────┬──────────────┐")
            print("│ Result ID │ Operational Metric                           │ Value        │ Unit         │")
            print("├───────────┼──────────────────────────────────────────────┼──────────────┼──────────────┤")
            print(f"│ Result 1  │ Raw Telemetry Database Size (N_raw)          │ {self.conditioning_metrics['Raw_Count']:<12,} │ data points  │")
            print(f"│ Result 2  │ Computed Roughness Floor Threshold (Th_rough)│ {self.conditioning_metrics['Threshold_Floor']:<12.4f} │ --           │")
            print(f"│ Result 3  │ Filtered Severe Terrain Domain (N_filtered)  │ {self.conditioning_metrics['Filtered_Count']:<12,} │ data points  │")
            print("└───────────┴──────────────────────────────────────────────┴──────────────┴──────────────┘")

            print(f"\n{self.C_RESET}{self.C_BOLD}TABLE II: CALCULATED STATISTICAL MOMENTS AND ENVELOPE FOR MOTOR TORQUE{self.C_RESET}")
            print(f"{self.C_BOLD}{self.C_YELLOW}┌───────────┬──────────────────────────────────────────────┬──────────────┬──────────────┐")
            print("│ Result ID │ Parameter Metric                             │ Computed     │ Unit         │")
            print("├───────────┼──────────────────────────────────────────────┼──────────────┼──────────────┤")
            print(f"│ Result 4  │ Arithmetic Mean Torque (mu_T)                │ {self.torque_moments['Mean']:<12.4f} │ Nm           │")
            print(f"│ Result 5  │ Operational Median Torque (M_T)              │ {self.torque_moments['Median']:<12.4f} │ Nm           │")
            print(f"│ Result 6  │ Standard Deviation (sigma_T)                 │ {self.torque_moments['Std_Dev']:<12.4f} │ Nm           │")
            print(f"│ Result 7  │ Mechanical Sample Variance (sigma_T^2)       │ {self.torque_moments['Variance']:<12.4f} │ Nm^2         │")
            print(f"│ Result 8  │ Boundary Minimum (T_min)                     │ {self.df_cleaned['Motor_Torque'].min():<12.4f} │ Nm           │")
            print(f"│ Result 9  │ Transient Peak Maximum (T_max)               │ {self.df_cleaned['Motor_Torque'].max():<12.4f} │ Nm           │")
            print(f"│ Result 10 │ Fisher-Pearson Skewness Coefficient (S_T)    │ {self.torque_moments['Skewness']:<12.4f} │ --           │")
            print(f"│ Result 11 │ Calculated Upper Outlier Ceiling (IQR Stand) │ {self.torque_moments['Upper_Bound']:<12.4f} │ Nm           │")
            print(f"│ Result 12 │ Outlier Anomalies Flagged across Dataset     │ {self.torque_moments['Outliers_Count']:<12,} │ rows         │")
            print("└───────────┴──────────────────────────────────────────────┴──────────────┴──────────────┘")

            print(f"\n{self.C_RESET}{self.C_BOLD}TABLE III: MULTI-DOMAIN COMPARATIVE SYSTEM MATRIX{self.C_RESET}")
            print(f"{self.C_BOLD}{self.C_YELLOW}┌──────────────────────────────────────────────┬──────────────────────────────┬──────────────────────────────┐")
            print("│ Metric Category                              │ Group A: Normal Terrain      │ Group B: Severe Terrain      │")
            print("├──────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────┤")
            print(f"│ Result 13: Sample Data Point Volume          │ {self.comparative_metrics['Normal_Count']:<28,} │ {self.comparative_metrics['Severe_Count']:<28,} │")
            print(f"│ Result 14: Mean Mechanical Torque Load       │ {self.comparative_metrics['Normal_Mean_Torque']:<25.4f} Nm │ {self.comparative_metrics['Severe_Mean_Torque']:<25.4f} Nm │")
            print(f"│ Result 15: Mean High-Frequency Vibration     │ {self.comparative_metrics['Normal_Mean_Vib']:<26.4f}g  │ {self.comparative_metrics['Severe_Mean_Vib']:<26.4f}g  │")
            print("└──────────────────────────────────────────────┴──────────────────────────────┴──────────────────────────────┘")

            # --- PLOTTING LOGS ---
            print(f"\n {self.C_BOLD}[INFO]{self.C_RESET} Exporting assets into the target 'outputs/' directory...")
            
            # Slice downsampled subset for high-speed calculation matrix
            sample_df = self.df_cleaned.sample(min(2000, len(self.df_cleaned)), random_state=42).copy()
            sample_df.sort_values(by='Motor_RPM', inplace=True)
            
            # Figure 1: Static Torsional PDF
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.histplot(data=self.df_cleaned, x='Motor_Torque', kde=True, color='#1f77b4', stat='density', alpha=0.6, ax=ax)
            ax.axvline(self.torque_moments['Mean'], color='red', linestyle='--', linewidth=1.5, label=fr"Mean ($\mu_T$): {self.torque_moments['Mean']:.2f} Nm")
            ax.axvline(self.torque_moments['Median'], color='green', linestyle='-.', linewidth=1.5, label=fr"Median ($M_T$): {self.torque_moments['Median']:.2f} Nm")
            ax.set_title('Figure 1: Torsional Load PDF and Statistical Asymmetry Spectrum', fontweight='bold')
            ax.set_xlabel('Motor Torque (Nm)'), ax.set_ylabel('Probability Density'), ax.legend()
            plt.tight_layout()
            plt.savefig('outputs/static_figure_1_torque_pdf.png', dpi=300)
            plt.close()

            # Figure 2: 3D TRISURF LANDSCAPE MESH
            fig2_3d = plt.figure(figsize=(8, 6))
            ax2_3d = fig2_3d.add_subplot(111, projection='3d')
            
            # Delaunay triangulation over raw coordinates mapping non-linear fatigue allocations directly
            surf2 = ax2_3d.plot_trisurf(
                sample_df['Motor_RPM'], 
                sample_df['Motor_Torque'], 
                sample_df['Fatigue_Damage'], 
                cmap='plasma', 
                edgecolor='none', 
                alpha=0.85
            )
            fig2_3d.colorbar(surf2, ax=ax2_3d, shrink=0.5, aspect=10, label='Kinematic Fatigue Component Allocation')
            
            ax2_3d.set_title('Figure 2: 3D Triangulated Mesh of Non-Linear Fatigue Damage Profiles', fontweight='bold', pad=15)
            ax2_3d.set_xlabel('Angular Velocity (RPM)')
            ax2_3d.set_ylabel('Motor Torsional Torque (Nm)')
            ax2_3d.set_zlabel('Calculated Fatigue Index (%)')
            
            plt.tight_layout()
            plt.savefig('outputs/static_figure_2_vibration_3d_surface.png', dpi=300)
            plt.close()

            # Figure 3: Static Speed Decoupling Plot
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(data=sample_df, x='Motor_RPM', y='Motor_Torque', alpha=0.3, color='#2ca02c', edgecolor='none', ax=ax, label='Telemetry Samples')
            sns.regplot(data=self.df_cleaned, x='Motor_RPM', y='Motor_Torque', scatter=False, color='black', 
                        line_kws={'linewidth': 2, 'label': f"Linear Vector (r = {self.comparative_metrics['RPM_Corr']:.4f})"}, ax=ax)
            
            ax.set_title('Figure 3: Electro-Mechanical Operating Envelope\nSpeed Decoupling Curve', 
                         fontweight='bold', fontsize=11, pad=12)
            
            ax.set_xlabel('Motor Speed (RPM)')
            ax.set_ylabel('Motor Torque (Nm)')
            ax.legend()
            plt.tight_layout()
            plt.savefig('outputs/static_figure_3_speed_decoupling.png', dpi=300)
            plt.close()

            # Figure 4: Static 3D Fatigue Surface
            fig4 = plt.figure(figsize=(8, 6))
            ax4 = fig4.add_subplot(111, projection='3d')
            xi = np.linspace(sample_df['Motor_Torque'].min(), sample_df['Motor_Torque'].max(), 40)
            yi = np.linspace(sample_df['Vibration_Amplitude'].min(), sample_df['Vibration_Amplitude'].max(), 40)
            X, Y = np.meshgrid(xi, yi)
            from scipy.interpolate import griddata
            Z_fatigue = griddata((sample_df['Motor_Torque'], sample_df['Vibration_Amplitude']), 
                                 sample_df['Fatigue_Damage'], (X, Y), method='linear', fill_value=0)
            surf4 = ax4.plot_surface(X, Y, Z_fatigue, cmap='jet', edgecolor='none', alpha=0.9)
            fig4.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10, label='Fatigue Damage Component Allocation')
            ax4.set_title('Figure 4: Torsional and Vibrational Fatigue 3D Response Surface', fontweight='bold', pad=15)
            ax4.set_xlabel('Torsional Stress (Nm)'), ax4.set_ylabel('Vibration Amplitude (g)'), ax4.set_zlabel('System Fatigue Damage (%)')
            plt.tight_layout()
            plt.savefig('outputs/static_figure_4_3d_fatigue_surface.png', dpi=300)
            plt.close()

            # Figure 5: Static 3D Power Map
            fig5 = plt.figure(figsize=(8, 6))
            ax5 = fig5.add_subplot(111, projection='3d')
            x_rpm = np.linspace(sample_df['Motor_RPM'].min(), sample_df['Motor_RPM'].max(), 40)
            y_trq = np.linspace(sample_df['Motor_Torque'].min(), sample_df['Motor_Torque'].max(), 40)
            X_mesh, Y_mesh = np.meshgrid(x_rpm, y_trq)
            Z_power = griddata((sample_df['Motor_RPM'], sample_df['Motor_Torque']), 
                               sample_df['Mechanical_Power_kW'], (X_mesh, Y_mesh), method='linear', fill_value=0)
            surf5 = ax5.plot_surface(X_mesh, Y_mesh, Z_power, cmap='magma', edgecolor='none', alpha=0.9)
            fig5.colorbar(surf5, ax=ax5, shrink=0.5, aspect=10, label='Mechanical Output Power (kW)')
            ax5.set_title('Figure 5: Electro-Mechanical Power Dissipation Topology 3D Map', fontweight='bold', pad=15)
            ax5.set_xlabel('Angular Velocity (RPM)'), ax5.set_ylabel('Motor Torque (Nm)'), ax5.set_zlabel('Power Dissipation (kW)')
            plt.tight_layout()
            plt.savefig('outputs/static_figure_5_3d_power_surface.png', dpi=300)
            plt.close()

            # --- INTERACTIVE JAVASCRIPT PLAYER GENERATION ---
            print(f"  ├─ Building interactive HTML5 visualization dashboards (Bypassing FFmpeg)...")
            anim_df = self.df_cleaned.head(200).copy().reset_index(drop=True)
            step_interval = 2
            frames_to_render = len(anim_df) // step_interval
            
            # Figure 6: Interactive HTML Stripchart
            fig_anim1, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            fig_anim1.suptitle("Figure 6: Real-Time Multi-Channel Stripchart Simulation", fontweight='bold', fontsize=12)
            line1, = ax1.plot([], [], color='#1f77b4', lw=1.5, label="Torque (Nm)")
            line2, = ax2.plot([], [], color='#d62728', lw=1.5, label="Vibration (g)")
            ax1.legend(loc="upper left"), ax2.legend(loc="upper left")
            ax1.set_ylabel("Torque Force (Nm)"), ax2.set_ylabel("Acceleration (g)")
            ax2.set_xlabel("Downsampled Telemetry Evaluation Index")
            ax1.set_ylim(anim_df['Motor_Torque'].min() - 10, anim_df['Motor_Torque'].max() + 10)
            ax2.set_ylim(anim_df['Vibration_Amplitude'].min() - 0.2, anim_df['Vibration_Amplitude'].max() + 0.2)
            ax1.set_xlim(0, frames_to_render)
            ax2.set_xlim(0, frames_to_render)

            def update_stripchart(frame):
                actual_idx = frame * step_interval
                x_range = range(frame + 1)
                y1_data = anim_df['Motor_Torque'].iloc[:actual_idx+1:step_interval]
                y2_data = anim_df['Vibration_Amplitude'].iloc[:actual_idx+1:step_interval]
                safe_len = min(len(x_range), len(y1_data), len(y2_data))
                line1.set_data(x_range[:safe_len], y1_data[:safe_len])
                line2.set_data(x_range[:safe_len], y2_data[:safe_len])
                return line1, line2

            ani1 = animation.FuncAnimation(fig_anim1, update_stripchart, frames=frames_to_render, interval=100, blit=True)
            with open('outputs/animated_figure_6_telemetry_stripchart.html', 'w') as f:
                f.write(ani1.to_jshtml())
            plt.close(fig_anim1)

            # Figure 7: Interactive HTML Hysteresis Envelope Loop
            fig_anim2, ax_traj = plt.subplots(figsize=(7, 4.5))
            ax_traj.set_title("Figure 7: Electro-Mechanical Operating Envelope Hysteresis Loop", fontweight='bold')
            ax_traj.set_xlim(anim_df['Motor_RPM'].min() - 100, anim_df['Motor_RPM'].max() + 100)
            ax_traj.set_ylim(anim_df['Motor_Torque'].min() - 20, anim_df['Motor_Torque'].max() + 20)
            ax_traj.set_xlabel("Motor Operational Speed (RPM)")
            ax_traj.set_ylabel("Motor Torsional Torque Load (Nm)")
            line7, = ax_traj.plot([], [], '-', color='#1f77b4', alpha=0.7, lw=1.5, label='Hysteresis Path')
            dot7, = ax_traj.plot([], [], 'o', color='#d62728', ms=6, label='Active State Vector')
            ax_traj.legend(loc="upper left")

            def update_hysteresis(frame):
                line7.set_data(anim_df['Motor_RPM'].iloc[:frame+1], anim_df['Motor_Torque'].iloc[:frame+1])
                if frame < len(anim_df):
                    dot7.set_data([anim_df['Motor_RPM'].iloc[frame]], [anim_df['Motor_Torque'].iloc[frame]])
                return line7, dot7

            ani2 = animation.FuncAnimation(fig_anim2, update_hysteresis, frames=len(anim_df), interval=100, blit=True)
            with open('outputs/animated_figure_7_speed_trajectory.html', 'w') as f:
                f.write(ani2.to_jshtml())
            plt.close(fig_anim2)

            print(f"\n {self.C_GREEN}STATUS: High-resolution publication suite compiled cleanly inside directories.{self.C_RESET}")
            print(f"  └─ File Repository Location: '{os.getcwd()}/outputs/' (Figures 1 through 7 ready)")
            print(f"{self.C_CYAN}──────────────────────────────────────────────────────────────────{self.C_RESET}")
            return True
        except Exception as e:
            print(f"\n {self.C_RED}CRITICAL STAGE 4 VISUALIZATION FAULT:{self.C_RESET} {e}")
            return False

    # ==========================================================================
    # WORKFLOW ORCHESTRATION CONTROL
    # ==========================================================================
    def run_pipeline(self):
        print(f"{self.C_BOLD}{self.C_CYAN}==================================================================")
        print("     STARTING STRUCTURAL DATA SYSTEMS PIPELINE APPLICATION        ")
        print(f"=================================================================={self.C_RESET}")
        
        if self.module_1_ingestion():
            if self.module_2_cleaning():
                if self.module_3_analysis():
                    if self.module_4_plotting_engine():
                        print(f"\n{self.C_BOLD}{self.C_GREEN}==================================================================")
                        print(" [SUCCESS] >>> PIPELINE OPERATION COMPLETE: ASSETS EXPORTED TO /outputs/")
                        print(f"==================================================================\n{self.C_RESET}")
                        return


if __name__ == "__main__":
    pipeline = EDS_TorsionalPipeline(filepath="data/dataset_original.csv")
    pipeline.run_pipeline()