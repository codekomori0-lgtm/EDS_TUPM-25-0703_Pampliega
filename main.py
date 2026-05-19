import os
import numpy as np
import pandas as pd
from scipy.stats import skew


class EDS_TorsionalPipeline:
    def __init__(self, filepath="data/data_dataset_original.csv"):
        # Explicitly locking the exact file name provided
        self.raw_filepath = filepath
        self.df = None
        self.df_cleaned = None
        self.metrics = {}
        
        # Ensure the sandboxed directory exists for output saves
        os.makedirs("data", exist_ok=True)

    # ==========================================================================
    # MODULE 1: DATA INGESTION (Gathering Stage)
    # ==========================================================================
    def module_1_ingestion(self):
        print("\n==================================================================")
        print(" [STAGE 1/3] >>> INITIALIZING DATA INGESTION (GATHERING DATA)    ")
        print("==================================================================")
        print(" [INFO] Starting the gathering stage...")
        print(f" -> Searching for exact file path: '{self.raw_filepath}'")
        
        try:
            if not os.path.exists(self.raw_filepath):
                raise FileNotFoundError(
                    f"The file '{self.raw_filepath}' could not be located in this folder.\n"
                    f"   Please verify that the file sits in the same directory as main.py\n"
                    f"   Current Active Terminal Directory: {os.getcwd()}"
                )
            
            # Read and ingest raw CSV dataset
            self.df = pd.read_csv(self.raw_filepath)
            
            print(" -> STATUS: Telemetry data successfully gathered from local directory.")
            print(f" -> RESULTS: Loaded {len(self.df)} structural rows into memory.")
            print(f" -> DETECTED CHANNELS: {list(self.df.columns[:6])} ...")
            print("==================================================================")
            return True
        except Exception as e:
            print(f" [CRITICAL INGESTION BREAKDOWN]: {e}")
            print("==================================================================")
            return False

    # ==========================================================================
    # MODULE 2: DATA PIPELINE & CLEANING (Refining Stage)
    # ==========================================================================
    def module_2_cleaning(self):
        print("\n==================================================================")
        print(" [STAGE 2/3] >>> INITIALIZING DATA PIPELINE (CLEANING & FILTERING)")
        print("==================================================================")
        print(" [INFO] Starting the cleaning and refining stage...")
        print(" -> Purging data anomalies and setting up unique filter bounds...")
        
        try:
            # Drop null values and duplicate records
            initial_count = len(self.df)
            self.df.dropna(inplace=True)
            self.df.drop_duplicates(inplace=True)
            purged = initial_count - len(self.df)
            print(f" -> CLEANING ENGINE: Dropped {purged} null/duplicate rows from data frame.")

            # Dynamic column mapping to handle variations cleanly
            col_mapping = {}
            for col in self.df.columns:
                c_low = col.lower()
                if 'torque' in c_low: col_mapping[col] = 'Motor_Torque'
                elif 'rpm' in c_low or 'speed' in c_low: col_mapping[col] = 'Motor_RPM'
                elif 'vib' in c_low: col_mapping[col] = 'Vibration_Amplitude'
                elif 'temp' in c_low: col_mapping[col] = 'Motor_Temperature'
                elif 'rough' in c_low or 'terrain' in c_low: col_mapping[col] = 'Route_Roughness'
                elif 'time' in c_low or 'date' in c_low: col_mapping[col] = 'Timestamp'
            
            if col_mapping:
                self.df.rename(columns=col_mapping, inplace=True)
                print(f" -> HEADER ENGINE: Unified internal channels to: {list(col_mapping.values())}")

            # Safe type conversion for numerical analysis
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
            for col in ['Motor_Torque', 'Motor_RPM', 'Vibration_Amplitude', 'Motor_Temperature', 'Route_Roughness']:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            # UNIQUE SEVERE TERRAIN FILTER LOGIC (80th Percentile)
            print(" -> FILTER ENGINE: Scanning profile array for unique boundary condition...")
            roughness_threshold = self.df['Route_Roughness'].quantile(0.80)
            print(f" -> UNIQUE THRESHOLD SLICE: Isolating Route_Roughness >= {roughness_threshold:.4f}")
            
            # Extract target rows satisfying the strict boundary condition
            self.df_cleaned = self.df[self.df['Route_Roughness'] >= roughness_threshold].copy()
            self.df_cleaned.sort_values(by='Timestamp', inplace=True)
            
            # Export refined dataset copy to data directory
            cleaned_path = 'data/dataset_cleaned.csv'
            self.df_cleaned.to_csv(cleaned_path, index=False)
            
            print(" -> STATUS: Unique severe terrain data isolated cleanly.")
            print(f" -> RESULTS: Pipeline trimmed domain from {len(self.df)} down to {len(self.df_cleaned)} rows.")
            print(f" -> EXPORT SECURE: Cached filtered data matrix to '{cleaned_path}'.")
            print("==================================================================")
            return True
        except Exception as e:
            print(f" [CRITICAL PIPELINE CLEANING BREAKDOWN]: {e}")
            print("==================================================================")
            return False

    # ==========================================================================
    # MODULE 3: STATISTICAL ANALYSIS (Interpretation Stage)
    # ==========================================================================
    def module_3_analysis(self):
        print("\n==================================================================")
        print(" [STAGE 3/3] >>> INITIALIZING STATISTICAL MATH (INTERPRETATION)  ")
        print("==================================================================")
        print(" [INFO] Starting the interpretation stage...")
        print(" -> Loading vectorized C-arrays into low-level math modules...")
        
        try:
            # Feed data array directly into vectorized processors
            torque_array = self.df_cleaned['Motor_Torque'].to_numpy()
            
            # High-performance NumPy and SciPy operations
            self.metrics['Mean'] = np.mean(torque_array)
            self.metrics['Median'] = np.median(torque_array)
            self.metrics['Std_Dev'] = np.std(torque_array)
            self.metrics['Variance'] = np.var(torque_array)
            self.metrics['Skewness'] = skew(torque_array)
            
            print(" -> STATUS: Vectorized array calculations completed successfully.")
            print(" -> CALCULATED MECHANICAL DRIVETRAIN PARAMETERS:")
            print("    --------------------------------------------------")
            print(f"    * Mean Motor Torque       (μ_T) : {self.metrics['Mean']:.4f} Nm")
            print(f"    * Median Motor Torque     (M_T) : {self.metrics['Median']:.4f} Nm")
            print(f"    * Standard Deviation      (σ_T) : {self.metrics['Std_Dev']:.4f} Nm")
            print(f"    * System Sample Variance  (σ_T²): {self.metrics['Variance']:.4f} Nm²")
            print(f"    * Asymmetric Load Skewness(S_T) : {self.metrics['Skewness']:.4f}")
            print("    --------------------------------------------------")
            print(" [INFO] Mathematical interpretation complete. Metrics ready for research report.")
            print("==================================================================")
            return True
        except Exception as e:
            print(f" [CRITICAL MATHEMATICAL INTERPRETATION BREAKDOWN]: {e}")
            print("==================================================================")
            return False

    # ==========================================================================
    # WORKFLOW ORCHESTRATION CONTROL
    # ==========================================================================
    def run_pipeline(self):
        print("==================================================================")
        print("     STARTING STRUCTURAL DATA SYSTEMS PIPELINE APPLICATION        ")
        print("==================================================================")
        
        # Continuous execution chain through stages
        if self.module_1_ingestion():
            if self.module_2_cleaning():
                if self.module_3_analysis():
                    print("\n==================================================================")
                    print(" [SUCCESS] >>> DATA GATHERED, CLEANED, AND INTERPRETED COMPLETELY ")
                    print("==================================================================")
                    return


if __name__ == "__main__":
    # Locking down the exact filename configuration
    pipeline = EDS_TorsionalPipeline(filepath="data_dataset_original.csv")
    pipeline.run_pipeline()