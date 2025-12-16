# ============================================
# GUI APPLICATION FOR 5-MODEL PREDICTIONS
# ============================================

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

class ModelPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cancer Stage Prediction - 5 Models")
        self.root.geometry("1200x900")
        
        # Load models and preprocessing info
        self.load_models()
        
        # Create GUI
        self.create_widgets()
        
    def load_models(self):
        """Load all saved models and preprocessing objects."""
        try:
            # Load models
            with open('saved_models/log_reg_model.pkl', 'rb') as f:
                self.log_reg = pickle.load(f)
            
            with open('saved_models/rf_model.pkl', 'rb') as f:
                self.rf = pickle.load(f)
            
            with open('saved_models/xgb_model.pkl', 'rb') as f:
                self.xgb_model = pickle.load(f)
            
            with open('saved_models/ann_model.pkl', 'rb') as f:
                self.ann_model = pickle.load(f)
            
            with open('saved_models/svm_model.pkl', 'rb') as f:
                self.svm_model = pickle.load(f)
            
            # Load preprocessing objects
            with open('saved_models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open('saved_models/preprocessing_info.pkl', 'rb') as f:
                self.preprocessing_info = pickle.load(f)
            
            with open('saved_models/gui_info.pkl', 'rb') as f:
                self.gui_info = pickle.load(f)
            
            print("Models loaded successfully!")
            
        except FileNotFoundError as e:
            messagebox.showerror("Error", 
                f"Could not find saved models. Please run the training script first.\n\nError: {str(e)}")
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading models: {str(e)}")
            self.root.destroy()
    
    def create_widgets(self):
        """Create the GUI widgets."""
        # Main container with scrollbar
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Cancer Stage Prediction System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input tab
        input_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(input_frame, text="Input Data")
        
        # Results tab
        results_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(results_frame, text="Predictions")
        
        self.create_input_tab(input_frame)
        self.create_results_tab(results_frame)
    
    def create_input_tab(self, parent):
        """Create the input tab with all feature fields."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Store input widgets
        self.input_widgets = {}
        
        # Demographic section
        demo_frame = ttk.LabelFrame(scrollable_frame, text="Demographic Features", padding="10")
        demo_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        row = 0
        self.create_input_field(demo_frame, "Current Age", "numeric", row)
        row += 1
        self.create_input_field(demo_frame, "Sex", "categorical", row, 
                               self.gui_info['categorical_options'].get('Sex', []))
        row += 1
        self.create_input_field(demo_frame, "Race", "categorical", row,
                               self.gui_info['categorical_options'].get('Race', []))
        row += 1
        self.create_input_field(demo_frame, "Ethnicity", "categorical", row,
                               self.gui_info['categorical_options'].get('Ethnicity', []))
        row += 1
        self.create_input_field(demo_frame, "Smoking History (NLP)", "categorical", row,
                               self.gui_info['categorical_options'].get('Smoking History (NLP)', []))
        row += 1
        self.create_input_field(demo_frame, "Number of Samples Per Patient", "numeric", row)
        
        # Clinical section
        clinical_frame = ttk.LabelFrame(scrollable_frame, text="Clinical Features", padding="10")
        clinical_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        row = 0
        self.create_input_field(clinical_frame, "Cancer Type", "categorical", row,
                               self.gui_info['categorical_options'].get('Cancer Type', []))
        row += 1
        self.create_input_field(clinical_frame, "Cancer Type Detailed", "categorical", row,
                               self.gui_info['categorical_options'].get('Cancer Type Detailed', []))
        row += 1
        self.create_input_field(clinical_frame, "Primary Tumor Site", "categorical", row,
                               self.gui_info['categorical_options'].get('Primary Tumor Site', []))
        
        # Tumor Site binary fields
        tumor_site_frame = ttk.LabelFrame(clinical_frame, text="Tumor Sites", padding="5")
        tumor_site_frame.grid(row=row+1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        tumor_sites = [
            "Tumor Site: Adrenal Glands (NLP)",
            "Tumor Site: Bone (NLP)",
            "Tumor Site: CNS/Brain (NLP)",
            "Tumor Site: Intra Abdominal",
            "Tumor Site: Liver (NLP)",
            "Tumor Site: Lung (NLP)",
            "Tumor Site: Lymph Node (NLP)",
            "Tumor Site: Pleura (NLP)",
            "Tumor Site: Reproductive Organs (NLP)",
            "Tumor Site: Other (NLP)"
        ]
        
        ts_row = 0
        for i, site in enumerate(tumor_sites):
            col = i % 2
            if i % 2 == 0:
                ts_row += 1
            options = self.gui_info['categorical_options'].get(site, ['Yes', 'No', 'Unknown'])
            self.create_input_field(tumor_site_frame, site, "categorical", ts_row-1, options, col=col)
        
        # Genomic section
        genomic_frame = ttk.LabelFrame(scrollable_frame, text="Genomic Features", padding="10")
        genomic_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        row = 0
        self.create_input_field(genomic_frame, "TMB (nonsynonymous)", "numeric", row)
        row += 1
        self.create_input_field(genomic_frame, "Mutation Count", "numeric", row)
        row += 1
        self.create_input_field(genomic_frame, "Fraction Genome Altered", "numeric", row)
        row += 1
        self.create_input_field(genomic_frame, "MSI Score", "numeric", row)
        row += 1
        self.create_input_field(genomic_frame, "MSI Type", "categorical", row,
                               self.gui_info['categorical_options'].get('MSI Type', []))
        row += 1
        self.create_input_field(genomic_frame, "Gene Panel", "categorical", row,
                               self.gui_info['categorical_options'].get('Gene Panel', []))
        row += 1
        self.create_input_field(genomic_frame, "Somatic Status", "categorical", row,
                               self.gui_info['categorical_options'].get('Somatic Status', []))
        row += 1
        self.create_input_field(genomic_frame, "Sample coverage", "numeric", row)
        row += 1
        self.create_input_field(genomic_frame, "Tumor Purity", "numeric", row)
        
        # Pack canvas and scrollbar
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Predict button
        predict_btn = ttk.Button(scrollable_frame, text="Predict with All 5 Models", 
                                command=self.predict_all_models)
        predict_btn.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Clear button
        clear_btn = ttk.Button(scrollable_frame, text="Clear All Fields", 
                              command=self.clear_all_fields)
        clear_btn.grid(row=4, column=0, columnspan=2, pady=5)
    
    def create_input_field(self, parent, field_name, field_type, row, options=None, col=0):
        """Create an input field (numeric entry or categorical combobox)."""
        label = ttk.Label(parent, text=field_name + ":", width=30, anchor="w")
        label.grid(row=row, column=col*2, sticky=tk.W, padx=5, pady=2)
        
        if field_type == "numeric":
            entry = ttk.Entry(parent, width=20)
            entry.grid(row=row, column=col*2+1, sticky=tk.W, padx=5, pady=2)
            self.input_widgets[field_name] = entry
        else:  # categorical
            if options:
                combo = ttk.Combobox(parent, width=17, values=options, state="readonly")
            else:
                combo = ttk.Combobox(parent, width=17, state="normal")
            combo.grid(row=row, column=col*2+1, sticky=tk.W, padx=5, pady=2)
            combo.set("")  # Default empty
            self.input_widgets[field_name] = combo
    
    def create_results_tab(self, parent):
        """Create the results tab to display predictions."""
        # Results text area
        self.results_text = scrolledtext.ScrolledText(parent, width=100, height=40, 
                                                      font=("Courier", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial message
        self.results_text.insert(tk.END, "Enter patient data in the 'Input Data' tab and click 'Predict with All 5 Models' to see predictions here.\n\n")
        self.results_text.config(state=tk.DISABLED)
    
    def clear_all_fields(self):
        """Clear all input fields."""
        for widget in self.input_widgets.values():
            if isinstance(widget, ttk.Entry):
                widget.delete(0, tk.END)
            elif isinstance(widget, ttk.Combobox):
                widget.set("")
    
    def get_input_data(self):
        """Collect input data from all fields."""
        data = {}
        for field_name, widget in self.input_widgets.items():
            if isinstance(widget, ttk.Entry):
                value = widget.get().strip()
                data[field_name] = value if value else None
            elif isinstance(widget, ttk.Combobox):
                value = widget.get().strip()
                data[field_name] = value if value else None
        return data
    
    def preprocess_input(self, input_data):
        """Preprocess input data the same way as training data."""
        # Create DataFrame with single row
        df_input = pd.DataFrame([input_data])
        
        # Apply missing value imputation
        num_cols = self.preprocessing_info['num_cols']
        cat_cols = self.preprocessing_info['cat_cols']
        tumor_site_cols = self.preprocessing_info['tumor_site_cols']
        medians = self.preprocessing_info['medians']
        
        # Fill numerical columns with median
        for col in num_cols:
            if col in df_input.columns:
                val = df_input[col].iloc[0]
                if pd.isna(val) or val == "" or val is None:
                    df_input[col] = medians.get(col, 0)
                else:
                    try:
                        df_input[col] = pd.to_numeric(df_input[col])
                    except:
                        df_input[col] = medians.get(col, 0)
            else:
                # Column not in input, use median
                df_input[col] = medians.get(col, 0)
        
        # Fill categorical columns with "Unknown"
        for col in cat_cols:
            if col in df_input.columns:
                val = df_input[col].iloc[0]
                if pd.isna(val) or val == "" or val is None:
                    df_input[col] = "Unknown"
            else:
                df_input[col] = "Unknown"
        
        # Fill tumor site columns
        for col in tumor_site_cols:
            if col in df_input.columns:
                val = df_input[col].iloc[0]
                if pd.isna(val) or val == "" or val is None:
                    df_input[col] = "Unknown"
            else:
                df_input[col] = "Unknown"
        
        # Apply log1p transformation to skewed variables (before one-hot encoding)
        for col in ['TMB (nonsynonymous)', 'Mutation Count']:
            if col in df_input.columns:
                df_input[col] = np.log1p(df_input[col])
        
        # Encode tumor site binary columns (before one-hot encoding)
        for col in tumor_site_cols:
            if col in df_input.columns:
                df_input[col] = df_input[col].map({'Yes': 1, 'No': 0, 'Unknown': 0})
                # Fill any NaN that might result from unmapped values
                df_input[col] = df_input[col].fillna(0)
        
        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df_input, columns=cat_cols, drop_first=False)
        
        # Add engineered features (after log transformation)
        if 'TMB (nonsynonymous)' in df_encoded.columns:
            df_encoded['TMB_high'] = (df_encoded['TMB (nonsynonymous)'] > 10).astype(int)
        if 'Fraction Genome Altered' in df_encoded.columns:
            df_encoded['FGA_high'] = (df_encoded['Fraction Genome Altered'] > 0.2).astype(int)
        if 'MSI Score' in df_encoded.columns:
            df_encoded['MSI_high'] = (df_encoded['MSI Score'] > 10).astype(int)
        if 'Tumor Purity' in df_encoded.columns:
            df_encoded['Purity_high'] = (df_encoded['Tumor Purity'] > 0.5).astype(int)
        
        # Ensure all feature columns from training are present
        feature_columns = self.preprocessing_info['feature_columns']
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reorder columns to match training data exactly
        # Only keep columns that exist in feature_columns
        missing_cols = [col for col in feature_columns if col not in df_encoded.columns]
        for col in missing_cols:
            df_encoded[col] = 0
        
        df_encoded = df_encoded[feature_columns]
        
        return df_encoded
    
    def predict_all_models(self):
        """Run predictions through all 5 models and display results."""
        try:
            # Get input data
            input_data = self.get_input_data()
            
            # Preprocess
            X_input = self.preprocess_input(input_data)
            
            # Scale for models that need it
            X_input_scaled = self.scaler.transform(X_input)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            # Logistic Regression
            pred_log = self.log_reg.predict(X_input_scaled)[0]
            prob_log = self.log_reg.predict_proba(X_input_scaled)[0]
            predictions['Logistic Regression'] = pred_log
            probabilities['Logistic Regression'] = prob_log
            
            # Random Forest
            pred_rf = self.rf.predict(X_input)[0]
            prob_rf = self.rf.predict_proba(X_input)[0]
            predictions['Random Forest'] = pred_rf
            probabilities['Random Forest'] = prob_rf
            
            # XGBoost
            pred_xgb = self.xgb_model.predict(X_input)[0]
            prob_xgb = self.xgb_model.predict_proba(X_input)[0]
            predictions['XGBoost'] = pred_xgb
            probabilities['XGBoost'] = prob_xgb
            
            # ANN
            pred_ann = self.ann_model.predict(X_input_scaled)[0]
            prob_ann = self.ann_model.predict_proba(X_input_scaled)[0]
            predictions['ANN'] = pred_ann
            probabilities['ANN'] = prob_ann
            
            # SVM
            pred_svm = self.svm_model.predict(X_input_scaled)[0]
            prob_svm = self.svm_model.predict_proba(X_input_scaled)[0]
            predictions['SVM'] = pred_svm
            probabilities['SVM'] = prob_svm
            
            # Display results
            self.display_results(predictions, probabilities)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", 
                f"An error occurred during prediction:\n\n{str(e)}\n\nPlease check your input data.")
            import traceback
            traceback.print_exc()
    
    def display_results(self, predictions, probabilities):
        """Display prediction results in the results tab."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Header
        self.results_text.insert(tk.END, "=" * 80 + "\n")
        self.results_text.insert(tk.END, "CANCER STAGE PREDICTION RESULTS - ALL 5 MODELS\n")
        self.results_text.insert(tk.END, "=" * 80 + "\n\n")
        
        # Model predictions
        stage_labels = {0: "Stage 1-3", 1: "Stage 4"}
        
        for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'ANN', 'SVM']:
            pred = predictions[model_name]
            prob = probabilities[model_name]
            
            self.results_text.insert(tk.END, f"{model_name}:\n")
            self.results_text.insert(tk.END, f"  Prediction: {stage_labels[pred]}\n")
            self.results_text.insert(tk.END, f"  Confidence:\n")
            self.results_text.insert(tk.END, f"    Stage 1-3: {prob[0]*100:.2f}%\n")
            self.results_text.insert(tk.END, f"    Stage 4:   {prob[1]*100:.2f}%\n")
            self.results_text.insert(tk.END, "\n")
        
        # Summary
        self.results_text.insert(tk.END, "-" * 80 + "\n")
        self.results_text.insert(tk.END, "SUMMARY:\n")
        self.results_text.insert(tk.END, "-" * 80 + "\n")
        
        stage_1_3_count = sum(1 for p in predictions.values() if p == 0)
        stage_4_count = sum(1 for p in predictions.values() if p == 1)
        
        self.results_text.insert(tk.END, f"Models predicting Stage 1-3: {stage_1_3_count}/5\n")
        self.results_text.insert(tk.END, f"Models predicting Stage 4:   {stage_4_count}/5\n\n")
        
        # Consensus
        if stage_1_3_count > stage_4_count:
            consensus = "Stage 1-3"
        elif stage_4_count > stage_1_3_count:
            consensus = "Stage 4"
        else:
            consensus = "Tie (2.5 vs 2.5)"
        
        self.results_text.insert(tk.END, f"Consensus Prediction: {consensus}\n")
        
        self.results_text.insert(tk.END, "\n" + "=" * 80 + "\n")
        
        self.results_text.config(state=tk.DISABLED)
        
        # Switch to results tab
        self.notebook.select(1)


def main():
    root = tk.Tk()
    app = ModelPredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

