# ============================================
# MSK-CHORD 2024 PREPROCESSING PIPELINE (CLEAN)
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load Data ----------
df = pd.read_csv('msk_chord_2024_clinical_data.tsv', sep='\t')

# Filter to only keep Stage 1-3 and Stage 4
df = df[df['Stage (Highest Recorded)'].isin(['Stage 1-3', 'Stage 4'])]


# ---------- Select Features ----------
# NOTE: MSI Comment is not available in the dataset (only MSI Score and MSI Type exist)
# All other requested features are included below

selected_features_df = df[[
    # Demographic Features
    'Current Age',                # Current Age
    'Sex',                        # Sex
    'Race',                       # Race
    'Ethnicity',                  # Ethnicity
    'Smoking History (NLP)',     # Smoking History (NLP)
    'Number of Samples Per Patient',  # Number of Samples Per Patient

    # Clinical Features
    'Cancer Type',                # Cancer Type
    'Cancer Type Detailed',       # Cancer Type Detailed
    'Primary Tumor Site',         # Primary Tumor Site
    'Tumor Site: Adrenal Glands (NLP)',  # Tumor Site: Adrenal Glands (NLP)
    'Tumor Site: Bone (NLP)',    # Tumor Site: Bone (NLP)
    'Tumor Site: CNS/Brain (NLP)',  # Tumor Site: CNS/Brain (NLP)
    'Tumor Site: Intra Abdominal',  # Tumor Site: Intra Abdominal
    'Tumor Site: Liver (NLP)',   # Tumor Site: Liver (NLP)
    'Tumor Site: Lung (NLP)',    # Tumor Site: Lung (NLP)
    'Tumor Site: Lymph Node (NLP)',  # Tumor Site: Lymph Node (NLP)
    'Tumor Site: Pleura (NLP)',  # Tumor Site: Pleura (NLP)
    'Tumor Site: Reproductive Organs (NLP)',  # Tumor Site: Reproductive Organs (NLP)
    'Tumor Site: Other (NLP)',   # Tumor Site: Other (NLP)

    # Genomic Features
    'TMB (nonsynonymous)',        # TMB (nonsynonymous)
    'Mutation Count',             # Mutation Count
    'Fraction Genome Altered',    # Fraction Genome Altered
    'MSI Score',                  # MSI Score
    'MSI Type',                   # MSI Type
    # 'MSI Comment' - NOT AVAILABLE in dataset
    'Gene Panel',                 # Gene Panel (IMPACT341 / IMPACT468)
    'Somatic Status',             # Somatic Status
    'Sample coverage',            # Sample coverage
    'Tumor Purity'                # Tumor Purity
]].copy()

# =====================================================
#           CASE NORMALIZATION (Fix case inconsistencies)
# =====================================================

# Normalize case for categorical columns to fix inconsistencies like "Breast" vs "breast"
# Use title case (first letter uppercase, rest lowercase) for consistency
# Preserve "Unknown" and "Other" as-is

categorical_cols_to_normalize = [
    'Primary Tumor Site', 'Cancer Type', 'Cancer Type Detailed',
    'Race', 'Ethnicity', 'Smoking History (NLP)'
]

def normalize_case(value):
    """Normalize case while preserving special values."""
    if pd.isna(value) or value == "":
        return value
    value_str = str(value).strip()
    # Handle special values - standardize to title case for consistency
    value_lower = value_str.lower()
    if value_lower in ['unknown', 'other', 'nan']:
        return value_str.title()
    # Use title case for everything else (e.g., "breast" -> "Breast", "Sigmoid colon" -> "Sigmoid Colon")
    return value_str.title()

for col in categorical_cols_to_normalize:
    if col in selected_features_df.columns:
        # Count before normalization
        before_count = selected_features_df[col].nunique()
        # Normalize
        selected_features_df[col] = selected_features_df[col].apply(normalize_case)
        # Count after normalization
        after_count = selected_features_df[col].nunique()
        if before_count != after_count:
            print(f"Case normalization: {col} - Reduced from {before_count} to {after_count} unique values")

print("Case normalization complete.\n")


# =====================================================
#           STAGE MAPPING (Using Stage Highest Recorded)
# =====================================================

# Use "Stage (Highest Recorded)" directly as the dependent variable
# Map: Stage 1-3 -> 0, Stage 4 -> 1 (binary classification)

selected_features_df['Stage_ML'] = df['Stage (Highest Recorded)'].map({
    'Stage 1-3': 0,  # Stage 1-3 -> Class 0
    'Stage 4': 1     # Stage 4 -> Class 1
})

print("\nStage Distribution (using Stage Highest Recorded):")
print(selected_features_df['Stage_ML'].value_counts().sort_index())
print("\n  Stage 1-3:", (selected_features_df['Stage_ML'] == 0).sum())
print("  Stage 4:", (selected_features_df['Stage_ML'] == 1).sum())


# =====================================================
#           MISSING VALUE IMPUTATION
# =====================================================

# Numerical columns → fill with median
num_cols = [
    'Current Age', 'Number of Samples Per Patient',
    'Sample coverage', 'Tumor Purity', 'TMB (nonsynonymous)',
    'Mutation Count', 'Fraction Genome Altered', 'MSI Score'
]

for col in num_cols:
    if col in selected_features_df.columns:
        selected_features_df[col] = selected_features_df[col].fillna(selected_features_df[col].median())

# Categorical columns → fill with "Unknown"
cat_cols = [
    'Sex', 'Race', 'Ethnicity', 'Smoking History (NLP)',
    'Cancer Type', 'Cancer Type Detailed', 'Primary Tumor Site',
    'MSI Type', 'Gene Panel', 'Somatic Status'
]

for col in cat_cols:
    if col in selected_features_df.columns:
        selected_features_df[col] = selected_features_df[col].fillna("Unknown")

# Tumor Site binary columns → fill with "Unknown" (they are Yes/No/Unknown)
tumor_site_cols = [
    'Tumor Site: Adrenal Glands (NLP)', 'Tumor Site: Bone (NLP)', 'Tumor Site: CNS/Brain (NLP)', 'Tumor Site: Intra Abdominal',
    'Tumor Site: Liver (NLP)', 'Tumor Site: Lung (NLP)', 'Tumor Site: Lymph Node (NLP)', 'Tumor Site: Pleura (NLP)',
    'Tumor Site: Reproductive Organs (NLP)', 'Tumor Site: Other (NLP)'
]

for col in tumor_site_cols:
    if col in selected_features_df.columns:
        selected_features_df[col] = selected_features_df[col].fillna("Unknown")


# =====================================================
#           REMOVE RARE CATEGORIES
# =====================================================

# Group rare categories (appearing < threshold times) into "Other"
# This reduces dimensionality and prevents noise from sparse categories
RARE_CATEGORY_THRESHOLD = 10  # Categories with <10 samples will be grouped as "Other"

# Columns to apply rare category grouping (exclude binary/ordinal columns)
cols_to_group = [
    'Cancer Type Detailed', 'Primary Tumor Site',
    'Race', 'Ethnicity'  # These often have many rare categories
]

for col in cols_to_group:
    if col in selected_features_df.columns:
        # Count frequency of each category
        value_counts = selected_features_df[col].value_counts()
        # Identify rare categories (excluding "Unknown" and "Other" if they exist)
        rare_categories = value_counts[value_counts < RARE_CATEGORY_THRESHOLD].index.tolist()
        # Exclude "Unknown" and "Other" from being grouped
        rare_categories = [cat for cat in rare_categories if str(cat) not in ['Unknown', 'Other', 'nan']]
        
        if len(rare_categories) > 0:
            # Replace rare categories with "Other"
            selected_features_df[col] = selected_features_df[col].replace(rare_categories, 'Other')
            print(f"  {col}: Grouped {len(rare_categories)} rare categories into 'Other'")

print("\nRare category grouping complete.\n")


# =====================================================
#            GENOMIC TRANSFORMATIONS
# =====================================================

# log1p-transform skewed variables (after imputation)
for col in ['TMB (nonsynonymous)', 'Mutation Count']:
    if col in selected_features_df.columns:
        selected_features_df[col] = np.log1p(selected_features_df[col])


# =====================================================
#           ENCODING (Categorical & Binary)
# =====================================================

# Save unique categorical values before encoding (for GUI dropdowns)
# This ensures dropdowns show normalized values that match what the model expects
categorical_options_for_gui = {}
for col in cat_cols:
    if col in selected_features_df.columns:
        unique_vals = selected_features_df[col].dropna().unique().tolist()
        categorical_options_for_gui[col] = sorted([str(v) for v in unique_vals if pd.notna(v)])

# Save tumor site options (before encoding to 0/1)
tumor_site_options_for_gui = {}
for col in tumor_site_cols:
    if col in selected_features_df.columns:
        unique_vals = selected_features_df[col].dropna().unique().tolist()
        tumor_site_options_for_gui[col] = sorted([str(v) for v in unique_vals if pd.notna(v)])

# Label encode Tumor Site binary columns (Yes/No/Unknown -> 1/0/0)
# We'll use binary encoding for tumor site features to reduce dimensionality
# Yes -> 1, No -> 0, Unknown -> 0 (treating Unknown as negative)
# Alternative: Unknown -> 0.5 if you want to preserve uncertainty
for col in tumor_site_cols:
    if col in selected_features_df.columns:
        # Map: Yes -> 1, No -> 0, Unknown -> 1 (treating Unknown as positive)
        # Or you could use: Yes -> 1, No -> 0, Unknown -> 0.5
        selected_features_df[col] = selected_features_df[col].map({'Yes': 1, 'No': 0, 'Unknown': 0})

# One-hot encode categorical columns
selected_features_df = pd.get_dummies(selected_features_df, columns=cat_cols, drop_first=True)


# =====================================================
#         FEATURE ENGINEERING (Nonlinear Relationships)
# =====================================================

# Add biologically meaningful binary flags for nonlinear relationships
# These help models (especially Logistic Regression) capture thresholds

if 'TMB (nonsynonymous)' in selected_features_df.columns:
    # High TMB threshold (typically >10 mutations/Mb is considered high)
    selected_features_df['TMB_high'] = (selected_features_df['TMB (nonsynonymous)'] > 10).astype(int)

if 'Fraction Genome Altered' in selected_features_df.columns:
    # High FGA threshold (>0.2 = 20% of genome altered)
    selected_features_df['FGA_high'] = (selected_features_df['Fraction Genome Altered'] > 0.2).astype(int)

if 'MSI Score' in selected_features_df.columns:
    # High MSI threshold (typically >10 is considered high MSI)
    selected_features_df['MSI_high'] = (selected_features_df['MSI Score'] > 10).astype(int)

if 'Tumor Purity' in selected_features_df.columns:
    # High tumor purity (>0.5 = 50% tumor cells)
    selected_features_df['Purity_high'] = (selected_features_df['Tumor Purity'] > 0.5).astype(int)

print("Feature engineering complete: Added TMB_high, FGA_high, MSI_high, Purity_high flags\n")


# =====================================================
#         FEATURE CLEANUP (Remove Sparse Features)
# =====================================================

# Remove extremely sparse one-hot encoded features (<1% positive samples)
# This reduces noise and improves model performance
X_temp = selected_features_df.drop(columns=['Stage_ML'])
sparse_threshold = 0.01  # 1% of samples

features_to_keep = []
for col in X_temp.columns:
    if X_temp[col].dtype in ['int64', 'float64']:
        # For binary/numeric features, check if they're too sparse
        if X_temp[col].nunique() == 2:  # Binary feature
            positive_ratio = X_temp[col].sum() / len(X_temp)
            if positive_ratio >= sparse_threshold or positive_ratio <= (1 - sparse_threshold):
                features_to_keep.append(col)
            else:
                continue  # Skip this sparse feature
        else:
            features_to_keep.append(col)  # Keep numeric features
    else:
        features_to_keep.append(col)  # Keep non-numeric features

# Filter to keep only non-sparse features
selected_features_df = selected_features_df[features_to_keep + ['Stage_ML']]
print(f"Feature cleanup: Removed {len(X_temp.columns) - len(features_to_keep)} sparse features")
print(f"Remaining features: {len(features_to_keep)}\n")


# =====================================================
#         FINAL: X, y dataset ready for ML
# =====================================================

X = selected_features_df.drop(columns=['Stage_ML'])
y = selected_features_df['Stage_ML']

print("\nFinal X shape:", X.shape)
print("Final y value counts:\n", y.value_counts())

# =====================================================
#           FEATURE COUNT BY CATEGORY (After Preprocessing)
# =====================================================

# Define feature category prefixes
demographic_prefixes = [
    'Current Age', 'Sex', 'Race', 'Ethnicity',
    'Smoking History (NLP)', 'Number of Samples Per Patient', 
]

clinical_prefixes = [
    'Cancer Type', 'Cancer Type Detailed', 'Primary Tumor Site',
    'Tumor Site: Adrenal Glands (NLP)', 'Tumor Site: Bone (NLP)', 'Tumor Site: CNS/Brain (NLP)', 'Tumor Site: Intra Abdominal',
    'Tumor Site: Liver (NLP)', 'Tumor Site: Lung (NLP)', 'Tumor Site: Lymph Node (NLP)', 'Tumor Site: Pleura (NLP)',
    'Tumor Site: Reproductive Organs (NLP)', 'Tumor Site: Other (NLP)'
]

genomic_prefixes = [
    'TMB (nonsynonymous)', 'Mutation Count', 'Fraction Genome Altered',
    'MSI Score', 'MSI Type', 'Gene Panel', 'Somatic Status',
    'Sample coverage', 'Tumor Purity'
]

# Helper function to find columns matching prefixes
def cols_from_prefixes(cols, prefixes):
    sel = []
    for c in cols:
        if any(str(c).startswith(pref) for pref in prefixes):
            sel.append(c)
    return sel

# Count features in each category
dem_cols = cols_from_prefixes(X.columns, demographic_prefixes)
clin_cols = cols_from_prefixes(X.columns, clinical_prefixes)
geno_cols = cols_from_prefixes(X.columns, genomic_prefixes)

# Also count engineered features (TMB_high, FGA_high, MSI_high, Purity_high)
engineered_features = [col for col in X.columns if col in ['TMB_high', 'FGA_high', 'MSI_high', 'Purity_high']]

print("\n" + "=" * 70)
print("FEATURE COUNT BY CATEGORY (After Preprocessing)")
print("=" * 70)
print(f"Demographic features: {len(dem_cols)}")
print(f"Clinical features: {len(clin_cols)}")
print(f"Genomic features: {len(geno_cols)}")
print(f"Engineered features: {len(engineered_features)}")
print(f"Total features: {X.shape[1]}")
print("=" * 70)
print()


# =====================================================
#           TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# =====================================================
#           CLASS WEIGHTING (Handle Imbalance)
# =====================================================

from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights for balanced learning
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

print("\nClass weights (for balanced learning):")
for stage, weight in class_weight_dict.items():
    if stage == 0:
        stage_label = "Stage 1-3"
    else:
        stage_label = "Stage 4"
    print(f"  {stage_label}: {weight:.3f}")

# For XGBoost, we need sample weights
sample_weights = np.array([class_weight_dict[y] for y in y_train])


# =====================================================
#           Feature scaling (only for logistic regression)
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================================================
# 1️⃣ LOGISTIC REGRESSION (with class weighting)
# ===========================================================
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    solver='lbfgs'  # Good for multiclass
)
log_reg.fit(X_train_scaled, y_train)
log_pred = log_reg.predict(X_test_scaled)

print("\n=== Logistic Regression ===")
print("Note: Class 0 = Stage 1-3, Class 1 = Stage 4")
print(classification_report(y_test, log_pred))

# Confusion matrix
log_cm = confusion_matrix(y_test, log_pred)

# Calculate per-class accuracy percentages
def print_class_accuracy(y_true, y_pred, model_name):
    """Print accuracy percentage for each class."""
    print(f"\n{model_name} - Per-Class Accuracy (%):")
    print("-" * 50)
    class_labels = ['Stage 1-3', 'Stage 4']
    
    for class_idx in sorted(np.unique(y_true)):
        # Get actual samples of this class
        actual_mask = y_true == class_idx
        actual_count = actual_mask.sum()
        
        # Get correct predictions for this class
        correct_mask = (y_true == class_idx) & (y_pred == class_idx)
        correct_count = correct_mask.sum()
        
        # Calculate accuracy percentage
        accuracy_pct = (correct_count / actual_count * 100) if actual_count > 0 else 0.0
        
        label = class_labels[class_idx] if class_idx < len(class_labels) else f"Class {class_idx}"
        print(f"  {label:12s}: {correct_count:4d}/{actual_count:4d} = {accuracy_pct:5.2f}%")
    
    # Overall accuracy
    overall_acc = (y_true == y_pred).sum() / len(y_true) * 100
    print("-" * 50)
    print(f"  Overall Accuracy: {overall_acc:.2f}%")
    print()

print_class_accuracy(y_test, log_pred, "Logistic Regression")


# ===========================================================
# 2️⃣ RANDOM FOREST (with class weighting and tuning)
# ===========================================================
rf = RandomForestClassifier(
    n_estimators=400,  # More trees
    max_depth=20,  # Limit depth to prevent overfitting
    min_samples_split=10,  # Require more samples to split
    min_samples_leaf=5,  # Require more samples in leaf
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n=== Random Forest ===")
print("Note: Class 0 = Stage 1-3, Class 1 = Stage 4")
print(classification_report(y_test, rf_pred))

rf_cm = confusion_matrix(y_test, rf_pred)

print_class_accuracy(y_test, rf_pred, "Random Forest")


# ===========================================================
# 3️⃣ XGBOOST (tuned with class weighting)
# ===========================================================
xgb_model = xgb.XGBClassifier(
    n_estimators=600,  # More trees for better performance
    learning_rate=0.02,  # Lower learning rate for better convergence
    max_depth=8,  # Deeper trees
    min_child_weight=3,  # Balanced min_child_weight
    subsample=0.85,  # Slightly higher subsample
    colsample_bytree=0.85,  # Slightly higher column sampling
    gamma=0.1,  # Regularization
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.5,  # L2 regularization
    objective="binary:logistic",  # Binary classification objective
    eval_metric="logloss",  # Binary classification metric
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=None  # Using sample weights instead
)
# Fit with sample weights to handle class imbalance
xgb_model.fit(
    X_train, 
    y_train,
    sample_weight=sample_weights,
    verbose=False
)
xgb_pred = xgb_model.predict(X_test)

print("\n=== XGBoost ===")
print("Note: Class 0 = Stage 1-3, Class 1 = Stage 4")
print(classification_report(y_test, xgb_pred))

xgb_cm = confusion_matrix(y_test, xgb_pred)

print_class_accuracy(y_test, xgb_pred, "XGBoost")


# ===========================================================
# 4️⃣ ARTIFICIAL NEURAL NETWORK (ANN) (with class weighting)
# ===========================================================
ann_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers: 100 neurons, then 50 neurons
    activation='relu',  # Rectified Linear Unit activation
    solver='adam',  # Adam optimizer
    alpha=0.01,  # L2 regularization parameter
    batch_size='auto',  # Automatic batch size
    learning_rate='adaptive',  # Adaptive learning rate
    learning_rate_init=0.001,  # Initial learning rate
    max_iter=500,  # Maximum iterations
    shuffle=True,  # Shuffle samples each iteration
    random_state=42,
    early_stopping=True,  # Enable early stopping
    validation_fraction=0.1,  # Fraction of training data for validation
    n_iter_no_change=10  # Number of iterations with no improvement before stopping
)
# Fit with sample weights to handle class imbalance
ann_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
ann_pred = ann_model.predict(X_test_scaled)

print("\n=== Artificial Neural Network (ANN) ===")
print("Note: Class 0 = Stage 1-3, Class 1 = Stage 4")
print(classification_report(y_test, ann_pred))

ann_cm = confusion_matrix(y_test, ann_pred)

print_class_accuracy(y_test, ann_pred, "Artificial Neural Network")


# ===========================================================
# 5️⃣ SUPPORT VECTOR MACHINE (SVM) (with class weighting)
# ===========================================================
svm_model = SVC(
    kernel='rbf',  # Radial Basis Function kernel
    C=1.0,  # Regularization parameter
    gamma='scale',  # Kernel coefficient
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    probability=True  # Enable probability estimates
)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

print("\n=== Support Vector Machine (SVM) ===")
print("Note: Class 0 = Stage 1-3, Class 1 = Stage 4")
print(classification_report(y_test, svm_pred))

svm_cm = confusion_matrix(y_test, svm_pred)

print_class_accuracy(y_test, svm_pred, "Support Vector Machine")


# ===========================================================
#           SUMMARY: Per-Class Accuracy Comparison
# ===========================================================

def get_class_accuracy_dict(y_true, y_pred):
    """Get accuracy percentage for each class as a dictionary."""
    accuracies = {}
    for class_idx in sorted(np.unique(y_true)):
        actual_mask = y_true == class_idx
        actual_count = actual_mask.sum()
        correct_count = ((y_true == class_idx) & (y_pred == class_idx)).sum()
        accuracy_pct = (correct_count / actual_count * 100) if actual_count > 0 else 0.0
        accuracies[class_idx] = accuracy_pct
    return accuracies

# Get accuracies for all models
log_acc = get_class_accuracy_dict(y_test, log_pred)
rf_acc = get_class_accuracy_dict(y_test, rf_pred)
xgb_acc = get_class_accuracy_dict(y_test, xgb_pred)
ann_acc = get_class_accuracy_dict(y_test, ann_pred)
svm_acc = get_class_accuracy_dict(y_test, svm_pred)

# Create comparison DataFrame
class_labels = ['Stage 1-3', 'Stage 4']
comparison_df = pd.DataFrame({
    'Class': class_labels,
    'Logistic Regression (%)': [log_acc.get(i, 0) for i in range(2)],
    'Random Forest (%)': [rf_acc.get(i, 0) for i in range(2)],
    'XGBoost (%)': [xgb_acc.get(i, 0) for i in range(2)],
    'ANN (%)': [ann_acc.get(i, 0) for i in range(2)],
    'SVM (%)': [svm_acc.get(i, 0) for i in range(2)]
})

# Add overall accuracy row
overall_log = (y_test == log_pred).sum() / len(y_test) * 100
overall_rf = (y_test == rf_pred).sum() / len(y_test) * 100
overall_xgb = (y_test == xgb_pred).sum() / len(y_test) * 100
overall_ann = (y_test == ann_pred).sum() / len(y_test) * 100
overall_svm = (y_test == svm_pred).sum() / len(y_test) * 100

comparison_df.loc[len(comparison_df)] = [
    'Overall',
    overall_log,
    overall_rf,
    overall_xgb,
    overall_ann,
    overall_svm
]

print("\n" + "=" * 70)
print("PER-CLASS ACCURACY COMPARISON")
print("=" * 70)
print(comparison_df.to_string(index=False))
print("=" * 70)
print()

# ------------------------------
# Feature Importance Grouped Contribution Code
# ------------------------------

# Use X_test columns for feature names
X_test_df = X_test.copy()

# Use built-in XGBoost feature importance
feature_importance = xgb_model.feature_importances_
feature_importance_per_feature = pd.Series(feature_importance, index=X_test_df.columns).sort_values(ascending=False)

# --- Map features to your 3 categories (after encoding) ---
demographic_prefixes = [
    'Current Age', 'Sex', 'Race', 'Ethnicity',
    'Smoking History (NLP)', 'Number of Samples Per Patient', 
]

clinical_prefixes = [
    'Cancer Type', 'Cancer Type Detailed', 'Primary Tumor Site',
    'Tumor Site: Adrenal Glands (NLP)', 'Tumor Site: Bone (NLP)', 'Tumor Site: CNS/Brain (NLP)', 'Tumor Site: Intra Abdominal',
    'Tumor Site: Liver (NLP)', 'Tumor Site: Lung (NLP)', 'Tumor Site: Lymph Node (NLP)', 'Tumor Site: Pleura (NLP)',
    'Tumor Site: Reproductive Organs (NLP)', 'Tumor Site: Other (NLP)'
]

genomic_prefixes = [
    'TMB (nonsynonymous)', 'Mutation Count', 'Fraction Genome Altered',
    'MSI Score', 'MSI Type', 'Gene Panel', 'Somatic Status',
    'Sample coverage', 'Tumor Purity'
]

def cols_from_prefixes(cols, prefixes):
    sel = []
    for c in cols:
        if any(str(c).startswith(pref) for pref in prefixes):
            sel.append(c)
    return sel

dem_cols = cols_from_prefixes(X_test_df.columns, demographic_prefixes)
clin_cols = cols_from_prefixes(X_test_df.columns, clinical_prefixes)
geno_cols = cols_from_prefixes(X_test_df.columns, genomic_prefixes)

# Sum feature importance within each group
dem_score = feature_importance_per_feature.loc[feature_importance_per_feature.index.intersection(dem_cols)].sum()
clin_score = feature_importance_per_feature.loc[feature_importance_per_feature.index.intersection(clin_cols)].sum()
geno_score = feature_importance_per_feature.loc[feature_importance_per_feature.index.intersection(geno_cols)].sum()

group_scores = pd.Series({
    'Demographic': dem_score,
    'Clinical': clin_score,
    'Genomic': geno_score
}).sort_values(ascending=False)

# Normalize to percent of total (optional, clearer)
group_percent = 100 * group_scores / group_scores.sum()

# --- Plots ---
plt.figure(figsize=(7,5))
# Create a DataFrame for the barplot to use hue parameter (fixes FutureWarning)
plot_df = pd.DataFrame({
    'Group': group_percent.index,
    'Percent': group_percent.values
})
# Map groups to colors
color_map = {
    'Demographic': 'skyblue',
    'Clinical': 'lightgreen',
    'Genomic': 'salmon'
}
sns.barplot(data=plot_df, x='Percent', y='Group', hue='Group', palette=color_map, legend=False)
plt.xlabel("Percent contribution to feature importance (group sum)")
plt.title("Grouped Feature Importance: Demographic vs Clinical vs Genomic")
for i, v in enumerate(group_percent.values):
    plt.text(v + 0.5, i, f"{v:.1f}%", va='center')
plt.xlim(0, group_percent.max()*1.15)
plt.tight_layout()
plt.show()

# Print numeric summary
print("\nGrouped Feature Importance scores (sum of feature importance per feature in group):")
print(group_scores)
print("\nGrouped Feature Importance percentages:")
print(group_percent)

# --- Top 5 Contributing Features ---
TOP_N = 5
top_features = feature_importance_per_feature.head(TOP_N)

# Print top 5 features
print(f"\nTop {TOP_N} Contributing Features:")
print("-" * 60)
for i, (feature, importance) in enumerate(top_features.items(), 1):
    print(f"{i}. {feature}: {importance:.6f}")

# Visualize top 5 features
plt.figure(figsize=(10, 6))
top_features_sorted = top_features.sort_values(ascending=True)  # Sort ascending for horizontal bar plot
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features_sorted)))
plt.barh(range(len(top_features_sorted)), top_features_sorted.values, color=colors)
plt.yticks(range(len(top_features_sorted)), top_features_sorted.index)
plt.xlabel("Feature Importance")
plt.title(f"Top {TOP_N} Contributing Features")
plt.gca().invert_yaxis()  # Show highest at top
for i, v in enumerate(top_features_sorted.values):
    plt.text(v + 0.001, i, f"{v:.4f}", va='center')
plt.xlim(0, max(top_features_sorted.values) * 1.1)  # 10% extra space for the labels
plt.tight_layout()
plt.show()

# Save results for later reporting
group_scores.to_csv('grouped_feature_importance_scores.csv')
top_features.to_csv('top_5_features.csv')

# ===========================================================
#           SAVE MODELS AND PREPROCESSING OBJECTS FOR GUI
# ===========================================================
import pickle
import os

# Create a directory for saved models
os.makedirs('saved_models', exist_ok=True)

# Save all models
with open('saved_models/log_reg_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

with open('saved_models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('saved_models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('saved_models/ann_model.pkl', 'wb') as f:
    pickle.dump(ann_model, f)

with open('saved_models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Save preprocessing objects
with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature information needed for preprocessing
preprocessing_info = {
    'feature_columns': list(X.columns),
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'tumor_site_cols': tumor_site_cols,
    'medians': {col: selected_features_df[col].median() for col in num_cols if col in selected_features_df.columns},
    'rare_category_threshold': RARE_CATEGORY_THRESHOLD,
    'cols_to_group': cols_to_group
}

with open('saved_models/preprocessing_info.pkl', 'wb') as f:
    pickle.dump(preprocessing_info, f)

# Save original feature names for GUI input
original_features = [
    'Current Age', 'Sex', 'Race', 'Ethnicity', 'Smoking History (NLP)',
    'Number of Samples Per Patient', 'Cancer Type', 'Cancer Type Detailed',
    'Primary Tumor Site', 'Tumor Site: Adrenal Glands (NLP)', 'Tumor Site: Bone (NLP)',
    'Tumor Site: CNS/Brain (NLP)', 'Tumor Site: Intra Abdominal', 'Tumor Site: Liver (NLP)',
    'Tumor Site: Lung (NLP)', 'Tumor Site: Lymph Node (NLP)', 'Tumor Site: Pleura (NLP)',
    'Tumor Site: Reproductive Organs (NLP)', 'Tumor Site: Other (NLP)',
    'TMB (nonsynonymous)', 'Mutation Count', 'Fraction Genome Altered',
    'MSI Score', 'MSI Type', 'Gene Panel', 'Somatic Status',
    'Sample coverage', 'Tumor Purity'
]

# Use the categorical options saved before encoding (normalized and grouped)
categorical_options = categorical_options_for_gui.copy()
categorical_options.update(tumor_site_options_for_gui)

gui_info = {
    'original_features': original_features,
    'categorical_options': categorical_options
}

with open('saved_models/gui_info.pkl', 'wb') as f:
    pickle.dump(gui_info, f)

print("\n" + "=" * 70)
print("MODELS AND PREPROCESSING OBJECTS SAVED")
print("=" * 70)
print("Saved to 'saved_models' directory:")
print("  - log_reg_model.pkl")
print("  - rf_model.pkl")
print("  - xgb_model.pkl")
print("  - ann_model.pkl")
print("  - svm_model.pkl")
print("  - scaler.pkl")
print("  - preprocessing_info.pkl")
print("  - gui_info.pkl")
print("=" * 70)


