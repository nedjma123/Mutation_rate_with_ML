import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import joblib  # <--- ADDED: To save the model

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor

# XGBoost Import
try:
    import xgboost as xgb

    print("✅ XGBoost detected.")
except ImportError:
    print("⚠️ XGBoost not found. Please install it using: pip install xgboost")

# ==========================================
# 1. CONFIGURATION & MULTI-FILE LOADING
# ==========================================
# List of your specific files
file_paths = [
    'test_functions.csv - mutation_rate_dataset_CMOEDDMA.csv.csv',
    'test_functions.csv - cnsga2_result.csv.csv',
    'test_functions.csv - cmoead_population_study.csv (1).csv',
    'CMOEA_DMA_Randomized_Dataset_1.csv'
]

print(f"\n[1] Loading {len(file_paths)} files...")

dfs = []
for fp in file_paths:
    try:
        temp_df = pd.read_csv(fp)
        temp_df.columns = temp_df.columns.str.strip()  # Clean headers
        temp_df['Source_File'] = fp
        dfs.append(temp_df)
        print(f"   ✅ Loaded: {fp} ({len(temp_df)} rows)")
    except Exception as e:
        print(f"   ⚠️ Skipped {fp}: {e}")

if not dfs:
    raise ValueError("❌ No files were loaded successfully. Check your file paths.")

df = pd.concat(dfs, ignore_index=True)
print(f"   Total Data Loaded: {len(df)} rows.")

# ==========================================
# [1.5] INTELLIGENT TARGET CREATION
# ==========================================
print("\n[1.5] Verifying Target Variable...")
TARGET = 'Normalized_HV'


def get_normalized_score(group):
    col_to_use = 'Hypervolume'
    if 'HV' in group.columns: col_to_use = 'HV'

    if col_to_use not in group.columns:
        return pd.Series([0.5] * len(group), index=group.index)

    min_val = group[col_to_use].min()
    max_val = group[col_to_use].max()

    if max_val - min_val == 0:
        return pd.Series([0.5] * len(group), index=group.index)

    return (group[col_to_use] - min_val) / (max_val - min_val)


if TARGET not in df.columns:
    print(f"   ℹ️ Creating '{TARGET}'...")
    if 'Problem' in df.columns:
        df['Problem'] = df['Problem'].astype(str).str.lower().str.strip()
        df[TARGET] = df.groupby('Problem', group_keys=False).apply(get_normalized_score, include_groups=False)
    else:
        df[TARGET] = get_normalized_score(df)
    print(f"   ✅ Created '{TARGET}' successfully.")
else:
    print(f"   ✅ '{TARGET}' already exists.")

# ==========================================
# 2. DATA CLEANING
# ==========================================
print("\n[2] Cleaning Data...")

text_cols = ['Algorithm', 'Problem', 'Problem Type', 'Crossover Type', 'Mutation Type']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
        df[col] = df[col].str.replace(r"[\[\]'\" ]", "", regex=True)
        df[col] = df[col].str.replace("-", "").str.replace("_", "")

# --- CHANGE 1: DO NOT DROP 'Problem' YET ---
# We keep 'Problem' here so we can use it for the table in Step 7.
# We will drop it explicitly inside the Training section (Step 4).
cols_to_drop = ['Execution Time', 'Hypervolume', 'HV', 'GD', 'IGD', 'Spacing', 'Spread', 'Reference Point',
                'Source_File']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# Fill Missing Values
fill_defaults = {
    'Decision Variables Number': df['Decision Variables Number'].median(),
    'Objectives Number': df['Objectives Number'].median(),
    'Constraints Number': 0
}
df = df.fillna(fill_defaults)

# ==========================================
# 3. ADVANCED FEATURE ENGINEERING
# ==========================================
print("[3] Generating Advanced Features...")

df['Complexity_Index'] = df['Objectives Number'] * df['Decision Variables Number']
if 'Constraints Number' in df.columns:
    df['Constraint_Density'] = df['Constraints Number'] / (df['Decision Variables Number'] + 1e-5)

df['Mutation_Strength'] = df['Mutation Rate'] * df['Decision Variables Number']

if 'ELA_Ruggedness' in df.columns:
    df['Mut_x_Ruggedness'] = df['Mutation Rate'] * df['ELA_Ruggedness']

# ==========================================
# 4. TRAINING THE "HONEST" MODEL
# ==========================================
print("\n[4] Training Ensemble Model...")

# --- CHANGE 2: EXPLICITLY DROP CHEATING COLUMNS HERE ---
# We drop 'Problem' and 'Algorithm' so the model relies ONLY on math features.
drop_for_training = [TARGET, 'Problem', 'Algorithm', 'Problem Type', 'Crossover Type', 'Mutation Type']
X = df.drop(columns=[c for c in drop_for_training if c in df.columns], errors='ignore')
y = df[TARGET]

# Filter infinite/NaN
mask = np.isfinite(y)
X = X[mask]
y = y[mask]

print(f"   Training Features: {list(X.columns)}")  # Verify only math features remain

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('num', RobustScaler(), X.select_dtypes(include=['number']).columns),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X.select_dtypes(include=['object']).columns)
])

# Voting Ensemble
xgb_expert = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=8, n_jobs=-1, random_state=42)
rf_expert = RandomForestRegressor(n_estimators=800, max_depth=20, n_jobs=-1, random_state=42)
et_expert = ExtraTreesRegressor(n_estimators=800, max_depth=20, n_jobs=-1, random_state=42)

voting_model = VotingRegressor(
    estimators=[('xgb', xgb_expert), ('rf', rf_expert), ('et', et_expert)],
    weights=[4, 2, 2]
)

final_pipeline = Pipeline([('prep', preprocessor), ('model', voting_model)])
final_pipeline.fit(X_train, y_train)

# --- CHANGE 3: SAVE THE MODEL ---
model_filename = "honest_ai_model.pkl"
joblib.dump(final_pipeline, model_filename)
print(f"   💾 Honest AI Model saved to '{model_filename}'")

# ==========================================
# 5. RESULTS & OPTIMIZATION DEMO
# ==========================================
y_pred = final_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("-" * 40)
print(f"🚀 FINAL ENSEMBLE RESULTS (Honest R²)")
print(f"R² Score: {r2:.5f}")
print("-" * 40)


def optimize_mutation_rate(problem_row, model):
    # 1. Calculate the Physics Anchor (Standard 1/n)
    n_vars = problem_row['Decision Variables Number']

    # Safety for division by zero
    if n_vars <= 0: n_vars = 1
    standard_rate = 1.0 / n_vars

    # 2. DEFINE SMART BOUNDARIES (The Fix)
    # We allow the AI to go 5x lower or 3x higher than standard, but NEVER above 0.4
    lower_bound = standard_rate / 5.0
    upper_bound = min(standard_rate * 3.0, 0.4)

    # Ensure lower bound is sane
    if lower_bound < 0.0001: lower_bound = 0.0001

    # 3. Create the Scan Grid (Logarithmic)
    # This forces the AI to look closely AROUND the standard rate
    rates = np.logspace(np.log10(lower_bound), np.log10(upper_bound), 50)

    # 4. Prepare Data for Prediction
    sim_df = pd.DataFrame([problem_row] * len(rates))
    sim_df['Mutation Rate'] = rates

    # Recalculate features
    sim_df['Mutation_Strength'] = sim_df['Mutation Rate'] * sim_df['Decision Variables Number']
    if 'ELA_Ruggedness' in sim_df.columns:
        sim_df['Mut_x_Ruggedness'] = sim_df['Mutation Rate'] * sim_df['ELA_Ruggedness']

    # Drop non-math columns for prediction
    drop_cols = ['Normalized_HV', 'Problem', 'Algorithm', 'Problem Type',
                 'Crossover Type', 'Mutation Type', 'Source_File']
    sim_input = sim_df.drop(columns=[c for c in drop_cols if c in sim_df.columns], errors='ignore')

    # 5. Predict and Find Best
    scores = model.predict(sim_input)
    best_idx = np.argmax(scores)

    return rates[best_idx], scores[best_idx], rates, scores


# Run Demo
print("\n[5] Running Optimization Demo...")
# We need a raw row from the original DF (before X split) to keep the structure valid
sample_idx = X_test.index[0]
sample_row = df.loc[sample_idx]
best_rate, best_score, rates, scores = optimize_mutation_rate(sample_row, final_pipeline)

plt.figure(figsize=(10, 6))
plt.semilogx(rates, scores, linewidth=2, label='Predicted Performance')  # Log scale plot
plt.scatter([best_rate], [best_score], color='red', s=100, label=f'Best Rate: {best_rate:.4f}')
plt.xlabel('Mutation Rate (Log Scale)')
plt.ylabel('Predicted Normalized HV')
plt.title('AI-Based Mutation Rate Tuning (Log Scan)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.show()

print(f"✅ Recommendation: Use Mutation Rate {best_rate:.4f}")

# ==========================================
# 6. FEATURE IMPORTANCE ANALYSIS
# ==========================================
print("\n[6] Generating Feature Importance Plot...")
# (Feature importance code remains largely the same, just robust checks)
trained_voter = final_pipeline.named_steps['model']
importances_list = []
for name, model in trained_voter.named_estimators_.items():
    if hasattr(model, 'feature_importances_'):
        importances_list.append(model.feature_importances_)

if importances_list:
    avg_importance = np.mean(importances_list, axis=0)
    # Get feature names from preprocessor
    try:
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        cat_encoder = final_pipeline.named_steps['prep'].transformers_[1][1]
        cat_features = cat_encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns)
        feature_names = num_cols + list(cat_features)
    except:
        feature_names = [f"Feature {i}" for i in range(len(avg_importance))]

    # Plot
    min_len = min(len(feature_names), len(avg_importance))
    fi_df = pd.DataFrame({'Feature': feature_names[:min_len], 'Importance': avg_importance[:min_len]})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Top Drivers of Optimization Success', fontsize=14)
    plt.tight_layout()
    plt.show()

# ==========================================
# 7. GENERATE CMOP EXPERIMENT TABLE
# ==========================================
print("\n[7] Generating CMOP Verification Table...")

cmop_mask = (df['Constraints Number'] > 0) | (df['Problem'].str.contains('c-|cmop|constr', case=False, regex=True))
cmop_df = df[cmop_mask].copy()

if cmop_df.empty:
    print("⚠️ No specific CMOPs found. Using complex problems instead.")
    unique_problems = df.sort_values('Complexity_Index', ascending=False).drop_duplicates('Problem').head(5)
else:
    # Grouping works now because we kept 'Problem' in the main df
    unique_problems = cmop_df.groupby('Problem', as_index=False).first()

results = []
print(f"   Optimizing {len(unique_problems)} problems...")

for idx, row in unique_problems.iterrows():
    try:
        n_vars = row['Decision Variables Number']
        std_rate = 1.0 / n_vars if n_vars > 0 else 0.01

        # Use the updated Logspace optimizer
        ai_rate, ai_score, _, _ = optimize_mutation_rate(row, final_pipeline)

        results.append({
            'Problem': row['Problem'],
            'Variables': int(n_vars),
            'Constraints': int(row['Constraints Number']),
            'Standard_Rate (1/n)': round(std_rate, 5),
            'AI_Recommended_Rate': round(ai_rate, 5),
            'Predicted_HV_Gain': round(ai_score, 4),
            'Difference': round(ai_rate - std_rate, 5)
        })
    except Exception as e:
        print(f"   ⚠️ Skipped {row['Problem']}: {e}")

if results:
    results_df = pd.DataFrame(results).sort_values(by='Difference', ascending=False)
    print("\n🏆 CMOP RECOMMENDATION TABLE (Top 5):")
    print(results_df.head(5).to_string(index=False))
    results_df.to_csv('cmop_verification_experiment.csv', index=False)
    print("\n✅ Saved experiment plan to 'cmop_verification_experiment.csv'")