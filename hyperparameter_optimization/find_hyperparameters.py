import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt

# Configuration
DATA_TYPE = 'reflectance'  # Change to 'reflectance' if needed
N_TRIALS = 500  # Number of optimization trials
RANDOM_STATE = 42

# Load data
print(f"Loading {DATA_TYPE} data...")
input_train = np.load('../train_data/laser_parameters_train.npy')
m = 20
output_train = np.load(f'../train_data/{DATA_TYPE}_train.npy')[:, ::m]

print(f"Input shape: {input_train.shape}")
print(f"Output shape: {output_train.shape}")

# Split data into train, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    input_train, 
    output_train,
    test_size=0.2, 
    random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, 
    y_train_full,
    test_size=0.15,  # 15% of training data for validation
    random_state=RANDOM_STATE
)

print(f"Train set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")


def objective(trial):
    """Objective function for Optuna optimization"""
    
    # Define hyperparameter search space
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # For multi-output regression, we'll calculate average RMSE across all outputs
    model = xgb.XGBRegressor(**params)
    
    # Fit model with early stopping
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Calculate RMSE for each output and take the mean
    rmse_per_output = np.sqrt(mean_squared_error(y_val, y_pred, multioutput='raw_values'))
    mean_rmse = np.mean(rmse_per_output)
    
    return mean_rmse


# Create Optuna study
print(f"\nStarting hyperparameter optimization for {DATA_TYPE}...")
study = optuna.create_study(
    direction='minimize',
    study_name=f'xgboost_{DATA_TYPE}_optimization'
)

# Run optimization
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# Print results
print("\n" + "="*80)
print("OPTIMIZATION RESULTS")
print("="*80)
print(f"\nBest trial number: {study.best_trial.number}")
print(f"Best RMSE: {study.best_trial.value:.6f}")
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best parameters on full training set
print("\n" + "="*80)
print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("="*80)

best_params = study.best_params.copy()
best_params['objective'] = 'reg:squarederror'
best_params['eval_metric'] = 'rmse'
best_params['random_state'] = RANDOM_STATE
best_params['n_jobs'] = -1

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(
    X_train_full,
    y_train_full,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# Evaluate on test set
y_pred_test = final_model.predict(X_test)
test_rmse_per_output = np.sqrt(mean_squared_error(y_test, y_pred_test, multioutput='raw_values'))
test_rmse_mean = np.mean(test_rmse_per_output)

print(f"\nTest set RMSE (mean across outputs): {test_rmse_mean:.6f}")
print(f"Test set RMSE (std across outputs): {np.std(test_rmse_per_output):.6f}")
print(f"Test set RMSE (min): {np.min(test_rmse_per_output):.6f}")
print(f"Test set RMSE (max): {np.max(test_rmse_per_output):.6f}")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save best parameters to text file
with open(f'model_results/best_hyperparameters_{DATA_TYPE}.txt', 'w') as f:
    f.write(f"Best hyperparameters for {DATA_TYPE}\n")
    f.write("="*50 + "\n\n")
    f.write(f"Best validation RMSE: {study.best_trial.value:.6f}\n")
    f.write(f"Test RMSE: {test_rmse_mean:.6f}\n\n")
    f.write("Hyperparameters:\n")
    for key, value in study.best_params.items():
        f.write(f"{key}: {value}\n")

print(f"Best hyperparameters saved to: model_results/best_hyperparameters_{DATA_TYPE}.txt")

# Save best parameters as numpy dictionary for easy loading
np.save(f'model_results/best_hyperparameters_{DATA_TYPE}.npy', study.best_params)
print(f"Best hyperparameters saved to: model_results/best_hyperparameters_{DATA_TYPE}.npy")

# Create and save optimization history plot
try:
    fig = plot_optimization_history(study)
    fig.write_image(f'model_results/optimization_history_{DATA_TYPE}.png')
    print(f"Optimization history plot saved to: model_results/optimization_history_{DATA_TYPE}.png")
except Exception as e:
    print(f"Could not save optimization history plot: {e}")

# Create and save parameter importance plot
try:
    fig = plot_param_importances(study)
    fig.write_image(f'model_results/param_importances_{DATA_TYPE}.png')
    print(f"Parameter importance plot saved to: model_results/param_importances_{DATA_TYPE}.png")
except Exception as e:
    print(f"Could not save parameter importance plot: {e}")

# Save study object for later analysis
import joblib
joblib.dump(study, f'model_results/optuna_study_{DATA_TYPE}.pkl')
print(f"Optuna study object saved to: model_results/optuna_study_{DATA_TYPE}.pkl")

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE!")
print("="*80)

# Print Python dictionary format for easy copy-paste
print("\nBest hyperparameters (copy-paste ready):")
print("{")
for key, value in study.best_params.items():
    if isinstance(value, float):
        print(f"    '{key}': {value},")
    else:
        print(f"    '{key}': {value},")
print("    'objective': 'reg:squarederror',")
print("    'random_state': 44")
print("}")