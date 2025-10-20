import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# --- Configuración inicial ---
workspace_dir = os.getcwd()
data_path = os.path.join(workspace_dir, "data", "winequality-red.csv")
mlruns_dir = os.path.join(workspace_dir, "mlruns")
os.makedirs(mlruns_dir, exist_ok=True)

tracking_uri = f"file:///{mlruns_dir.replace(os.sep, '/')}"
mlflow.set_tracking_uri(tracking_uri)
experiment_name = "CI-CD-Lab2"

# Crear o cargar experimento
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# --- Cargar dataset ---
df = pd.read_csv(data_path, sep=',')

# --- Preprocesamiento ---
# 1. Separar features y target
X = df.drop("quality", axis=1)
y = df["quality"]

# 2. Eliminar outliers (basado en 1.5*IQR)
Q1 = X.quantile(0.02)
Q3 = X.quantile(0.98)
IQR = Q3 - Q1
mask = ~((X < (Q1 - 1.5*IQR)) | (X > (Q3 + 1.5*IQR))).any(axis=1)
X = X[mask]
y = y[mask]

# 3. Eliminar variables altamente correlacionadas (>0.9)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 1.00)]
X.drop(columns=to_drop, inplace=True)

# 4. Escalamiento
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 5. División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- GridSearch con RandomForest ---
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# --- Evaluación con cross-validation ---
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
mean_cv_r2 = cv_scores.mean()
y_pred = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)

print(f"🔍 Mejor R² CV: {mean_cv_r2:.4f}")
print(f"🔍 Test R²: {test_r2:.4f}, MSE: {test_mse:.4f}")

# --- Registro en MLflow ---
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("mean_cv_r2", mean_cv_r2)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("test_mse", test_mse)
    
    signature = infer_signature(X_train, best_model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )

print("✅ Entrenamiento completado y modelo registrado en MLflow.")
