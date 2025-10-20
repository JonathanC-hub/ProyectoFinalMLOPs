import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

# --- Configuración de entorno ---
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# --- Paths y MLflow ---
workspace_dir = Path(os.getenv("GITHUB_WORKSPACE", ".")) if IS_GITHUB_ACTIONS else Path.cwd()
mlruns_dir = workspace_dir / "mlruns"

if os.getenv("MLFLOW_TRACKING_URI"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
elif os.name == 'nt':
    tracking_uri = "./mlruns"
else:
    tracking_uri = "./mlruns"

mlflow.set_tracking_uri(tracking_uri)

print(f"📁 Workspace: {workspace_dir}")
print(f"📊 MLflow URI: {tracking_uri}")

experiment_name = "CI-CD-Lab2"

# --- Umbrales de validación ---
THRESHOLD_MSE = 0.6
THRESHOLD_R2 = 0.4
print(f"📏 Umbrales: MSE ≤ {THRESHOLD_MSE}, R² ≥ {THRESHOLD_R2}")

# --- Cargar dataset ---
data_path = workspace_dir / "data" / "winequality-red.csv"
if not data_path.exists():
    print(f"❌ No se encuentra {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path, sep=',')
print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# --- Preprocesamiento idéntico a train.py ---
X = df.drop("quality", axis=1)
y = df["quality"]

# Valores nulos
if X.isnull().sum().sum() > 0:
    X.fillna(X.mean(), inplace=True)

# Eliminación de outliers
z_scores = np.abs((X - X.mean()) / X.std())
mask = (z_scores < 3).all(axis=1)
X = X[mask]
y = y[X.index]

# Eliminación de variables correlacionadas (>0.9)
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
if to_drop:
    X.drop(columns=to_drop, inplace=True)

# --- Cargar última run del experimento ---
client = MlflowClient()
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    print(f"❌ Experimento '{experiment_name}' no encontrado.")
    sys.exit(1)

runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
if not runs:
    print(f"❌ No hay runs registradas en '{experiment_name}'. Ejecuta train.py primero.")
    sys.exit(1)

run_id = runs[0].info.run_id
print(f"🎯 Última run encontrada: {run_id}")

# --- Cargar modelo y scaler ---
model_uri = f"runs:/{run_id}/model"
scaler_uri = f"runs:/{run_id}/scaler/scaler.pkl"

model = mlflow.sklearn.load_model(model_uri)
scaler_local_path = mlflow.artifacts.download_artifacts(scaler_uri)
scaler = joblib.load(scaler_local_path)

# División de datos (idéntica a train)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = X_test[scaler.feature_names_in_]

# Aplicar escalamiento con el mismo scaler del entrenamiento
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=scaler.feature_names_in_)
print(f"📏 Features escaladas: {X_test_scaled.shape[1]} variables")

# --- Predicción y evaluación ---
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n{'='*50}")
print("📊 RESULTADOS DE VALIDACIÓN")
print(f"🔍 MSE: {mse:.4f} (umbral: ≤ {THRESHOLD_MSE})")
print(f"🔍 R²:  {r2:.4f} (umbral: ≥ {THRESHOLD_R2})")
print(f"{'='*50}\n")

# --- Validación de umbrales ---
mse_pass = mse <= THRESHOLD_MSE
r2_pass = r2 >= THRESHOLD_R2

print("📋 Verificación de criterios:")
print(f"   {'✅' if mse_pass else '❌'} MSE")
print(f"   {'✅' if r2_pass else '❌'} R²")
print()

# Registrar métricas en MLflow
try:
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("validation_mse", mse)
        mlflow.log_metric("validation_r2", r2)
        mlflow.log_metric("mse_threshold", THRESHOLD_MSE)
        mlflow.log_metric("r2_threshold", THRESHOLD_R2)
        mlflow.set_tag("validation_status", "passed" if (mse_pass and r2_pass) else "failed")
    print(f"📝 Métricas de validación registradas en MLflow")
except Exception as e:
    print(f"⚠️ No se pudieron registrar métricas en MLflow: {e}")

# Resultado final
if mse_pass and r2_pass:
    print("✅ El modelo CUMPLE los criterios de calidad.")
    sys.exit(0)
else:
    print("❌ El modelo NO CUMPLE los thresholds establecidos.")
    sys.exit(1)
