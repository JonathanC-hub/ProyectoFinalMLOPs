import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# --- Configuración de paths ---
workspace_dir = os.getcwd()
data_path = os.path.join(workspace_dir, "data", "winequality-red.csv")
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = f"file:///{mlruns_dir.replace(os.sep, '/')}"
mlflow.set_tracking_uri(tracking_uri)
experiment_name = "CI-CD-Lab2"

# --- Umbrales de validación ---
THRESHOLD_MSE = 0.6
THRESHOLD_R2 = 0.5

# --- Cargar dataset ---
df = pd.read_csv(data_path, sep=',')

# --- Preprocesamiento idéntico a train ---
X = df.drop("quality", axis=1)
y = df["quality"]

# Outliers
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X < (Q1 - 1.5*IQR)) | (X > (Q3 + 1.5*IQR))).any(axis=1)
X = X[mask]
y = y[mask]

# Correlación
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
X.drop(columns=to_drop, inplace=True)

# Escalamiento
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split test
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Cargar último modelo registrado ---
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    print(f"❌ Experimento {experiment_name} no encontrado.")
    sys.exit(1)

from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
if not runs:
    print(f"❌ No hay runs registradas en el experimento {experiment_name}.")
    sys.exit(1)

run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# --- Predicción y evaluación ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"🔍 MSE: {mse:.4f} (umbral: {THRESHOLD_MSE})")
print(f"🔍 R²: {r2:.4f} (umbral: {THRESHOLD_R2})")

# --- Validación ---
if mse <= THRESHOLD_MSE and r2 >= THRESHOLD_R2:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)
else:
    print("❌ El modelo NO cumple los thresholds establecidos. Deteniendo pipeline.")
    sys.exit(1)
