import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import sys

# Umbral de validación
THRESHOLD = 0.6  # R2 mínimo aceptable, por ejemplo

# --- Cargar dataset Wine (Red Wine Quality) ---
workspace_dir = os.getcwd()
data_path = os.path.join(workspace_dir, "data", "winequality-red.csv")
if not os.path.exists(data_path):
    print(f"❌ No se encuentra el archivo {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path, sep=',')
X = df.drop("quality", axis=1)
y = df["quality"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Cargar último modelo registrado de MLflow ---
experiment_name = "CI-CD-Lab2"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    print(f"❌ Experimento {experiment_name} no existe.")
    sys.exit(1)

client = MlflowClient()
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
if not runs:
    print(f"❌ No hay runs para el experimento {experiment_name}.")
    sys.exit(1)

run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# --- Predicción y validación ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"🔍 MSE: {mse:.4f} (umbral de aceptación: {THRESHOLD})")

if mse <= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)
else:
    print("❌ El modelo no cumple el umbral. Deteniendo pipeline.")
    sys.exit(1)
