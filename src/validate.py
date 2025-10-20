import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import sys

# Umbral de validación
THRESHOLD_MSE = 0.6   # MSE máximo aceptable
THRESHOLD_R2 = 0.6    # R2 mínimo aceptable

# --- Cargar dataset Wine (Red Wine Quality) ---
workspace_dir = os.getcwd()
data_path = os.path.join(workspace_dir, "data", "winequality-red.csv")
if not os.path.exists(data_path):
    print(f"❌ No se encuentra el archivo {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path, sep=',')
X = df.drop("quality", axis=1)
y = df["quality"]

# Manejo de valores nulos
if X.isnull().sum().sum() > 0:
    X.fillna(X.mean(), inplace=True)
    print("--- Debug: Se llenaron valores nulos con la media ---")

# Escalamiento de features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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
r2 = r2_score(y_test, y_pred)

print(f"🔍 MSE: {mse:.4f} (umbral: {THRESHOLD_MSE})")
print(f"🔍 R²: {r2:.4f} (umbral: {THRESHOLD_R2})")

if mse <= THRESHOLD_MSE and r2 >= THRESHOLD_R2:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)
else:
    print("❌ El modelo no cumple los umbrales establecidos. Deteniendo pipeline.")
    sys.exit(1)

