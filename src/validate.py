import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# --- Configuración de entorno ---
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# --- Paths y MLflow ---
if IS_GITHUB_ACTIONS:
    workspace_dir = Path(os.getenv("GITHUB_WORKSPACE", "."))
else:
    workspace_dir = Path.cwd()

mlruns_dir = workspace_dir / "mlruns"

# MLflow requiere formato específico según el OS
if os.getenv("MLFLOW_TRACKING_URI"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    print(f"🔧 Usando MLFLOW_TRACKING_URI de variable de entorno")
elif os.name == 'nt':  # Windows
    tracking_uri = "./mlruns"
else:  # Linux/Mac/GitHub Actions
    tracking_uri = f"file://{mlruns_dir.resolve()}"

mlflow.set_tracking_uri(tracking_uri)

print(f"📁 Workspace: {workspace_dir}")
print(f"📊 MLflow URI: {tracking_uri}")

experiment_name = "CI-CD-Lab2"

# --- Umbrales de validación ---
THRESHOLD_MSE = 0.6
THRESHOLD_R2 = 0.3

print(f"📏 Umbrales: MSE ≤ {THRESHOLD_MSE}, R² ≥ {THRESHOLD_R2}")

# --- Cargar dataset ---
data_path = workspace_dir / "data" / "winequality-red.csv"
if not data_path.exists():
    print(f"❌ No se encuentra {data_path}")
    print(f"   Directorio actual: {Path.cwd()}")
    print(f"   Archivos en workspace: {list(workspace_dir.iterdir())}")
    sys.exit(1)

print(f"📂 Cargando datos desde: {data_path}")
df = pd.read_csv(data_path, sep=',')
print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# --- Preprocesamiento IDÉNTICO a train.py ---
X = df.drop("quality", axis=1)
y = df["quality"]

# Manejo de valores nulos (mismo que train.py)
null_count = X.isnull().sum().sum()
if null_count > 0:
    print(f"⚠️  Rellenando {null_count} valores nulos")
    X.fillna(X.mean(), inplace=True)

# Eliminación de outliers con Z-score (mismo método que train.py)
z_scores = np.abs((X - X.mean()) / X.std())
mask = (z_scores < 3).all(axis=1)
X = X[mask]
y = y[X.index]
print(f"🧹 Outliers eliminados: {(~mask).sum()} filas")

# Eliminación de variables correlacionadas (mismo que train.py)
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
if to_drop:
    print(f"🔗 Variables correlacionadas eliminadas: {to_drop}")
    X.drop(columns=to_drop, inplace=True)

# Escalamiento (mismo que train.py)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(f"📏 Features escaladas: {X_scaled.shape[1]} variables")

# Split test (mismo que train.py)
_, X_test, _, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"📊 Test set: {X_test.shape[0]} muestras")

# --- Cargar último modelo registrado ---
print(f"\n🔍 Buscando experimento '{experiment_name}'...")
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    print(f"❌ Experimento '{experiment_name}' no encontrado.")
    print(f"   Asegúrate de haber ejecutado train.py primero.")
    sys.exit(1)

print(f"✅ Experimento encontrado (ID: {experiment.experiment_id})")

# Buscar la última run
client = MlflowClient()
runs = client.search_runs(
    experiment.experiment_id, 
    order_by=["start_time DESC"], 
    max_results=1
)

if not runs:
    print(f"❌ No hay runs registradas en el experimento '{experiment_name}'.")
    print(f"   Ejecuta train.py primero para entrenar un modelo.")
    sys.exit(1)

run_id = runs[0].info.run_id
print(f"🎯 Última run encontrada: {run_id}")

# Cargar el modelo
model_uri = f"runs:/{run_id}/model"
print(f"📦 Cargando modelo desde: {model_uri}")

try:
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    sys.exit(1)

# --- Predicción y evaluación ---
print(f"\n🔮 Realizando predicciones...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"📊 RESULTADOS DE VALIDACIÓN")
print(f"{'='*50}")
print(f"🔍 MSE: {mse:.4f} (umbral: ≤ {THRESHOLD_MSE})")
print(f"🔍 R²:  {r2:.4f} (umbral: ≥ {THRESHOLD_R2})")
print(f"{'='*50}\n")

# --- Validación de umbrales ---
mse_pass = mse <= THRESHOLD_MSE
r2_pass = r2 >= THRESHOLD_R2

print("📋 Verificación de criterios:")
print(f"   {'✅' if mse_pass else '❌'} MSE: {mse:.4f} {'≤' if mse_pass else '>'} {THRESHOLD_MSE}")
print(f"   {'✅' if r2_pass else '❌'} R²:  {r2:.4f} {'>=' if r2_pass else '<'} {THRESHOLD_R2}")
print()

# Registrar métricas de validación en MLflow
try:
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("validation_mse", mse)
        mlflow.log_metric("validation_r2", r2)
        mlflow.log_metric("mse_threshold", THRESHOLD_MSE)
        mlflow.log_metric("r2_threshold", THRESHOLD_R2)
        mlflow.set_tag("validation_status", "passed" if (mse_pass and r2_pass) else "failed")
    print(f"📝 Métricas de validación registradas en MLflow")
except Exception as e:
    print(f"⚠️  No se pudieron registrar métricas en MLflow: {e}")

# --- Resultado final ---
if mse_pass and r2_pass:
    print("✅ El modelo CUMPLE los criterios de calidad.")
    print("   El pipeline puede continuar.")
    sys.exit(0)
else:
    print("❌ El modelo NO CUMPLE los thresholds establecidos.")
    print("   Deteniendo pipeline.")
    if not mse_pass:
        print(f"   - MSE demasiado alto: {mse:.4f} > {THRESHOLD_MSE}")
    if not r2_pass:
        print(f"   - R² demasiado bajo: {r2:.4f} < {THRESHOLD_R2}")
    sys.exit(1)