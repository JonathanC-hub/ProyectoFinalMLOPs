import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mlflow.models import infer_signature
import sys
import traceback

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
artifact_location = "file://" + os.path.abspath(mlruns_dir)

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

os.makedirs(mlruns_dir, exist_ok=True)

# --- Configurar MLflow ---
mlflow.set_tracking_uri(tracking_uri)

# --- Crear o Establecer Experimento ---
experiment_name = "CI-CD-Lab2"
experiment_id = None
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
    print(f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: ID del Experimento Existente: {experiment_id} ---")
        else:
            print(f"--- ERROR: No se pudo obtener el experimento existente ---")
            sys.exit(1)
    else:
        raise e

if experiment_id is None:
    print(f"--- ERROR FATAL: No se pudo obtener un ID de experimento válido ---")
    sys.exit(1)

# --- Cargar dataset Wine (Red Wine Quality) ---
data_path = os.path.join(workspace_dir, "data", "winequality-red.csv")
if not os.path.exists(data_path):
    print(f"--- ERROR: No se encuentra el archivo {data_path} ---")
    sys.exit(1)

df = pd.read_csv(data_path, sep=',')

# --- Preprocesamiento ---
X = df.drop("quality", axis=1)
y = df["quality"]

# Manejo de valores nulos
if X.isnull().sum().sum() > 0:
    X.fillna(X.mean(), inplace=True)
    print("--- Debug: Se llenaron valores nulos con la media ---")

# Escalamiento de features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Entrenamiento del modelo ---
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# --- Evaluación ---
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"🔍 MSE: {mse:.4f}, R²: {r2:.4f}")

# --- MLflow Run ---
print(f"--- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---")
run = None
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"--- Debug: Run ID: {run_id} ---")
        
        # Log métricas
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Inferir firma de entrada
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Registrar modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        print(f"✅ Modelo registrado correctamente. MSE: {mse:.4f}, R²: {r2:.4f}")

except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    sys.exit(1)

