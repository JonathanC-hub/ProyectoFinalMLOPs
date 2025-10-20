import os
import sys
import traceback
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib

# --- Configuración de entorno ---
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# --- Paths y MLflow ---
if IS_GITHUB_ACTIONS:
    workspace_dir = Path(os.getenv("GITHUB_WORKSPACE", "."))
else:
    workspace_dir = Path.cwd()

mlruns_dir = workspace_dir / "mlruns"
mlruns_dir.mkdir(exist_ok=True)

# --- Configurar MLflow ---
if os.getenv("MLFLOW_TRACKING_URI"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    print(f"🔧 Usando MLFLOW_TRACKING_URI de variable de entorno")
elif os.name == 'nt':  # Windows
    tracking_uri = "./mlruns"
else:
    tracking_uri = "./mlruns"

mlflow.set_tracking_uri(tracking_uri)
print(f"📁 Workspace: {workspace_dir}")
print(f"📊 MLflow URI: {tracking_uri}")

experiment_name = "CI-CD-Lab2"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"✨ Experimento '{experiment_name}' creado")
else:
    experiment_id = experiment.experiment_id
    print(f"📂 Usando experimento existente: {experiment_name}")

# --- Cargar dataset ---
data_path = workspace_dir / "data" / "winequality-red.csv"
if not data_path.exists():
    print(f"❌ No se encuentra {data_path}")
    sys.exit(1)

print(f"📂 Cargando datos desde: {data_path}")
df = pd.read_csv(data_path, sep=",")
print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# --- Preprocesamiento ---
X = df.drop("quality", axis=1)
y = df["quality"]

# Manejo de valores nulos
if X.isnull().sum().sum() > 0:
    print(f"⚠️  Rellenando valores nulos")
    X.fillna(X.mean(), inplace=True)

# Eliminación de outliers
z_scores = np.abs((X - X.mean()) / X.std())
mask = (z_scores < 3).all(axis=1)
X, y = X[mask], y[mask]
print(f"🧹 Outliers eliminados: {(~mask).sum()} filas")

# Variables altamente correlacionadas
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
if to_drop:
    print(f"🔗 Variables correlacionadas eliminadas: {to_drop}")
    X.drop(columns=to_drop, inplace=True)

# Escalamiento (entrenar scaler aquí)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(f"📏 Features escaladas: {X_scaled.shape[1]} variables")

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"📊 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# --- GridSearchCV + Cross Validation ---
print("\n🔍 Iniciando GridSearchCV...")
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"🏆 Mejores hiperparámetros: {grid.best_params_}")
print(f"🔍 Mejor R² CV: {grid.best_score_:.4f}")
print(f"🔍 Test R²: {r2:.4f}")
print(f"🔍 Test MSE: {mse:.4f}")
print(f"{'='*50}\n")

# --- Registrar en MLflow ---
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id

        # Log de métricas
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("best_r2_cv", grid.best_score_)

        # Log de parámetros
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_features", X_train.shape[1])

        # Log del modelo
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )

        # Guardar scaler
        scaler_path = workspace_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path), artifact_path="scaler")
        print(f"✅ Scaler registrado en MLflow")

        if IS_GITHUB_ACTIONS:
            with open("run_id.txt", "w") as f:
                f.write(run_id)
            print(f"💾 Run ID guardado en run_id.txt")

        print(f"✅ Modelo registrado en MLflow (Run ID: {run_id})")

except Exception as e:
    print(f"❌ Error al registrar en MLflow: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Entrenamiento completado exitosamente")
sys.exit(0)
