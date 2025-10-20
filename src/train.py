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

# --- Configuración de entorno ---
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# --- Paths y MLflow ---
if IS_GITHUB_ACTIONS:
    # En GitHub Actions, usar rutas relativas al workspace
    workspace_dir = Path(os.getenv("GITHUB_WORKSPACE", "."))
else:
    # En local, usar el directorio actual
    workspace_dir = Path.cwd()

mlruns_dir = workspace_dir / "mlruns"
mlruns_dir.mkdir(exist_ok=True)

# MLflow requiere formato específico según el OS
# También verificar si ya hay una variable de entorno configurada
if os.getenv("MLFLOW_TRACKING_URI"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    print(f"🔧 Usando MLFLOW_TRACKING_URI de variable de entorno")
elif os.name == 'nt':  # Windows
    # Usar ruta relativa simple
    tracking_uri = "./mlruns"
else:  # Linux/Mac/GitHub Actions
    # Usar ruta relativa en lugar de absoluta para evitar problemas de permisos
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
    print(f"   Directorio actual: {Path.cwd()}")
    print(f"   Archivos en workspace: {list(workspace_dir.iterdir())}")
    sys.exit(1)

print(f"📂 Cargando datos desde: {data_path}")
df = pd.read_csv(data_path, sep=",")
print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# --- Preprocesamiento ---
X = df.drop("quality", axis=1)
y = df["quality"]

# Manejo de valores nulos
null_count = X.isnull().sum().sum()
if null_count > 0:
    print(f"⚠️  Rellenando {null_count} valores nulos")
    X.fillna(X.mean(), inplace=True)

# Eliminación de outliers (Z-score > 3)
z_scores = np.abs((X - X.mean()) / X.std())
mask = (z_scores < 3).all(axis=1)
X = X[mask]
y = y[X.index]
print(f"🧹 Outliers eliminados: {(~mask).sum()} filas")

# Eliminación de variables altamente correlacionadas (>0.9)
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
if to_drop:
    print(f"🔗 Variables correlacionadas eliminadas: {to_drop}")
    X.drop(columns=to_drop, inplace=True)

# Escalamiento
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

# --- MLflow ---
try:
    with mlflow.start_run(experiment_id=experiment_id):
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("best_r2_cv", grid.best_score_)
        
        # Log parameters
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Log model
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        run_id = mlflow.active_run().info.run_id
        print(f"✅ Modelo registrado en MLflow (Run ID: {run_id})")
        
        # Guardar run_id para uso posterior (útil en CI/CD)
        if IS_GITHUB_ACTIONS:
            with open("run_id.txt", "w") as f:
                f.write(run_id)
            print(f"💾 Run ID guardado en run_id.txt")

except Exception as e:
    print(f"❌ Error al registrar en MLflow: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Entrenamiento completado exitosamente")
sys.exit(0)