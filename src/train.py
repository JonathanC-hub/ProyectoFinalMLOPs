import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature

# --- Paths ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
os.makedirs(mlruns_dir, exist_ok=True)

# --- Tracking local compatible Windows ---
tracking_uri = f"file:///{mlruns_dir.replace(os.sep, '/')}"  # compatible Windows/Linux
mlflow.set_tracking_uri(tracking_uri)

# --- Experimento ---
experiment_name = "CI-CD-Lab2"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# --- Cargar dataset ---
data_path = os.path.join(workspace_dir, "data", "winequality-red.csv")
df = pd.read_csv(data_path, sep=',')

X = df.drop("quality", axis=1)
y = df["quality"]

# --- Preprocesamiento ---
# Llenar nulos con media
if X.isnull().sum().sum() > 0:
    X.fillna(X.mean(), inplace=True)

# Escalamiento de features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --- División de datos ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Modelo ---
model = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Métricas ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

# --- Log en MLflow ---
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train.iloc[:5])

print("✅ Entrenamiento local completado con preprocesamiento.")
