# Makefile para pipeline MLflow CI/CD

# --- Variables ---
PYTHON := python
REQ := requirements.txt

# --- Instalación de dependencias ---
install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r $(REQ)
	pip install mlflow joblib scikit-learn pandas

# --- Ejecutar entrenamiento ---
train:
	@echo "🚀 Iniciando entrenamiento..."
	$(PYTHON) src/train.py
	@echo "✅ Entrenamiento finalizado."

# --- Ejecutar validación ---
validate:
	@echo "🔍 Iniciando validación..."
	$(PYTHON) src/validate.py
	@echo "✅ Validación finalizada."

# --- Ejecutar flujo completo (train + validate) ---
ci: train validate
	@echo "🎯 CI completo: entrenamiento y validación finalizados."

# --- Limpiar artefactos locales ---
clean:
	@echo "🧹 Limpiando artefactos locales..."
	rm -rf mlruns __pycache__
	@echo "🗑 Limpieza completa."
