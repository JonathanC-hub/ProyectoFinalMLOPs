# Makefile simplificado

# Ejecutar entrenamiento
train:
	python src/train.py

# Ejecutar validación
validate:
	python src/validate.py

# Ejecutar flujo completo (train + validate)
ci: train validate
	@echo "✅ CI completo: entrenamiento y validación finalizados."

# Limpiar artefactos locales
clean:
	rm -rf mlruns __pycache__
	@echo "🗑 Limpieza completa."
