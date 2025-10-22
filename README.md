\# 🚀 Proyecto Final MLOps: Pipeline de Machine Learning con CI/CD



Este proyecto implementa un \*\*pipeline reproducible de Machine Learning\*\* para predecir la calidad del vino tinto usando \*\*Random Forest\*\*, con \*\*MLflow\*\* para tracking y \*\*GitHub Actions\*\* para CI/CD.



---



## Estructura del Proyecto



mlflow-deploy/
├── data/
│ └── winequality-red.csv # Dataset externo (CSV)
├── src/
│ ├── train.py # Script de entrenamiento
│ └── validate.py # Script de validación
├── .github/workflows/
│ ├── mlflow-ci.yml # Workflow de CI
│ └── mlflow-cd.yml # Workflow de CD
├── Makefile # Comandos automáticos
├── requirements.txt # Dependencias
└── README.md


---



\## Objetivo



Automatizar un **pipeline reproducible de Machine Learning** que permita:



1\. Entrenar un modelo de regresión.

2\. Validar su desempeño automáticamente.

3\. Registrar parámetros, métricas y modelo en \*\*MLflow\*\*.

4\. Integrar \*\*CI/CD\*\* mediante GitHub Actions.

5\. Preparar el modelo para despliegue (simulado).



---



\## Requisitos



\- Python ≥ 3.10  

\- Git  

\- MLflow  

\- scikit-learn, pandas, joblib  

\- GitHub Actions habilitado en el repositorio  



Instalar dependencias locales:





make install



---



\## Dataset



Se utiliza el dataset \*\*Wine Quality - Red Wine\*\* (CSV externo).  

Link: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download

Columnas principales:



\- fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, quality



> Fuente: \[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)



---



\## Makefile



| Comando          | Descripción                                         |

|-----------------|-----------------------------------------------------|

| `make install`   | Instala todas las dependencias del proyecto         |

| `make train`     | Ejecuta el pipeline de entrenamiento completo      |

| `make validate`  | Valida el modelo usando MSE y R² definidos          |

| `make ci`        | Ejecuta `train` + `validate`                       |

| `make clean`     | Limpia artefactos locales (mlruns, \_\_pycache\_\_)    |



---



\## Ejecución Local



1\. Entrenar modelo:



make train



2\. Validar desempeño:



make validate



3\. Ejecutar pipeline completo:





make ci



4\. Limpiar artefactos:



make clean





---



\## CI/CD con GitHub Actions



\### CI - Entrenamiento y Validación

Archivo: `.github/workflows/mlflow-ci.yml`



\- Se ejecuta en cada push a `main`.

\- Pasos:

&nbsp; 1. Clonar repositorio.

&nbsp; 2. Configurar Python 3.10.

&nbsp; 3. Instalar dependencias (`make install`).

&nbsp; 4. Entrenar el modelo (`make train`).

&nbsp; 5. Validar el modelo (`make validate`).

&nbsp; 6.  Subir el modelo validado como artefacto (`mlruns/**/model`).



\### CD - Promoción y Despliegue (Simulado)

Archivo: `.github/workflows/mlflow-cd.yml`



\- Se ejecuta manualmente (`workflow\_dispatch`).

\- Pasos:

&nbsp; 1. Clonar repositorio.

&nbsp; 2. Configurar Python.

&nbsp; 3. Instalar dependencias.

&nbsp; 4. Descargar artefactos del CI (`modelo-validado`).

&nbsp; 5. Simular promoción a producción.

\- Opcional: servir modelo usando `mlflow models serve`.



---



\## MLflow Tracking



\- \*\*URI de tracking local\*\*: `file://./mlruns`

\- \*\*Experimento\*\*: `CI-CD-Lab2`

\- Métricas registradas:

&nbsp; - MSE

\- Artefactos registrados:

&nbsp; - Modelo RandomForest

\- Firma e input example inferidos automáticamente



---



## 🧪 Evidencia

- ✅ Modelo entrenado y validado exitosamente en **GitHub Actions CI**.  
- ✅ Artefactos subidos correctamente.  
- ❌ El modelo no fue promovido a producción debido a no cumplir el umbral de R² ≥ 0.4.  

> Ejemplo de ejecución: [GitHub Actions CI](https://github.com/JonathanC-hub/ProyectoFinalMLOPs/actions)




---



\## Notas



\- Todos los scripts se pueden ejecutar desde consola.

\- El flujo CI/CD permite \*\*reproducibilidad y validación automática\*\*.

\- Para producción real, se puede reemplazar la promoción simulada con `mlflow models serve` o integración con MLflow Registry.



---



\## Licencia



MIT License



