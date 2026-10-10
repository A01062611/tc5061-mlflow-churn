# Telco Customer Churn — experimentación con MLflow

Script de entrenamiento parametrizado que predice el abandono de clientes (`Churn`) con una Regresión Logística y registra cada corrida (parámetros, métricas, artefactos y modelo) en MLflow Tracking.

**Curso:** TC5061 · **Autor:** Alan Arturo Hernández Chichitz (A01062611)

## Estructura

```
├── train.py              # Script de entrenamiento: datos → modelo → evaluación → MLflow
├── Makefile              # Atajos: instalar, entrenar, correr experimentos, abrir la UI
├── requirements.txt      # Dependencias con versiones fijadas
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco Customer Churn (Kaggle)
└── notebooks/
    └── eda.ipynb         # Solo exploración de datos y justificación de la limpieza
```

## Requisitos

- Python **3.14**
- `make` (opcional; ver [Sin make](#sin-make))

## Instalación

```bash
git clone https://github.com/A01062611/tc5061-mlflow-churn.git
cd tc5061-mlflow-churn
python3.14 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Una corrida con los valores por defecto (reproduce el baseline):

```bash
python train.py
```

Cambiando hiperparámetros, sin editar el código:

```bash
python train.py --C 0.1 --class-weight balanced --seed 42
```

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | `1.0` | Inverso de la fuerza de regularización (más chico = más regularización) |
| `--l1-ratio` | `0.0` | Tipo de penalización: `0` = L2, `1` = L1, intermedio = elasticnet |
| `--class-weight` | `none` | `balanced` pondera más la clase minoritaria (churn) |
| `--max-iter` | `1000` | Tope de iteraciones del optimizador |
| `--test-size` | `0.2` | Proporción de datos para test |
| `--seed` | `42` | Semilla del split y del modelo |
| `--data-path` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | CSV de entrada (si es relativa, desde la raíz del repo) |
| `--experiment-name` | `telco-churn-logreg-hparams` | Experimento de MLflow |
| `--run-name` | automático | Nombre de la corrida; por defecto se arma con los hiperparámetros |

Ayuda completa: `python train.py --help`.

Cada corrida registra en MLflow:
- **Parámetros:** hiperparámetros, solver, semilla y tamaños de train/test.
- **Métricas:** accuracy, precision, recall, f1 (principal) y roc_auc sobre test.
- **Artefactos:** `confusion_matrix.png` y `metrics.json`.
- **Modelo:** el Pipeline completo (preprocesamiento + clasificador) con su signature.

## Experimentos

```bash
make experiments
```

Ejecuta las 7 corridas del experimento `telco-churn-logreg-hparams`:

| Ronda | # | Configuración | Pregunta |
|---|---|---|---|
| 1 | 1 | defaults | Baseline |
| 1 | 2 | `--C 0.01` | ¿Ayuda regularizar mucho? |
| 1 | 3 | `--C 100` | ¿Y casi no regularizar? |
| 1 | 4 | `--class-weight balanced` | ¿Efecto de balancear clases? |
| 1 | 5 | `--l1-ratio 1` | ¿Efecto de L1? |
| 2 | 6 | `--class-weight balanced --C 0.1` | ¿Mejora la mejor configuración ajustando C? |
| 2 | 7 | `--class-weight balanced --l1-ratio 1` | ¿Hay interacción entre L1 y balanced? |

La ronda 1 cambia un hiperparámetro a la vez respecto al baseline; la ronda 2 parte de la mejor configuración de la ronda 1. También se pueden correr por separado con `make round1` y `make round2`.

## Ver resultados

```bash
make ui
```

Abrir http://127.0.0.1:5000 → experimento `telco-churn-logreg-hparams` → seleccionar corridas → **Compare**.

Las corridas se guardan en `mlflow.db` y los artefactos en `mlruns/`, ambos en la raíz del repo (no se versionan). `make clean` los borra; después de hacerlo hay que reiniciar `make ui`.

## Reproducibilidad

- **Semilla:** `--seed` se pasa explícitamente a `train_test_split` y a `LogisticRegression`, y además fija `random` y `numpy`. Mismos argumentos → mismas métricas.
- **Dependencias:** versiones fijadas en `requirements.txt`.
- **Verificación:** `make reproducibility` corre el baseline dos veces en el experimento `telco-churn-reproducibilidad`; las métricas deben coincidir exactamente.

## Sin make

Equivalentes directos de cada comando del Makefile:

| `make` | Comando |
|---|---|
| `make install` | `pip install -r requirements.txt` |
| `make train ARGS="..."` | `python train.py ...` |
| `make experiments` | Los 7 comandos `python train.py ...` de la tabla de [Experimentos](#experimentos) |
| `make ui` | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| `make clean` | Borrar `mlflow.db` y `mlruns/` |

Si el venv está en otra ubicación: `make <target> PYTHON=ruta/a/python`.
