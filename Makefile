# Comandos del proyecto. Uso: make <target>
# Con un venv fuera del repo: make <target> PYTHON=ruta/a/python
PYTHON ?= python
ARGS ?=
export MLFLOW_DISABLE_AGENT_HINT := 1  # silencia un aviso informativo de MLflow 3.x

.PHONY: help install train experiments round1 round2 reproducibility ui clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-z0-9]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

install:  ## Instala las dependencias
	$(PYTHON) -m pip install -r requirements.txt

train:  ## Una corrida; argumentos con ARGS="--C 0.1 --class-weight balanced"
	$(PYTHON) train.py $(ARGS)

experiments: round1 round2  ## Ejecuta todas las corridas del experimento (ronda 1 + 2)

# Ronda 1 — diseño OFAT: cada corrida cambia UN hiperparámetro respecto al baseline
round1:  ## Ronda 1: 5 corridas, un hiperparámetro a la vez
	$(PYTHON) train.py                          # 1. Baseline (= notebook Act. 2)
	$(PYTHON) train.py --C 0.01                 # 2. Regularización fuerte
	$(PYTHON) train.py --C 100                  # 3. Regularización casi nula
	$(PYTHON) train.py --class-weight balanced  # 4. Clases balanceadas
	$(PYTHON) train.py --l1-ratio 1             # 5. Penalización L1

# Ronda 2 — se parte de la mejor de la ronda 1 (balanced) y se combina con otro hiperparámetro
round2:  ## Ronda 2: combinaciones sobre la mejor configuración (balanced)
	$(PYTHON) train.py --class-weight balanced --C 0.1         # 6. Hueco de C entre 0.01 y 1
	$(PYTHON) train.py --class-weight balanced --l1-ratio 1    # 7. Interacción L1 + balanced

# Mismo comando dos veces en un experimento aparte: las métricas deben ser idénticas
reproducibility:  ## Corre el baseline 2 veces para verificar que da lo mismo
	$(PYTHON) train.py --experiment-name telco-churn-reproducibilidad --run-name repeticion-1
	$(PYTHON) train.py --experiment-name telco-churn-reproducibilidad --run-name repeticion-2

ui:  ## Abre la UI de MLflow en http://127.0.0.1:5000
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:  ## Borra las corridas locales (mlflow.db y mlruns/)
	rm -rf mlflow.db mlruns
