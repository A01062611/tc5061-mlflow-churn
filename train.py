"""Entrena un modelo de churn (Telco Customer Churn) y registra la corrida en MLflow."""
import argparse
import json
import random
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent  # raíz del repo, sin importar desde dónde se ejecute
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")  # relativa a BASE_DIR
TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"  # todas las corridas en la misma base
NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"


def parse_args() -> argparse.Namespace:
    """Lee hiperparámetros y configuración desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena LogisticRegression para predecir churn y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # muestra los defaults en --help
    )
    # Hiperparámetros del modelo
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverso de la fuerza de regularización (más chico = más regularización)")
    parser.add_argument("--l1-ratio", type=float, default=0.0,
                        help="Mezcla de penalización: 0 = L2, 1 = L1 (apaga variables), intermedio = elasticnet")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                        help="'balanced' pondera más la clase minoritaria (churn)")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="Tope de iteraciones del optimizador")
    # Datos y reproducibilidad
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proporción de datos reservada para test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para el split y el modelo (reproducibilidad)")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH,
                        help="Ruta al CSV de Telco Customer Churn (si es relativa, se toma desde la raíz del repo)")
    # MLflow
    parser.add_argument("--experiment-name", default="telco-churn-logreg-hparams",
                        help="Experimento de MLflow donde se agrupan las corridas")
    parser.add_argument("--run-name", default=None,
                        help="Nombre de la corrida; si se omite se genera con los hiperparámetros")

    args = parser.parse_args()
    if not 0 <= args.l1_ratio <= 1:
        parser.error("--l1-ratio debe estar entre 0 y 1")
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1 (exclusivo)")
    args.data_path = BASE_DIR / args.data_path  # si ya es absoluta, Path la deja igual
    return args


def set_seed(seed: int) -> None:
    """Fija los generadores aleatorios globales (red de seguridad).

    El split y el modelo reciben la semilla explícitamente vía random_state;
    esto cubre cualquier uso futuro de random / np.random.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_data(path: Path) -> pd.DataFrame:
    """Lee el CSV crudo sin transformarlo."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las decisiones de limpieza justificadas en notebooks/eda.ipynb."""
    df = df.copy()  # no modificar el DataFrame recibido
    # 11 blancos = clientes con tenure 0, aún sin facturar -> cargo acumulado real es 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    # Categórica codificada como 0/1: se pasa a No/Yes para el OneHotEncoder
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    # Target numérico (lo requieren sklearn y las métricas)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    # Identificador único: no generaliza
    return df.drop(columns="customerID")


def split_data(df: pd.DataFrame, test_size: float, seed: int):
    """Separa features/target y hace un split estratificado train/test."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    # stratify: ambos conjuntos conservan la proporción de churn (~26.5%)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def build_pipeline(num_features: list, cat_features: list, seed: int,
                   C: float = 1.0, l1_ratio: float = 0.0,
                   class_weight: str | None = None, max_iter: int = 1000) -> Pipeline:
    """Construye preprocesamiento + LogisticRegression (sin entrenar)."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ])
    # lbfgs solo soporta L2; cualquier componente L1 requiere saga
    solver = "lbfgs" if l1_ratio == 0 else "saga"
    classifier = LogisticRegression(
        C=C, l1_ratio=l1_ratio, class_weight=class_weight,
        solver=solver, max_iter=max_iter, random_state=seed,
    )
    return Pipeline([("preprocesamiento", preprocessor), ("clasificador", classifier)])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Calcula métricas de clasificación sobre un modelo ya entrenado."""
    pred = model.predict(X)               # clase 0/1 (umbral 0.5)
    proba = model.predict_proba(X)[:, 1]  # P(churn), para ROC-AUC
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def save_confusion_matrix(model: Pipeline, X: pd.DataFrame, y: pd.Series, path: Path) -> Path:
    """Dibuja la matriz de confusión del modelo y la guarda como PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=["No churn", "Churn"],
                                          cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("Matriz de confusión (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)  # libera memoria; en un script no hay quién la muestre
    return path


def setup_mlflow(experiment_name: str) -> None:
    """Apunta MLflow a la base local y selecciona (o crea) el experimento."""
    mlflow.set_tracking_uri(TRACKING_URI)
    # Al crear el experimento se fija dónde van sus artefactos; sin esto dependería del cwd
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=(BASE_DIR / "mlruns").as_uri())
    mlflow.set_experiment(experiment_name)


def log_config(args: argparse.Namespace, model: Pipeline, n_train: int, n_test: int) -> None:
    """Registra la configuración de la corrida: hiperparámetros, datos y tags."""
    params = {
        "C": args.C,
        "l1_ratio": args.l1_ratio,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,
        "solver": model.named_steps["clasificador"].solver,  # derivado de l1_ratio
        "seed": args.seed,
        "test_size": args.test_size,
        "data_file": args.data_path.name,
        "n_train": n_train,
        "n_test": n_test,
    }
    for name, value in params.items():
        mlflow.log_param(name, value)
    mlflow.set_tags({"autor": "Alan Hernández", "modelo": "LogisticRegression",
                     "metrica_principal": "f1"})


def log_results(model: Pipeline, metrics: dict, X_train: pd.DataFrame,
                X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Registra métricas, artefactos y el modelo entrenado en la corrida activa."""
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    # log_artifact sube archivos que ya existen: se generan en una carpeta temporal
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mlflow.log_artifact(save_confusion_matrix(model, X_test, y_test, tmp / "confusion_matrix.png"))
        metrics_path = tmp / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(metrics_path)

    # Se guarda el Pipeline completo: recibe datos crudos y devuelve 0/1
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, name="model", signature=signature,
                             input_example=X_train.head(3))


def main():
    args = parse_args()
    set_seed(args.seed)
    class_weight = None if args.class_weight == "none" else args.class_weight  # la CLI solo entrega texto
    setup_mlflow(args.experiment_name)

    df = clean_data(load_data(args.data_path))
    X_train, X_test, y_train, y_test = split_data(df, test_size=args.test_size, seed=args.seed)
    cat_features = [c for c in X_train.columns if c not in NUM_FEATURES]

    model = build_pipeline(NUM_FEATURES, cat_features, seed=args.seed,
                           C=args.C, l1_ratio=args.l1_ratio,
                           class_weight=class_weight, max_iter=args.max_iter)
    run_name = args.run_name or f"C={args.C}_cw={args.class_weight}_l1={args.l1_ratio}"

    # La corrida se abre después de cargar datos (si el CSV falla, no queda un run vacío)
    # y antes de entrenar (si el fit falla, el run queda FAILED con sus params).
    with mlflow.start_run(run_name=run_name) as run:
        log_config(args, model, n_train=len(X_train), n_test=len(X_test))
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        log_results(model, metrics, X_train, X_test, y_test)

    print(f"Run '{run_name}' ({run.info.run_id}) en experimento '{args.experiment_name}'")
    for name, value in metrics.items():
        print(f"  {name:10s} {value:.3f}")


if __name__ == "__main__":
    main()
