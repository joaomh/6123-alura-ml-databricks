# Databricks notebook source
# Databricks notebook source
# ==============================================================================
%pip install databricks
%pip install xgboost lightgbm optuna -q
%pip install shap -q
%pip install databricks-feature-engineering
dbutils.library.restartPython()

import optuna
import pandas as pd
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

# Imports dos algoritmos
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PARÂMETROS DE AMBIENTE E DATAS
# ==========================================
env_catalog = dbutils.widgets.get("env_catalog")
esquema_gold = "gold"

# ==========================================
# 2. CARREGAMENTO DOS DADOS (Feature Lookup)
# ==========================================
tabela_target = f"{env_catalog}.{esquema_gold}.target_compras_eletronicos"
tabela_features = f"{env_catalog}.{esquema_gold}.user_profile_features_temporal"

df_labels = spark.table(tabela_target)

fe = FeatureEngineeringClient()

lookups = [
    FeatureLookup(
        table_name=tabela_features,
        lookup_key="id_usuario",
        timestamp_lookup_key="timestamp_registro"
    )
]

training_set = fe.create_training_set(
    df=df_labels,
    feature_lookups=lookups,
    label="comprou_eletronico",
    exclude_columns=["id_usuario", "timestamp_registro", "dia_prtc"]
)

print("Convertendo o Training Set para Pandas...")
df_pandas = training_set.load_df().toPandas()

X = df_pandas.drop(columns=["comprou_eletronico"])
y = df_pandas["comprou_eletronico"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Volume de Treino: {len(X_train)} linhas")
print(f"Volume de Teste: {len(X_test)} linhas\n")

# ==========================================
# 3. TREINAMENTO DOS MODELOS BASELINE
# ==========================================
print("Iniciando treinamento dos modelos baseline...")

modelos_baseline = {
    "Logistic_Regression_Base": LogisticRegression(max_iter=1000, random_state=42),
    "Random_Forest_Base": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
    "Gradient_Boosting_Base": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "XGBoost_Base": XGBClassifier(n_estimators=50, max_depth=5, random_state=42, eval_metric='logloss'),
    "LightGBM_Base": LGBMClassifier(n_estimators=50, max_depth=5, random_state=42, verbosity=-1)
}

for nome_modelo, classificador in modelos_baseline.items():
    with mlflow.start_run(run_name=nome_modelo):
        classificador.fit(X_train, y_train)
        
        y_pred_proba = classificador.predict_proba(X_test)[:,1]
        y_pred_class = classificador.predict(X_test)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        acc_score = accuracy_score(y_test, y_pred_class)
        
        mlflow.log_param("algoritmo", classificador.__class__.__name__)
        mlflow.log_metric("auc_roc", auc_score)
        mlflow.log_metric("accuracy", acc_score)
        
        print(f"{nome_modelo} -> AUC-ROC: {auc_score:.4f} | Acurácia: {acc_score:.4f}")

# ==========================================
# 4. OPTIMIZAÇÃO COM OPTUNA (LIGHTGBM)
# ==========================================
def objetivo_optuna_lgbm(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 12)
    num_leaves = trial.suggest_int('num_leaves', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 50)
    
    with mlflow.start_run(run_name=f"Optuna_LGBM_Trial_{trial.number}", nested=True):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("num_leaves", num_leaves)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("min_child_samples", min_child_samples)
        
        clf = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            random_state=42,
            verbosity=-1
        )
        
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_metric("AUC-ROC", auc)
        return auc

print("\nIniciando busca refinada de hiperparâmetros para o LightGBM com Optuna...")
estudo_lgbm = optuna.create_study(direction="maximize")

with mlflow.start_run(run_name="Optuna_LGBM"):
    estudo_lgbm.optimize(objetivo_optuna_lgbm, n_trials=100)

print("\nTuning do LightGBM concluído!")
print(f"Melhor AUC-ROC obtida: {estudo_lgbm.best_value:.4f}")

# ==========================================
# 5. TREINO FINAL E REGISTRO NO UNITY CATALOG
# ==========================================
# Usando a mesma nomenclatura baseada no env_catalog dinâmico
nome_modelo_final = f"{env_catalog}.{esquema_gold}.propensao_compra_modelo"

melhores_parametros = estudo_lgbm.best_params
print(f"\nMelhores parâmetros: {melhores_parametros}")

melhor_lgbm = LGBMClassifier(**melhores_parametros, random_state=42, verbosity=-1)
melhor_lgbm.fit(X_train, y_train)

y_proba_classe_1 = melhor_lgbm.predict_proba(X_test)[:, 1]
signature = infer_signature(X_train, y_proba_classe_1)

auc = roc_auc_score(y_test, y_proba_classe_1)
acc = accuracy_score(y_test, melhor_lgbm.predict(X_test))

with mlflow.start_run(run_name="Modelo_Final_LGBM_Production"):
    for param_name, param_val in melhores_parametros.items():
        mlflow.log_param(param_name, param_val)
    
    mlflow.log_metric("AUC-ROC", auc)
    mlflow.log_metric("Accuracy", acc)
    
    print(f"Registrando o modelo no Unity Catalog como: {nome_modelo_final}...")
    
    model_info = mlflow.lightgbm.log_model(
        lgb_model=melhor_lgbm,
        artifact_path="model",
        registered_model_name=nome_modelo_final,
        signature=signature
    )
    
print(f"Nova versão do modelo registrada com sucesso! AUC Final: {auc:.4f}")

# ==========================================
# 6. PROMOÇÃO A CHAMPION
# ==========================================
client = MlflowClient()

nova_versao = model_info.registered_model_version
print(f"Promovendo a Versão {nova_versao} para a tag @Champion...")

try:
    client.delete_registered_model_alias(nome_modelo_final, "Champion")
except Exception:
    pass

client.set_registered_model_alias(nome_modelo_final, "Champion", nova_versao)
print(f"Concluído! A inferência em produção usará automaticamente a Versão {nova_versao} do LightGBM.")