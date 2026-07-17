# Databricks notebook source
# Databricks notebook source
# ==============================================================================
## instalando os pacotes
%pip install databricks lightgbm
%pip install databricks-feature-engineering
dbutils.library.restartPython()

import pandas as pd
import pyspark.sql.functions as F
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# ==========================================
# 1. PARÂMETROS DINÂMICOS DO WORKFLOW
# ==========================================
env_catalog = dbutils.widgets.get("env_catalog")
p_inicio = dbutils.widgets.get("data_inicio")
p_fim = dbutils.widgets.get("data_fim")

esquema_gold = "gold"

# Ajustando nomes com base no ambiente (dev, homlog, prod)
model_name = f"{env_catalog}.{esquema_gold}.propensao_compra_modelo"
nome_tabela_prod = f"{env_catalog}.{esquema_gold}.base_tabela_prod"
target = "comprou_eletronico"

print(f"Iniciando re-treinamento usando dados de {p_inicio} a {p_fim} no catálogo {env_catalog}...")

# ==========================================
# 2. CARREGAMENTO DA NOVA REALIDADE (DRIFT)
# ==========================================
df_spark = spark.table(nome_tabela_prod).filter(
    F.col("dia_prtc").between(p_inicio, p_fim)
)

df = df_spark.toPandas()

# Isolando apenas as features (removendo target e metadados)
colunas_excluidas = ["dia_prtc", "id_unico", target]
features = [c for c in df.columns if c not in colunas_excluidas]

X = df[features]
y = df[target]

# Dividindo em treino e validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. TREINAMENTO E REGISTRO MLFLOW
# ==========================================
mlflow.lightgbm.autolog()

with mlflow.start_run(run_name="retreino_correcao_drift_lgbm") as run:
    
    print("Treinando o modelo LightGBM atualizado...")
    
    # Instanciando o classificador LGBM
    modelo_lgbm = lgb.LGBMClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42,
        objective='binary',
        verbosity=-1
    )
  
    modelo_lgbm.fit(X_train, y_train)

    # Validação e registro da métrica principal
    probs = modelo_lgbm.predict_proba(X_test)[:, 1]
    signature = infer_signature(X_train, probs)
    auc = roc_auc_score(y_test, probs)
    
    mlflow.log_metric("val_roc_auc", auc)
    
    print(f"Registrando a nova versão no Model Registry ({model_name})...")
    model_info = mlflow.lightgbm.log_model(
        lgb_model=modelo_lgbm,
        artifact_path="modelo",
        registered_model_name=model_name,
        signature=signature
    )

print(f"Nova versão do modelo registrada com sucesso! AUC: {auc:.4f}")

# ==========================================
# 4. PROMOÇÃO A CHAMPION
# ==========================================
client = MlflowClient()

nova_versao = model_info.registered_model_version
print(f"Promovendo a Versão {nova_versao} para a tag @Champion...")

# Removemos o Alias antigo por segurança
try:
    client.delete_registered_model_alias(model_name, "Champion")
except Exception:
    pass

# Aplicamos o alias na nova versão
client.set_registered_model_alias(model_name, "Champion", nova_versao)

print(f"Concluído! A inferência em produção usará automaticamente a Versão {nova_versao} do LightGBM.")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from dev.gold.base_tabela_prod