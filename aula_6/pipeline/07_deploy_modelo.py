# Databricks notebook source
# ==============================================================================
%pip install databricks mlflow lightgbm pandas pyarrow
dbutils.library.restartPython()

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
import mlflow
import mlflow.lightgbm
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 1. PARÂMETROS DINÂMICOS DO WORKFLOW E DATAS
# ==========================================
env_catalog = dbutils.widgets.get("env_catalog")

p_fim_original = dbutils.widgets.get("data_fim")
data_obj = datetime.strptime(p_fim_original, "%Y%m%d")
p_fim = (data_obj - timedelta(days=1)).strftime("%Y%m%d")

esquema_gold = "gold"

nome_modelo = f"{env_catalog}.{esquema_gold}.propensao_compra_modelo"
tabela_entrada = f"{env_catalog}.{esquema_gold}.base_tabela_prod"
tabela_saida = f"{env_catalog}.{esquema_gold}.escoragem_propensao_compra"

print(f"Iniciando Inferência em Lote (Batch Scoring).")
print(f"Data recebida do Job: {p_fim_original} | Data processada (D-1): {p_fim}")
print(f"Alvo: Tabela {tabela_saida}")

# ==========================================
# 2. CARREGAMENTO DOS DADOS NOVOS
# ==========================================
df_novos_dados = spark.table(tabela_entrada).filter(
    F.col("dia_prtc") == p_fim
)

qtd_dados = df_novos_dados.count()
if qtd_dados == 0:
    dbutils.notebook.exit(f"Processo encerrado: Nenhum dado novo encontrado para escoragem no dia {p_fim}.")

print(f"Foram encontrados {qtd_dados} registros para receberem o score.")

# Isolando as features
colunas_excluidas = ["dia_prtc", "id_unico", "comprou_eletronico"]
features_modelo = [c for c in df_novos_dados.columns if c not in colunas_excluidas]

# ==========================================
# 3. WORKAROUND SERVERLESS: CLOSURE PANDAS UDF
# ==========================================
# 1. Baixa o modelo no DRIVER (Ignora o bloqueio do Unity Catalog nos workers)
model_uri = f"models:/{nome_modelo}@Champion"
print(f"Baixando o modelo no Driver: {model_uri}")

# Carrega o modelo LightGBM para a memória do nó principal
modelo_local = mlflow.lightgbm.load_model(model_uri)

# 2. Criamos nossa própria UDF vetorizada em Pandas
# O PySpark empacota a variável 'modelo_local' do escopo externo via Closure
@F.pandas_udf(DoubleType())
def predict_udf_custom(*cols: pd.Series) -> pd.Series:
    import pandas as pd 
    
    # Reconstrói as colunas em um DataFrame Pandas
    df_features = pd.concat(cols, axis=1)
    df_features.columns = features_modelo
    
    # O modelo_local é chamado diretamente aqui de dentro.
    # O LightGBM retorna nativamente um array 1D com as probabilidades
    preds = modelo_local.predict(df_features)
    
    return pd.Series(preds)

# ==========================================
# 4. APLICAÇÃO DO MODELO E REGRAS DE NEGÓCIO
# ==========================================
print("Aplicando o algoritmo aos dados produtivos...")

# Injetamos a lista dinâmica de colunas na UDF personalizada
df_escorado = df_novos_dados.withColumn(
    "score_propensao", 
    predict_udf_custom(*[F.col(c) for c in features_modelo])
)

# Criação da flag de negócio (probabilidade >= 50% = 1)
df_escorado = df_escorado.withColumn(
    "flag_propensao_alta",
    F.when(F.col("score_propensao") >= 0.5, 1).otherwise(0)
)

# Seleção de campos para entrega
df_final_saida = df_escorado.select(
    "id_unico",
    "dia_prtc",
    "score_propensao",
    "flag_propensao_alta",
    F.current_timestamp().alias("timestamp_escoragem")
)

# ==========================================
# 5. SALVAMENTO DOS RESULTADOS
# ==========================================
print("Gravando previsões na tabela final...")

df_final_saida.write \
    .format("delta") \
    .option("partitionOverwriteMode", "dynamic") \
    .partitionBy("dia_prtc") \
    .mode("overwrite") \
    .saveAsTable(tabela_saida)

print(f"✅ Processo concluído! Os scores já estão disponíveis em {tabela_saida}.")
