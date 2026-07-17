# Databricks notebook source
# ==============================================================================
%pip install databricks mlflow lightgbm
dbutils.library.restartPython()

import pyspark.sql.functions as F
import mlflow
from datetime import datetime, timedelta

# ==========================================
# 1. PARÂMETROS DINÂMICOS DO WORKFLOW E DATAS
# ==========================================
env_catalog = dbutils.widgets.get("env_catalog")

# 1. Pega a data exata passada pelo Job (ex: "20260110")
p_fim_original = dbutils.widgets.get("data_fim")

# 2. Converte para data, tira 1 dia e volta para string (ex: vira "20260109")
data_obj = datetime.strptime(p_fim_original, "%Y%m%d")
p_fim = (data_obj - timedelta(days=1)).strftime("%Y%m%d")

esquema_gold = "gold"

# Nomenclaturas
nome_modelo = f"{env_catalog}.{esquema_gold}.propensao_compra_modelo"
tabela_entrada = f"{env_catalog}.{esquema_gold}.base_tabela_prod"
tabela_saida = f"{env_catalog}.{esquema_gold}.escoragem_propensao_compra"

print(f"Iniciando Inferência em Lote (Batch Scoring).")
print(f"Data recebida do Job: {p_fim_original} | Data processada (D-1): {p_fim}")
print(f"Alvo: Tabela {tabela_saida}")

# ==========================================
# 2. CARREGAMENTO DOS DADOS NOVOS
# ==========================================
# Filtro exato para rodar apenas a partição do dia correspondente (D-1)
df_novos_dados = spark.table(tabela_entrada).filter(
    F.col("dia_prtc") == p_fim
)

qtd_dados = df_novos_dados.count()
if qtd_dados == 0:
    dbutils.notebook.exit(f"Processo encerrado: Nenhum dado novo encontrado para escoragem no dia {p_fim}.")

print(f"Foram encontrados {qtd_dados} registros para receberem o score.")

# ==========================================
# 3. CARREGAMENTO DO MODELO @CHAMPION
# ==========================================
# A URI do Unity Catalog para consumir a versão em produção automaticamente
model_uri = f"models:/{nome_modelo}@Champion"
print(f"Carregando o modelo preditivo: {model_uri}")

# Criação da UDF (User Defined Function) do Spark com o env_manager corrigido
predict_udf = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri, 
    result_type="double",
    env_manager="local" # <-- Trava que corrige o erro do InvalidVersion
)

# Isolando as features usando a mesma lógica do treinamento para não quebrar o schema
colunas_excluidas = ["dia_prtc", "id_unico", "comprou_eletronico"]
features_modelo = [c for c in df_novos_dados.columns if c not in colunas_excluidas]

# ==========================================
# 4. APLICAÇÃO DO MODELO E REGRAS DE NEGÓCIO
# ==========================================
print("Aplicando o algoritmo aos dados produtivos...")

# Passamos a lista de colunas dinamicamente para a UDF
df_escorado = df_novos_dados.withColumn(
    "score_propensao", 
    predict_udf(*[F.col(c) for c in features_modelo])
)

# Criação de uma flag de negócio (Exemplo: probabilidade acima de 50% = alta propensão)
df_escorado = df_escorado.withColumn(
    "flag_propensao_alta",
    F.when(F.col("score_propensao") >= 0.5, 1).otherwise(0)
)

# Selecionamos apenas o necessário para entregar à área de negócio
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
