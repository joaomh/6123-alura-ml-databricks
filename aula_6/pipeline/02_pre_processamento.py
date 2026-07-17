# Databricks notebook source
# Databricks notebook source
# ==============================================================================
%pip install databricks-feature-engineering
dbutils.library.restartPython()

import pyspark
from pyspark.sql.functions import col, avg, sum, first, lower, trim, to_date, when
from databricks.feature_engineering import FeatureEngineeringClient

# Pega o parâmetro de ambiente injetado pelo YAML referente ao catalogo (dev, homlog, prod)
#env_catalog = dbutils.widgets.get("env_catalog")
env_catalog="dev"
esquema_gold = "gold"
tabela_gold_eventos = f"{env_catalog}.{esquema_gold}.base_eventos_produto_gold"

# Lendo a tabela na camada gold
df_gold = spark.table(tabela_gold_eventos)

# ==========================================
# 1. CRIAÇÃO DO TARGET (Antes do GroupBy)
# ==========================================
# Criamos a target usando a tabela original para não perder as colunas de categoria e evento
df_target = df_gold.withColumn("dia_prtc", to_date(col("dia_prtc").cast("string"), "yyyyMMdd")) \
    .withColumn("comprou_eletronico", when(
        (col("categoria") == "Eletrônicos") & (col("tipo_evento") == "finalizar_compra"), 1
    ).otherwise(0)) \
    .select("id_usuario", "timestamp_registro", "dia_prtc", "comprou_eletronico")

nome_tabela_target = f"{env_catalog}.{esquema_gold}.target_compras_eletronicos"

# ==========================================
# GRAVAÇÃO COM DYNAMIC PARTITION OVERWRITE
# ==========================================
df_target.write \
    .format("delta") \
    .option("partitionOverwriteMode", "dynamic") \
    .partitionBy("dia_prtc") \
    .mode("overwrite") \
    .saveAsTable(nome_tabela_target)


# ==========================================
# 2. AGREGAÇÃO DE FEATURES
# ==========================================
df_user_features_temporal = df_gold.groupBy("id_usuario", "timestamp_registro").agg(
    lower(trim(first("idade"))).alias("idade_cliente"),
    lower(trim(first("genero"))).alias("genero_cliente"),
    lower(trim(first("regiao"))).alias("regiao_cliente"),
    first("renda_mensal_estimada_k").alias("renda_mensal_k"),
    avg("tempo_clique_segundos").alias("tempo_medio_clique_segundos"),
    avg("interacoes_chat_suporte").alias("media_interacoes_suporte"),
    avg("cupons_ativos_conta").alias("media_cupons_ativos"),
    avg("score_satisfacao_nps").alias("media_score_nps_cliente"),
    avg("dias_desde_ultima_visita").alias("media_dias_inatividade"),
    sum(col("preco_reais")).alias("total_gasto_acumulado_reais")
)

# Aplica o One-Hot Encoding nativo e tipagem
df_features_final_spark = df_user_features_temporal \
    .withColumn("idade_cliente", col("idade_cliente").cast("integer")) \
    .withColumn("renda_mensal_k", col("renda_mensal_k").cast("double")) \
    .withColumn("total_gasto_acumulado_reais", col("total_gasto_acumulado_reais").cast("double")) \
    .withColumn("genero_cliente_m", when(col("genero_cliente") == "m", 1).otherwise(0).cast("integer")) \
    .withColumn("regiao_cliente_nordeste", when(col("regiao_cliente") == "nordeste", 1).otherwise(0).cast("integer")) \
    .withColumn("regiao_cliente_norte", when(col("regiao_cliente") == "norte", 1).otherwise(0).cast("integer")) \
    .withColumn("regiao_cliente_sudeste", when(col("regiao_cliente") == "sudeste", 1).otherwise(0).cast("integer")) \
    .withColumn("regiao_cliente_sul", when(col("regiao_cliente") == "sul", 1).otherwise(0).cast("integer")) \
    .drop("genero_cliente", "regiao_cliente")


# ==========================================
# 3. REGISTRO NO FEATURE STORE
# ==========================================
fe = FeatureEngineeringClient()
nome_feature_table_temporal = f"{env_catalog}.{esquema_gold}.user_profile_features_temporal"

# Salvamos o DataFrame processado final (df_features_final_spark)
fe.create_table(
    name=nome_feature_table_temporal,
    primary_keys=["id_usuario","timestamp_registro"], 
    timeseries_columns=["timestamp_registro"],
    df=df_features_final_spark,
    schema=df_features_final_spark.schema,
    description="Perfil de recursos comportamentais e demográficos históricos por usuário."
)