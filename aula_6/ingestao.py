# Databricks notebook source
# ==============================================================================
# scr/ingestao.py
# Script simples para testar a injeção de parâmetros
# ==============================================================================
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# 1. Pega o parâmetro de ambiente injetado pelo YAML
env_catalog = dbutils.widgets.get("env_catalog")

print(f"🚀 Iniciando job de teste no schema: {env_catalog}")

# COMMAND ----------

# 2. Cria um DataFrame simples
dados_teste = [
    (1, "1", "@email.com", "cliente"),
    (2, "2", "@email.com", "admin")
]
colunas = ["id", "nome", "email", "perfil"]
df = spark.createDataFrame(dados_teste, colunas)

# Criando os catalogos DEV, STAGE, PROD
spark.sql(f"CREATE CATALOG IF NOT EXISTS {env_catalog}")

# 3. Salva a tabela
nome_tabela = f"{env_catalog}.default.usuarios_bronze"
print(f" Salvando tabela em: {nome_tabela}...")
df.write.format("delta").mode("overwrite").saveAsTable(nome_tabela)
print(" Tabela criada com sucesso!")
