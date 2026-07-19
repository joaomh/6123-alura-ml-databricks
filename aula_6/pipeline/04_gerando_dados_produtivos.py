# Databricks notebook source
# Databricks notebook source
# ==============================================================================
import pandas as pd
import pyspark
import numpy as np
from datetime import datetime, timedelta
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import uuid # Necessário para o id_unico

# ==========================================
# 1. PARÂMETROS DE AMBIENTE E DATAS
# ==========================================
env_catalog = dbutils.widgets.get("env_catalog")
esquema_gold = "gold"

nome_tabela_prod = f"{env_catalog}.{esquema_gold}.base_tabela_prod"

# ==========================================
# 1. CALCULO DINÂMICO DE DATAS
# ==========================================
print(f"Lendo a tabela {nome_tabela_prod} para descobrir a última data processada...")

try:
    # Busca o maior dia_prtc (último dia processado)
    ultima_data_str = spark.table(nome_tabela_prod).select(F.max("dia_prtc")).collect()[0][0]
    ultima_data = datetime.strptime(ultima_data_str, "%Y%m%d")
except Exception:
    # Fallback de segurança caso a tabela esteja completamente vazia
    ultima_data = datetime.now() - timedelta(days=1)

# Definição das novas janelas: Início é amanhã, Fim é em 15 dias
dt_inicio = ultima_data + timedelta(days=1)
dt_fim = dt_inicio + timedelta(days=15)

p_inicio = dt_inicio.strftime("%Y%m%d")
p_fim = dt_fim.strftime("%Y%m%d")

print(f"Gerando novos dados de produção. Período: {p_inicio} a {p_fim}")

# ==========================================
# 2. FUNÇÃO DE GERAÇÃO
# ==========================================
def gerar_producao_periodo(spark, dt_inicio_str, dt_fim_str, n_por_dia=100):
    
    # Gerando os dias particionados de forma aleatória dentro do range
    dt_inicio = datetime.strptime(dt_inicio_str, '%Y%m%d')
    dt_fim = datetime.strptime(dt_fim_str, '%Y%m%d')
    dias_delta = (dt_fim - dt_inicio).days
    
    # Tratamento caso os dias sejam iguais
    if dias_delta <= 0:
        dias_delta = 1
        
    n_total = n_por_dia * dias_delta
    
    regioes = np.random.choice(['nordeste', 'norte', 'sudeste', 'sul', 'outro'], n_total, p=[0.2, 0.1, 0.4, 0.2, 0.1])
    
    np.random.seed(int(dt_inicio_str))
    random_offsets = np.random.randint(0, dias_delta, n_total)

    dias_particao = [(dt_inicio + timedelta(days=int(offset))).strftime("%Y%m%d") for offset in random_offsets]
    
    pdf_prod = pd.DataFrame({
        "id_unico": [str(uuid.uuid4()) for _ in range(n_total)], # Mantendo o UUID da arquitetura
        "dia_prtc": dias_particao, # Coluna de partição
        "renda_mensal_k": np.random.uniform(0, 2000, n_total),
        "tempo_medio_clique_segundos": np.random.uniform(10, 30, n_total),
        "media_interacoes_suporte": np.random.uniform(30, 200, n_total),
        "media_cupons_ativos": np.random.uniform(0, 15, n_total),
        "media_score_nps_cliente": np.random.randint(0, 8, n_total),
        "media_dias_inatividade": np.random.randint(0, 120, n_total),
        "total_gasto_acumulado_reais":np.random.uniform(1000, 20500, n_total),
        "genero_cliente_m": np.random.binomial(1, 0.48, n_total),
        "regiao_cliente_nordeste": (regioes == 'nordeste').astype(int),
        "regiao_cliente_norte": (regioes == 'norte').astype(int),
        "regiao_cliente_sudeste": (regioes == 'sudeste').astype(int),
        "regiao_cliente_sul": (regioes == 'sul').astype(int)
    })
    
    # Target Prod
    prob_prod = 1 / (1 + np.exp(-(pdf_prod['total_gasto_acumulado_reais']*0.1 + pdf_prod['media_interacoes_suporte']*0.05 - 300)))
    pdf_prod['comprou_eletronico'] = np.random.binomial(1, prob_prod)

    schema_dados = StructType([
        StructField("id_unico", StringType(), True),
        StructField("dia_prtc", StringType(), True),
        StructField("renda_mensal_k", DoubleType(), True),
        StructField("tempo_medio_clique_segundos", DoubleType(), True),
        StructField("media_interacoes_suporte", DoubleType(), True),
        StructField("media_cupons_ativos", DoubleType(), True),
        StructField("media_score_nps_cliente", DoubleType(), True),
        StructField("media_dias_inatividade", DoubleType(), True),
        StructField("total_gasto_acumulado_reais", DoubleType(), True),
        StructField("genero_cliente_m", IntegerType(), True),
        StructField("regiao_cliente_nordeste", IntegerType(), True),
        StructField("regiao_cliente_norte", IntegerType(), True),
        StructField("regiao_cliente_sudeste", IntegerType(), True),
        StructField("regiao_cliente_sul", IntegerType(), True),
        StructField("comprou_eletronico", IntegerType(), True)
    ])

    return spark.createDataFrame(pdf_prod, schema=schema_dados)

# ==========================================
# 3. GERAÇÃO E GRAVAÇÃO
# ==========================================
df_prod = gerar_producao_periodo(spark, p_inicio, p_fim)

# ==========================================
# GRAVAÇÃO COM DYNAMIC PARTITION OVERWRITE
# ==========================================
df_prod.write \
    .format("delta") \
    .option("partitionOverwriteMode", "dynamic") \
    .partitionBy("dia_prtc") \
    .mode("overwrite") \
    .saveAsTable(nome_tabela_prod)

print(f"Carga finalizada. {df_prod.count()} registros salvos em {nome_tabela_prod}.")

# ==========================================
# 4. PASSANDO PARÂMETROS PARA A PRÓXIMA TASK
# ==========================================
print("Salvando as datas geradas para as próximas tarefas do Workflow...")

dbutils.jobs.taskValues.set(key="data_inicio", value=p_inicio)
dbutils.jobs.taskValues.set(key="data_fim", value=p_fim)

print(f"Variáveis 'data_inicioa' ({p_inicio}) e 'data_fim' ({p_fim}) salvas no contexto da Task.")