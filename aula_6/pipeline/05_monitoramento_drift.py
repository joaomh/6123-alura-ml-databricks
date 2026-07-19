# Databricks notebook source
# Databricks notebook source
# ==============================================================================
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, TimestampType
from scipy.stats import ks_2samp
from datetime import datetime

# ==========================================
# 1. PARÂMETROS E INTEGRAÇÃO DO WORKFLOW
# ==========================================
# Injeta o catálogo do ambiente definido no pipeline YAML
env_catalog = dbutils.widgets.get("env_catalog")

# Tenta pegar as datas via TaskValues (da task de geração), com fallback para Widgets
try:
    p_inicio = dbutils.jobs.taskValues.get(taskKey="task_geracao_dados", key="data_inicio", default=dbutils.widgets.get("data_inicio"))
    p_fim = dbutils.jobs.taskValues.get(taskKey="task_geracao_dados", key="data_fim", default=dbutils.widgets.get("data_fim"))
except Exception:
    p_inicio = dbutils.widgets.get("data_inicio")
    p_fim = dbutils.widgets.get("data_fim")

esquema = "gold"
nome_tabela_features = f"{env_catalog}.{esquema}.user_profile_features_temporal"
nome_tabela_target = f"{env_catalog}.{esquema}.target_compras_eletronicos"
nome_tabela_prod = f"{env_catalog}.{esquema}.base_tabela_prod"
nome_tabela_auditoria = f"{env_catalog}.{esquema}.log_monitoramento_drift"

# ==========================================
# 1.1 EXTRAÇÃO DINÂMICA DE FEATURES
# ==========================================
# Define o Target fixo por regra de negócio
p_target = "comprou_eletronico"

# Lê as colunas da tabela de produção para descobrir as features
todas_colunas = spark.table(nome_tabela_prod).columns

# Remove os metadados, chaves de particionamento/IDs e o próprio Target
colunas_excluidas = ["dia_prtc", "id_unico", "id_usuario", "timestamp_registro", p_target]

# Tudo o que sobrar na tabela será considerado uma feature para o KS Test
lista_features = [coluna for coluna in todas_colunas if coluna not in colunas_excluidas]

print(f"Iniciando framework de validação de qualidade de dados.")
print(f"Avaliando partições entre {p_inicio} e {p_fim} no ambiente: {env_catalog}")
print(f"Target detectado: {p_target}")
print(f"Total de Features detectadas: {len(lista_features)}")
print(f"Features: {lista_features}\n")

# ==========================================
# 2. LEITURA, RECONSTRUÇÃO DA BASE DE TREINO E VALIDAÇÃO
# ==========================================
# Reconstruindo a base de treino através do cruzamento das tabelas
df_features_treino = spark.table(nome_tabela_features)
df_target_treino = spark.table(nome_tabela_target)

df_treino = df_features_treino.join(
    df_target_treino,
    on=["id_usuario", "timestamp_registro"],
    how="inner"
)

# Pushdown filter aproveitando a partição dia_prtc na tabela produtiva
df_prod = spark.table(nome_tabela_prod).filter(
    F.col("dia_prtc").between(p_inicio, p_fim)
)

qtd_prod = df_prod.count()
if qtd_prod == 0:
    dbutils.notebook.exit(f"Nenhum dado produtivo na janela de {p_inicio} a {p_fim}.")

logs_execucao = []

def registrar(nome, var, val, thresh):
    is_p = "p-value" in nome.lower()
    drift = bool(val < thresh) if is_p else bool(val > thresh)
    logs_execucao.append((datetime.now(), p_inicio, p_fim, nome, var, float(val), float(thresh), drift))

# ==========================================
# 3. CÁLCULO DE DRIFT (FEATURES E TARGET)
# ==========================================
# Feature Drift Dinâmico
for feature in lista_features:
    print(f"Calculando KS Test para: {feature}")
    
    # Extraímos a coluna e convertemos para NumPy array nativo usando amostragem segura
    arr_treino = df_treino.select(feature).sample(fraction=0.1, seed=42).toPandas()[feature].to_numpy()
    arr_prod = df_prod.select(feature).sample(fraction=0.1, seed=42).toPandas()[feature].to_numpy()

    if len(arr_prod) > 0 and len(arr_treino) > 0:
        stat, p_value = ks_2samp(arr_treino, arr_prod)
        registrar("Feature drift KS: p-value", feature, p_value, 0.05)

# Concept Drift Dinâmico (Target)
print(f"Calculando Drift de Conversão para o target: {p_target}")
media_t = df_treino.select(F.avg(p_target)).collect()[0][0] or 0
media_p = df_prod.select(F.avg(p_target)).collect()[0][0] or 0

if media_t > 0:
    var_relativa = abs(media_p - media_t) / media_t
    registrar("Concept Drift (Variação Relativa de Conversão)", p_target, var_relativa, 0.15)

# ==========================================
# 4. GRAVAÇÃO DOS LOGS DE AUDITORIA
# ==========================================
schema_log = StructType([
    StructField("timestamp_execucao", TimestampType(), False),
    StructField("periodo_inicio", StringType(), False),
    StructField("periodo_fim", StringType(), False),
    StructField("nome_metrica", StringType(), False),
    StructField("variavel", StringType(), False),
    StructField("valor_calculado", DoubleType(), False),
    StructField("threshold_alerta", DoubleType(), False),
    StructField("drift_detectado", BooleanType(), False)
])

df_logs = spark.createDataFrame(logs_execucao, schema=schema_log)
df_logs.write.format("delta").mode("append").saveAsTable(nome_tabela_auditoria)

print(f"\nMétricas calculadas e registradas com sucesso em {nome_tabela_auditoria}.")

# ==========================================
# 5. GATILHO DE ALERTA (TASK VALUES)
# ==========================================
# Verifica se existe algum "True" na coluna de drift (índice 7 da tupla)
drift_encontrado = any(log[7] for log in logs_execucao)

print(f"\nStatus do Drift: {'Detectado' if drift_encontrado else 'Não Detectado'}")

# Passa o sinal verde (ou vermelho) para a interface do Workflow
dbutils.jobs.taskValues.set(key="drift_detectado", value=drift_encontrado)

# Repassa as datas da anomalia para garantir que o retreino pegue a janela exata
dbutils.jobs.taskValues.set(key="data_inicio", value=p_inicio)
dbutils.jobs.taskValues.set(key="data_fim", value=p_fim)

if drift_encontrado:
    print("🚨 Desvio detectado. A próxima task condicional do Workflow será acionada para retreino.")
else:
    print("✅ Dados saudáveis. O Workflow será encerrado normalmente.")