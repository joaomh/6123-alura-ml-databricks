# Databricks notebook source
# Databricks notebook source
# ==============================================================================
import pyspark

# Pega o parâmetro de ambiente injetado pelo YAML referente ao catalogo (dev, homlog, prod)
env_catalog = dbutils.widgets.get("env_catalog")

# Criando o catálogo (DEV, STAGE, PROD) injetado pelo Job
spark.sql(f"CREATE CATALOG IF NOT EXISTS {env_catalog}")

# ==========================================
# CAMADA BRONZE
# ==========================================
esquema_bronze = "bronze"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {env_catalog}.{esquema_bronze}")

tabela_bronze_eventos = f"{env_catalog}.{esquema_bronze}.tabela_eventos_bronze"

# ==========================================
# CAMADA SILVER
# ==========================================
esquema_silver = "silver"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {env_catalog}.{esquema_silver}")

tabela_silver_eventos = f"{env_catalog}.{esquema_silver}.tabela_eventos_silver"

# ==========================================
# CAMADA GOLD
# ==========================================
esquema_gold = "gold"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {env_catalog}.{esquema_gold}")

tabela_gold_eventos = f"{env_catalog}.{esquema_gold}.base_eventos_produto_gold"
tabela_gold_dim_produto = f"{env_catalog}.{esquema_gold}.dim_produtos"

print("Iniciando a criação das tabelas de entrada na Camada Bronze...")

# -------------------------------------------------------------------------
# 1. CRIAÇÃO DA TABELA BRONZE (LOGS DE EVENTOS DO E-COMMERCE)
# -------------------------------------------------------------------------
print(f"Ingerindo arquivo de eventos para a tabela Delta: {tabela_bronze_eventos}")

spark.sql(f"""
    CREATE OR REPLACE TABLE {tabela_bronze_eventos}
    USING DELTA
    AS 
    SELECT * FROM read_files(
      '/Volumes/workspace/bronze/raw/base_bronze_eventos.csv',
      format => 'csv',
      header => 'true',
      sep => ";",
      schema => '
        id_evento STRING,
        id_usuario STRING,
        tipo_evento STRING,
        timestamp_registro STRING,
        dados_origem_json STRING
      '
    )
""")

print(f"Processando a Camada Silver ({tabela_silver_eventos}) a partir dos dados do JSON...")

# Execução do pipeline via Spark SQL
spark.sql(f"""
CREATE OR REPLACE TABLE {tabela_silver_eventos}
    USING DELTA
    AS
    WITH dados_deduplicados AS (
        SELECT DISTINCT 
            id_evento,
            id_usuario,
            tipo_evento,
            CAST(timestamp_registro AS TIMESTAMP) AS timestamp_registro,
            
            -- LIMPEZA DO JSON (Tratamento para remover aspas duplicadas corporativas do CSV)
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    TRIM(dados_origem_json), 
                    '^"|"$', '' -- Remove as aspas que envelopam o texto nas pontas
                ), 
                '""', '"'      -- Transforma duas aspas "" em apenas uma "
            ) AS json_limpo
            
        FROM {tabela_bronze_eventos}
    ),
    dados_parseados AS (
        SELECT 
            id_evento,
            id_usuario,
            tipo_evento,
            timestamp_registro,
            
            -- Faz o parse em cima do campo texto que agora está perfeitamente higienizado
            from_json(
                json_limpo, 
                'id_produto STRING, 
                 dispositivo STRING, 
                 sistema_operacional STRING, 
                 origem_trafego STRING, 
                 idade INT, 
                 genero STRING, 
                 renda_mensal_estimada_k DECIMAL(18,2), 
                 regiao STRING, 
                 tempo_clique_segundos INT, 
                 interacoes_chat_suporte INT, 
                 cupons_ativos_conta INT, 
                 score_satisfacao_nps INT, 
                 dias_desde_ultima_visita INT'
            ) AS payload
        FROM dados_deduplicados
    )
    -- Projeção final espalhando as propriedades nas colunas relacionais
    SELECT 
        id_evento,
        id_usuario,
        tipo_evento,
        timestamp_registro,
        payload.id_produto,
        payload.dispositivo,
        payload.sistema_operacional,
        payload.origem_trafego,
        payload.idade,
        payload.genero,
        payload.renda_mensal_estimada_k,
        payload.regiao,
        payload.tempo_clique_segundos,
        payload.interacoes_chat_suporte,
        payload.cupons_ativos_conta,
        payload.score_satisfacao_nps,
        payload.dias_desde_ultima_visita
    FROM dados_parseados
""")

print("Camada Silver gerada com sucesso com colunas totalmente estruturadas!")

# -------------------------------------------------------------------------
# CRIAÇÃO DA TABELA DE DIMENSÃO (CADASTRO DE PRODUTOS)
# -------------------------------------------------------------------------
spark.sql(f"""
    CREATE OR REPLACE TABLE {tabela_gold_dim_produto}
    USING DELTA
    AS 
    SELECT * FROM read_files(
      '/Volumes/workspace/bronze/raw/base_dim_produtos.csv',
      format => 'csv',
      header => 'true',
      schema => '
        id_produto STRING,
        nome_produto STRING,
        categoria STRING,
        preco_reais DECIMAL(18,2)
      '
    )
""")

print(" Ingestão da tabela {tabela_produtos} concluída com sucesso!")
# Execução do pipeline via Spark SQL
spark.sql(f"""
CREATE OR REPLACE TABLE {tabela_gold_eventos}
    USING DELTA
    AS
        -- Cruzamento da Silver com a Dimensão de Produtos
        SELECT 
            s.id_usuario,
            s.tipo_evento,
            s.dispositivo,
            s.sistema_operacional,
            s.timestamp_registro,
            s.origem_trafego,
            s.idade,
            s.genero,
            s.renda_mensal_estimada_k,
            s.regiao,
            s.tempo_clique_segundos,
            s.interacoes_chat_suporte,
            s.cupons_ativos_conta,
            s.score_satisfacao_nps,
            s.dias_desde_ultima_visita,
            p.categoria,
            p.preco_reais,
        -- Extrai o carimbo de data/hora no formato yyyymmdd e converte para INT
        CAST(date_format(s.timestamp_registro, 'yyyyMMdd') AS INT) AS dia_prtc
        FROM workspace.silver.tabela_eventos_silver s
        INNER JOIN  workspace.default.dim_produtos p ON s.id_produto = p.id_produto
""")

print("Camada Gold consolidada com sucesso!")