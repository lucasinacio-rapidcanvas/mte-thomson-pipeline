# ================================================================================
  # RECIPE: product_view
  # ================================================================================
  # DEPENDÊNCIAS: Este recipe requer os seguintes datasets:
  #     1. new_multihorizon_products_abcxyz (previsões de vendas de produtos)
  #     2. new_monthly_portalvendas (histórico de vendas mensal)
  #     3. df_estrutura_produto (BOM - Bill of Materials)
  # ================================================================================
  #
  # PROPÓSITO: Preparar e sanitizar dados de produtos (previsões, histórico, BOM)
  #            para consumo no frontend, garantindo tipos corretos e removendo
  #            valores inválidos que poderiam causar erros em visualizações.
  #
  # INPUTS:
  #   - new_multihorizon_products_abcxyz (previsões com classificação ABC-XYZ)
  #   - new_monthly_portalvendas (vendas históricas agregadas por mês)
  #   - df_estrutura_produto (estrutura produto-componente)
  #
  # OUTPUTS:
  #   - df_sales_forecast_complete (previsões sanitizadas)
  #   - df_sales_historical (histórico sanitizado)
  #   - df_product_structure (BOM sanitizada)
  #
  # FILTROS APLICADOS:
  #   1. df[df['QTDE_PEDIDA'] > 0] (linha 87): Remove previsões zero ou negativas
  #      Exemplo: df_forecast_clean = df_forecast_clean[df_forecast_clean['QTDE_PEDIDA'] > 0]
  #      * Consequência: Apenas previsões positivas são mantidas para análise
  #
  #   2. sanitize_value(value) (linhas 47-59): Converte nulos/inválidos para zero
  #      Exemplo: df['QTDE_PEDIDA'].apply(sanitize_value)
  #      * Consequência: NaN, '', <NA>, inf viram 0, evitando erros em gráficos
  #
  #   3. df[df['G1_QUANT'] > 0] (linha 127): Remove componentes com quantidade inválida
  #      Exemplo: df_bom_clean = df_bom_clean[df_bom_clean['G1_QUANT'] > 0]
  #      * Consequência: BOM mantém apenas relacionamentos válidos (quantidade > 0)
  #
  # LÓGICA:
  #   FASE 1 - CARREGAMENTO (linhas 25-43):
  #     1. Carrega new_multihorizon_products_abcxyz (previsões)
  #     2. Carrega new_monthly_portalvendas (histórico)
  #     3. Carrega df_estrutura_produto (BOM)
  #     4. Print de shapes para validação
  #
  #   FASE 2 - FUNÇÕES AUXILIARES (linhas 46-72):
  #     1. sanitize_value(): Converte valores inválidos para 0
  #        - Trata: NaN, '', '<NA>', inf, erros de conversão
  #     2. sanitize_date(): Converte datas inválidas para None
  #        - Trata: NaT, erros de parsing
  #
  #   FASE 3 - PROCESSAR FORECAST (linhas 75-92):
  #     1. Converte DATA_PEDIDO e base_date para datetime
  #     2. Converte COD_MTE_COMP para string
  #     3. Sanitiza QTDE_PEDIDA (remove nulos/inválidos)
  #     4. Filtra QTDE_PEDIDA > 0
  #     5. Ordena por: COD_MTE_COMP + base_date + DATA_PEDIDO
  #
  #   FASE 4 - PROCESSAR HISTÓRICO (linhas 95-113):
  #     1. Converte MONTH para datetime
  #     2. Converte COD_MTE_COMP para string
  #     3. Sanitiza 6 colunas numéricas:
  #        - QTDE_PEDIDA, QTDE_SALDO, QTDE_ENTREGUE
  #        - VALOR_FATURADO, VALOR_SALDO, VALOR_PEDIDO
  #     4. Ordena por: COD_MTE_COMP + MONTH
  #
  #   FASE 5 - PROCESSAR BOM (linhas 116-132):
  #     1. Converte COD_PRODUTO e COD_COMPONENTE para string
  #     2. Sanitiza G1_QUANT (quantidade de componente por produto)
  #     3. Filtra G1_QUANT > 0
  #     4. Ordena por: COD_PRODUTO + COD_COMPONENTE
  #
  #   FASE 6 - SALVAR OUTPUTS (linhas 135-160):
  #     1. Salva df_sales_forecast_complete
  #     2. Salva df_sales_historical
  #     3. Salva df_product_structure
  #
  #   FASE 7 - ESTATÍSTICAS (linhas 163-200):
  #     1. Print resumo de registros e colunas
  #     2. Print total de produtos, componentes, datas base
  #     3. Print período de previsão e histórico
  #
  # EXEMPLO COMPLETO:
  #   INPUT:
  #     new_multihorizon_products_abcxyz:
  #       COD_MTE_COMP="PROD001", DATA_PEDIDO="2025-05-01", QTDE_PEDIDA=1000.5
  #     
  #     new_monthly_portalvendas:
  #       COD_MTE_COMP="PROD001", MONTH="2024-12-01", QTDE_PEDIDA=950,
  #       VALOR_FATURADO=45000
  #     
  #     df_estrutura_produto:
  #       COD_PRODUTO="PROD001", COD_COMPONENTE="COMP123", G1_QUANT=1.5
  #   
  #   PROCESSAMENTO:
  #   → Forecast: Converte tipos (datetime, str, numeric), remove QTDE_PEDIDA ≤ 0
  #   → Histórico: Sanitiza 6 colunas numéricas (nulos→0), ordena por produto+mês
  #   → BOM: Remove quantidades ≤ 0, converte códigos para string
  #   
  #   OUTPUT df_sales_forecast_complete:
  #   {
  #     "COD_MTE_COMP": "PROD001",
  #     "DATA_PEDIDO": "2025-05-01",
  #     "base_date": "2025-01-01",
  #     "QTDE_PEDIDA": 1000.5
  #   }
  #   
  #   OUTPUT df_sales_historical:
  #   {
  #     "COD_MTE_COMP": "PROD001",
  #     "MONTH": "2024-12-01",
  #     "QTDE_PEDIDA": 950.0,
  #     "VALOR_FATURADO": 45000.0
  #   }
  #   
  #   OUTPUT df_product_structure:
  #   {
  #     "COD_PRODUTO": "PROD001",
  #     "COD_COMPONENTE": "COMP123",
  #     "G1_QUANT": 1.5
  #   }
  # ================================================================================


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports

from utils.notebookhelpers.helpers import Helpers
from utils.dtos.templateOutputCollection import TemplateOutputCollection
from utils.dtos.templateOutput import TemplateOutput
from utils.dtos.templateOutput import OutputType
from utils.dtos.templateOutput import ChartType
from utils.dtos.variable import Metadata
from utils.rcclient.commons.variable_datatype import VariableDatatype
from utils.dtos.templateOutput import FileType
from utils.dtos.rc_ml_model import RCMLModel
from utils.notebookhelpers.helpers import Helpers
from utils.libutils.vectorStores.utils import VectorStoreUtils

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Your code goes here
import datetime
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# CARREGAR DATASETS DE ENTRADA
print("Carregando datasets...")

# Dataset 1: Previsões Multi-Horizonte
# Entity: new_multihorizon_products_abcxyz
df_forecast = Helpers.getEntityData(context, 'new_multihorizon_products_abcxyz')

# Dataset 2: Histórico de Vendas Mensal
# Entity: new_monthly_portalvendas
df_historical = Helpers.getEntityData(context, 'new_monthly_portalvendas')

# Dataset 3: Estrutura de Produto (BOM)
# Entity: estrutura_produto
df_bom = Helpers.getEntityData(context, 'df_estrutura_produto')

print(f"Datasets carregados:")
print(f"  - Forecast: {df_forecast.shape}")
print(f"  - Historical: {df_historical.shape}")
print(f"  - BOM: {df_bom.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# FUNÇÕES AUXILIARES
def sanitize_value(value):
    """
    Sanitiza valores numéricos: null/NaN/'' → 0
    """
    if pd.isna(value) or value == '' or value == '<NA>':
        return 0
    try:
        num = float(value)
        if np.isnan(num) or np.isinf(num):
            return 0
        return num
    except (ValueError, TypeError):
        return 0

def sanitize_date(date):
    """
    Sanitiza datas: inválidas → None
    """
    if pd.isna(date):
        return None
    try:
        return pd.to_datetime(date)
    except:
        return None

print("Funcoes auxiliares definidas")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Inicializa lista de auditoria
lista_dfs_sem_match = []

# ETAPA 1: PROCESSAR FORECAST (Previsões)
print("Processando previsoes de vendas...")

df_forecast_clean = df_forecast.copy()

# Garantir tipos de dados corretos
df_forecast_clean['DATA_PEDIDO'] = pd.to_datetime(df_forecast_clean['DATA_PEDIDO'])
df_forecast_clean['base_date'] = pd.to_datetime(df_forecast_clean['base_date'])
df_forecast_clean['COD_MTE_COMP'] = df_forecast_clean['COD_MTE_COMP'].astype(str)
df_forecast_clean['QTDE_PEDIDA'] = df_forecast_clean['QTDE_PEDIDA'].apply(sanitize_value)

# --- FILTRO 1: Remover registros com quantidade zero ou negativa ---
mask_forecast_zero = df_forecast_clean['QTDE_PEDIDA'] <= 0

if mask_forecast_zero.sum() > 0:
    df_removed_fc = df_forecast_clean[mask_forecast_zero].copy()
    df_audit_fc = df_removed_fc[['COD_MTE_COMP']].rename(columns={'COD_MTE_COMP': 'Cod_component'})
    df_audit_fc['origem'] = 'df_sales_forecast'
    df_audit_fc['motivo'] = 'Previsao de venda zerada ou negativa (QTDE_PEDIDA <= 0)'
    lista_dfs_sem_match.append(df_audit_fc)

# Aplica o filtro
df_forecast_clean = df_forecast_clean[~mask_forecast_zero]

# Ordenar por produto, base_date e data de pedido
df_forecast_clean = df_forecast_clean.sort_values(['COD_MTE_COMP', 'base_date', 'DATA_PEDIDO'])

print(f"Forecast processado: {df_forecast_clean.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 2: PROCESSAR HISTÓRICO DE VENDAS
print("Processando historico de vendas...")

df_historical_clean = df_historical.copy()

# Garantir tipos de dados corretos
df_historical_clean['MONTH'] = pd.to_datetime(df_historical_clean['MONTH'])
df_historical_clean['COD_MTE_COMP'] = df_historical_clean['COD_MTE_COMP'].astype(str)

# Sanitizar colunas numéricas
numeric_cols = ['QTDE_PEDIDA', 'QTDE_SALDO', 'QTDE_ENTREGUE', 'VALOR_FATURADO', 'VALOR_SALDO', 'VALOR_PEDIDO']
for col in numeric_cols:
    if col in df_historical_clean.columns:
        df_historical_clean[col] = df_historical_clean[col].apply(sanitize_value)

# Ordenar por produto e mês
df_historical_clean = df_historical_clean.sort_values(['COD_MTE_COMP', 'MONTH'])

print(f"Historico processado: {df_historical_clean.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 3: PROCESSAR ESTRUTURA DE PRODUTO (BOM)
print("Processando estrutura de produtos...")

df_bom_clean = df_bom.copy()

# Garantir tipos de dados corretos
df_bom_clean['COD_PRODUTO'] = df_bom_clean['COD_PRODUTO'].astype(str)
df_bom_clean['COD_COMPONENTE'] = df_bom_clean['COD_COMPONENTE'].astype(str)
df_bom_clean['G1_QUANT'] = df_bom_clean['G1_QUANT'].apply(sanitize_value)

# --- FILTRO 2: Remover componentes com quantidade zero ou negativa ---
mask_bom_zero = df_bom_clean['G1_QUANT'] <= 0

if mask_bom_zero.sum() > 0:
    df_removed_bom = df_bom_clean[mask_bom_zero].copy()
    # Usando COD_PRODUTO como chave principal (produto pai)
    df_audit_bom = df_removed_bom[['COD_PRODUTO']].rename(columns={'COD_PRODUTO': 'Cod_component'})
    df_audit_bom['origem'] = 'df_estrutura_produto'
    df_audit_bom['motivo'] = 'Quantidade na estrutura zerada ou negativa (G1_QUANT <= 0)'
    lista_dfs_sem_match.append(df_audit_bom)

# Aplica o filtro
df_bom_clean = df_bom_clean[~mask_bom_zero]

# Ordenar por produto e componente
df_bom_clean = df_bom_clean.sort_values(['COD_PRODUTO', 'COD_COMPONENTE'])

print(f"BOM processado: {df_bom_clean.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# CONSOLIDAÇÃO DO DATAFRAME DE AUDITORIA (df_sem_match)

if len(lista_dfs_sem_match) > 0:
    df_sem_match = pd.concat(lista_dfs_sem_match, ignore_index=True)
else:
    df_sem_match = pd.DataFrame(columns=['Cod_component', 'origem', 'motivo'])

# Garantir colunas e remover duplicatas
df_sem_match = df_sem_match[['Cod_component', 'origem', 'motivo']].drop_duplicates()
Helpers.save_output_dataset(context=context, output_name='df_sem_match_atual_6', data_frame=df_sem_match)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 4: SALVAR OUTPUTS
print("Salvando datasets de saida...")

# Output 1: Previsões de Vendas (completo)
Helpers.save_output_dataset(
    context=context,
    output_name='df_sales_forecast_complete',
    data_frame=df_forecast_clean
)
print(f"  Salvo: df_sales_forecast_complete ({df_forecast_clean.shape})")

# Output 2: Histórico de Vendas
Helpers.save_output_dataset(
    context=context,
    output_name='df_sales_historical',
    data_frame=df_historical_clean
)
print(f"  Salvo: df_sales_historical ({df_historical_clean.shape})")

# Output 3: Estrutura de Produto (BOM)
Helpers.save_output_dataset(
    context=context,
    output_name='df_product_structure',
    data_frame=df_bom_clean
)
print(f"  Salvo: df_product_structure ({df_bom_clean.shape})")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# RESUMO FINAL
print("=" * 80)
print("PROCESSAMENTO CONCLUIDO COM SUCESSO!")
print("=" * 80)
print("")
print("RESUMO DOS DATASETS GERADOS:")
print("")
print(f"  1. df_sales_forecast_complete")
print(f"     - Registros: {df_forecast_clean.shape[0]:,}")
print(f"     - Colunas: {df_forecast_clean.shape[1]}")
print(f"     - Uso: Previsoes de vendas por produto, base_date e mes")
print("")
print(f"  2. df_sales_historical")
print(f"     - Registros: {df_historical_clean.shape[0]:,}")
print(f"     - Colunas: {df_historical_clean.shape[1]}")
print(f"     - Uso: Historico real de vendas mensais")
print("")
print(f"  3. df_product_structure")
print(f"     - Registros: {df_bom_clean.shape[0]:,}")
print(f"     - Colunas: {df_bom_clean.shape[1]}")
print(f"     - Uso: Estrutura BOM (componentes por produto)")
print("")
print("=" * 80)
print("")
print("NOTA: Metadados e consumo de componentes serao calculados no frontend React")
print("      - Metadados extraidos dos 3 datasets principais")
print("      - Consumo calculado usando df_sales_forecast_complete + df_product_structure")
print("")
print("=" * 80)
print("")
print("ESTATISTICAS:")
print(f"  - Total de produtos: {len(df_forecast_clean['COD_MTE_COMP'].unique())}")
print(f"  - Total de datas base: {len(df_forecast_clean['base_date'].unique())}")
print(f"  - Total de componentes: {len(df_bom_clean['COD_COMPONENTE'].unique())}")
print(f"  - Periodo de previsao: {df_forecast_clean['DATA_PEDIDO'].min()} -> {df_forecast_clean['DATA_PEDIDO'].max()}")
print(f"  - Periodo historico: {df_historical_clean['MONTH'].min()} -> {df_historical_clean['MONTH'].max()}")
print("")
print("=" * 80)