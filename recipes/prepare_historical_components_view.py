# ================================================================================
  # RECIPE: prepare_historical_components
  # ================================================================================
  # DEPENDÊNCIAS: Este recipe requer os seguintes datasets:
  #     1. new_monthly_portalvendas_components (histórico de consumo mensal)
  # ================================================================================
  #
  # PROPÓSITO: Preparar histórico de consumo mensal de componentes para visualização
  #            no frontend, convertendo tipos, adicionando metadados temporais e
  #            ordenando para performance em gráficos de série temporal.
  #
  # INPUTS:
  #   - new_monthly_portalvendas_components (consumo histórico agregado por mês)
  #
  # OUTPUT:
  #   - monthly_components_prepared (histórico preparado para frontend)
  #
  # FILTROS APLICADOS:
  #   1. pd.to_numeric(..., errors='coerce') (linha 59): Converte inválidos para NaN
  #      Exemplo: df[col] = pd.to_numeric(df[col], errors='coerce')
  #      * Consequência: Valores não-numéricos viram NaN, não causam erro de conversão
  #
  #   2. sort_values(["Component", "MONTH"]) (linhas 80-83): Ordena cronologicamente
  #      Exemplo: df_historical.sort_values(["Component", "MONTH"], ignore_index=True)
  #      * Consequência: Dados ordenados por componente e tempo para gráficos eficientes
  #
  # LÓGICA:
  #   FASE 1 - CARREGAMENTO (linha 42):
  #     1. Carrega new_monthly_portalvendas_components (histórico mensal)
  #
  #   FASE 2 - CONVERSÃO DE TIPOS (linhas 45-59):
  #     1. Converte MONTH para datetime
  #     2. Converte Component para category (otimização de memória)
  #     3. Converte Consumption para float64 (métrica principal)
  #     4. Converte 5 colunas opcionais para numeric com tratamento de erros:
  #        - QTDE_SALDO, QTDE_ENTREGUE, VALOR_FATURADO, VALOR_SALDO, VALOR_PEDIDO
  #
  #   FASE 3 - ADIÇÃO DE METADADOS (linhas 62-76):
  #     1. month_str = MONTH.strftime("%Y-%m") (formato para filtros)
  #     2. consumption_rounded = Consumption.round(2) (exibição limpa)
  #     3. year = MONTH.year (análise anual)
  #     4. quarter = MONTH.quarter (análise trimestral)
  #     5. month_name = MONTH.strftime("%B") (nome do mês em inglês)
  #     6. is_current_year = (year == ano_corrente) (flag para análises)
  #
  #   FASE 4 - ORDENAÇÃO (linhas 79-83):
  #     1. Ordena por Component + MONTH (ordem cronológica)
  #     2. Reset de índice para sequência contínua
  #
  #   FASE 5 - SALVAMENTO (linhas 86-90):
  #     1. Salva monthly_components_prepared
  #
  # EXEMPLO COMPLETO:
  #   INPUT:
  #     new_monthly_portalvendas_components:
  #       Component="COMP123", MONTH="2024-11-01", Consumption=1250.5678,
  #       QTDE_ENTREGUE=1250, VALOR_FATURADO=37500
  #   
  #   PROCESSAMENTO:
  #   → Conversão tipos: MONTH→datetime, Component→category, Consumption→float64
  #   → Metadados temporais:
  #     - month_str = "2024-11"
  #     - consumption_rounded = 1250.57
  #     - year = 2024
  #     - quarter = 4
  #     - month_name = "November"
  #     - is_current_year = False (assumindo ano corrente = 2025)
  #   → Ordenação: Por Component ASC, MONTH ASC
  #   
  #   OUTPUT:
  #   {
  #     "Component": "COMP123",
  #     "MONTH": "2024-11-01",
  #     "Consumption": 1250.5678,
  #     "QTDE_ENTREGUE": 1250.0,
  #     "VALOR_FATURADO": 37500.0,
  #     "month_str": "2024-11",
  #     "consumption_rounded": 1250.57,
  #     "year": 2024,
  #     "quarter": 4,
  #     "month_name": "November",
  #     "is_current_year": false
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


# Required standard library imports
import pandas as pd  

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 1: Carregar histórico de consumo
df_historical = Helpers.getEntityData(context, 'new_monthly_portalvendas_components')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 2: Conversão de tipos
# Colunas de data
df_historical["MONTH"] = pd.to_datetime(df_historical["MONTH"])

# Colunas categóricas
df_historical["Component"] = df_historical["Component"].astype("category")

# Colunas numéricas - Consumption é a principal
df_historical["Consumption"] = df_historical["Consumption"].astype("float64")

# Outras colunas numéricas (se existirem)
numeric_cols = ["QTDE_SALDO", "QTDE_ENTREGUE", "VALOR_FATURADO", "VALOR_SALDO", "VALOR_PEDIDO"]
for col in numeric_cols:
    if col in df_historical.columns:
        df_historical[col] = pd.to_numeric(df_historical[col], errors='coerce')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 3: Adicionar metadados úteis para o frontend
# String de mês para facilitar filtros (formato YYYY-MM)
df_historical["month_str"] = df_historical["MONTH"].dt.strftime("%Y-%m")

# Adicionar coluna de consumo arredondado (para exibição limpa)
df_historical["consumption_rounded"] = df_historical["Consumption"].round(2)

# Adicionar informações temporais úteis
df_historical["year"] = df_historical["MONTH"].dt.year
df_historical["quarter"] = df_historical["MONTH"].dt.quarter
df_historical["month_name"] = df_historical["MONTH"].dt.strftime("%B")  # Nome do mês em inglês

# Adicionar flag de ano corrente (útil para análises)
current_year = pd.Timestamp.now().year
df_historical["is_current_year"] = df_historical["year"] == current_year

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 4: Ordenar para melhor performance
df_historical = df_historical.sort_values(
    ["Component", "MONTH"],
    ignore_index=True
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 5: Salvar dataset preparado
Helpers.save_output_dataset(
    context=context,
    output_name='monthly_components_prepared',
    data_frame=df_historical
)