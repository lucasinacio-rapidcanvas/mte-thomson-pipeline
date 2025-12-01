# ================================================================================
  # RECIPE: generate_component_consumption_by_products
  # ================================================================================
  # DEPENDÊNCIAS: Este recipe requer os seguintes datasets:
  #     1. new_multihorizon_products_abcxyz (previsão de vendas de produtos)
  #     2. df_estrutura_produto (BOM - Bill of Materials)
  # ================================================================================
  #
  # PROPÓSITO: Calcular consumo de componentes por produto final através de explosão
  #            BOM. Aplica regras de arredondamento para números inteiros em todas
  #            as métricas (Vendas, Consumo e Quantidade Técnica).
  #
  # INPUTS:
  #   - new_multihorizon_products_abcxyz (previsões de vendas por produto e mês)
  #   - df_estrutura_produto (estrutura produto-componente com quantidades)
  #
  # OUTPUT:
  #   - component_product_consumption (consumo detalhado por componente e produto)
  #
  # FILTROS APLICADOS:
  #   1. merge(..., how="inner") (linha 88-93): Mantém apenas produtos com estrutura
  #      Exemplo: df_merged = df_forecast_product.merge(df_estrutura, how="inner")
  #      * Consequência: Produtos sem BOM definida são excluídos do cálculo de consumo
  #
  #   2. fillna(0) (linha 103): Preenche consumos inválidos com zero
  #      Exemplo: df_merged["ComponentConsumption"].fillna(0)
  #      * Consequência: Valores nulos de multiplicação viram 0, não causam erro
  #
  #   3. loc[df["ComponentConsumption"] < 0, ...] = 0 (linha 104): Remove negativos
  #      Exemplo: df_merged.loc[df_merged["ComponentConsumption"] < 0, "ComponentConsumption"] = 0
  #      * Consequência: Consumos negativos (dados incorretos) viram 0
  #
  #   4. groupby([...], as_index=False).agg({...}) (linhas 108-117): Agrega por chaves
  #      Exemplo: df.groupby(["Componente", "base_date", "Codigo", "DATA_PEDIDO"]).agg(...)
  #      * Consequência: Soma consumo quando mesmo componente aparece múltiplas vezes no BOM
  #
  # LÓGICA:
  #   FASE 1 - CARREGAMENTO:
  #     1. Carrega new_multihorizon_products_abcxyz (previsões de produtos)
  #     2. Carrega df_estrutura_produto (relacionamento componente-produto)
  #
  #   FASE 2 - PADRONIZAÇÃO DE COLUNAS:
  #     1. Renomeia COD_PRODUTO → Codigo
  #     2. Renomeia COD_COMPONENTE → Componente
  #     3. Renomeia G1_QUANT → Quantidade
  #
  #   FASE 3 - CONVERSÃO DE TIPOS:
  #     1. Forecast: COD_MTE_COMP→str, DATA_PEDIDO→datetime, QTDE_PEDIDA→numeric
  #     2. Estrutura: Codigo→str, Componente→str, Quantidade→numeric
  #
  #   FASE 4 - MERGE:
  #     1. Inner join: forecast_product + estrutura usando COD_MTE_COMP = Codigo
  #     2. Mantém apenas produtos com estrutura definida
  #
  #   FASE 5 - CÁLCULO DE CONSUMO:
  #     1. ComponentConsumption = QTDE_PEDIDA × Quantidade
  #     2. Preenche nulos com 0
  #     3. Força negativos para 0
  #
  #   FASE 6 - AGREGAÇÃO:
  #     1. Agrupa por: Componente + base_date + Codigo + DATA_PEDIDO
  #     2. Soma QTDE_PEDIDA (vendas totais do produto)
  #     3. Soma ComponentConsumption (consumo total do componente)
  #     4. Mantém primeiro valor de Quantidade (constante no grupo)
  #
  #   FASE 7 - RENOMEAÇÃO:
  #     1. Componente → Component, Codigo → ProductCode
  #     2. QTDE_PEDIDA → ForecastSales, Quantidade → QuantityPerUnit
  #
  #   FASE 8 - CONVERSÕES FINAIS E ARREDONDAMENTO (INTEIROS):
  #     1. Converte Component e ProductCode para category
  #     2. Converte base_date e DATA_PEDIDO para datetime
  #     3. Aplica arredondamento (round 0) e converte para INTEIRO (int64) as colunas:
  #        - ForecastSales (Vendas)
  #        - ComponentConsumption (Consumo)
  #        - QuantityPerUnit (Quantidade Técnica - sem metades)
  #
  #   FASE 9 - ORDENAÇÃO:
  #     1. Ordena por: Component + base_date + ProductCode + DATA_PEDIDO
  #
  # EXEMPLO COMPLETO:
  #   INPUT:
  #     new_multihorizon_products_abcxyz:
  #       COD_MTE_COMP="PROD001", QTDE_PEDIDA=1000.6
  #     
  #     df_estrutura_produto:
  #       COD_PRODUTO="PROD001", G1_QUANT=1.6
  #   
  #   PROCESSAMENTO:
  #   → Cálculo Bruto: Consumo = 1000.6 × 1.6 = 1600.96
  #   → Arredondamento Forecast: 1000.6 → 1001 (int)
  #   → Arredondamento Consumo: 1600.96 → 1601 (int)
  #   → Arredondamento Qtd Unitária: 1.6 → 2 (int)
  #   
  #   OUTPUT:
  #   {
  #     "ForecastSales": 1001,
  #     "ComponentConsumption": 1601,
  #     "QuantityPerUnit": 2
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
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 1: Carregar datasets
df_forecast_product = Helpers.getEntityData(context, 'new_multihorizon_products_abcxyz')
df_estrutura = Helpers.getEntityData(context, 'df_estrutura_produto')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 2: Padronizar nomes de colunas da estrutura
df_estrutura = df_estrutura.rename(columns={
    "COD_PRODUTO": "Codigo",
    "COD_COMPONENTE": "Componente",
    "G1_QUANT": "Quantidade"
})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 3: Conversão de tipos antes do merge
# Dataset de produtos
df_forecast_product["COD_MTE_COMP"] = df_forecast_product["COD_MTE_COMP"].astype(str)
df_forecast_product["DATA_PEDIDO"] = pd.to_datetime(df_forecast_product["DATA_PEDIDO"])
df_forecast_product["base_date"] = pd.to_datetime(df_forecast_product["base_date"])
df_forecast_product["QTDE_PEDIDA"] = pd.to_numeric(df_forecast_product["QTDE_PEDIDA"], errors='coerce')

# Dataset de estrutura
df_estrutura["Codigo"] = df_estrutura["Codigo"].astype(str)
df_estrutura["Componente"] = df_estrutura["Componente"].astype(str)
df_estrutura["Quantidade"] = pd.to_numeric(df_estrutura["Quantidade"], errors='coerce')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 4: Merge forecast de produtos com estrutura (BOM)
df_merged = df_forecast_product.merge(
    df_estrutura,
    left_on="COD_MTE_COMP",
    right_on="Codigo",
    how="inner"  # INNER: apenas produtos que têm estrutura definida
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 5: Calcular consumo de componente
# Fórmula: Consumo = Vendas previstas do produto × Quantidade de componente por produto
df_merged["ComponentConsumption"] = (
    df_merged["QTDE_PEDIDA"] * df_merged["Quantidade"]
)

# Substituir valores inválidos por 0
df_merged["ComponentConsumption"] = df_merged["ComponentConsumption"].fillna(0)
df_merged.loc[df_merged["ComponentConsumption"] < 0, "ComponentConsumption"] = 0

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 6: Agrupar por Component + base_date + ProductCode + DATA_PEDIDO
df_aggregated = df_merged.groupby([
    "Componente",      # Código do componente
    "base_date",       # Mês base da previsão
    "Codigo",          # Código do produto pai
    "DATA_PEDIDO"      # Mês da previsão
], as_index=False).agg({
    "QTDE_PEDIDA": "sum",           # Total de vendas previstas do produto
    "ComponentConsumption": "sum",   # Total de consumo do componente
    "Quantidade": "first"            # Quantidade por unidade (constante por grupo)
})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 7: Renomear colunas para padrão do frontend
df_aggregated = df_aggregated.rename(columns={
    "Componente": "Component",
    "Codigo": "ProductCode",
    "QTDE_PEDIDA": "ForecastSales",
    "Quantidade": "QuantityPerUnit"
})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 8: Conversões finais e Arredondamento (Regra: Tudo Inteiro)
# Colunas categóricas
df_aggregated["Component"] = df_aggregated["Component"].astype("category")
df_aggregated["ProductCode"] = df_aggregated["ProductCode"].astype("category")

# Colunas de data
df_aggregated["base_date"] = pd.to_datetime(df_aggregated["base_date"])
df_aggregated["DATA_PEDIDO"] = pd.to_datetime(df_aggregated["DATA_PEDIDO"])

# Colunas numéricas: Arredondamento (round 0) e Conversão para Inteiro (int64)
# Aplica-se a Vendas, Consumo e Quantidade Unitária (sem metades/decimais)
numeric_cols = ["ForecastSales", "ComponentConsumption", "QuantityPerUnit"]

for col in numeric_cols:
    df_aggregated[col] = df_aggregated[col].round(0).astype("int64")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 9: Ordenar e salvar
df_aggregated = df_aggregated.sort_values(
    ["Component", "base_date", "ProductCode", "DATA_PEDIDO"],
    ignore_index=True
)

Helpers.save_output_dataset(
    context=context,
    output_name='component_product_consumption',
    data_frame=df_aggregated
)