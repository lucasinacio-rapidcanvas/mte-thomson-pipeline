# ================================================================================
  # RECIPE: enrich_forecast_components_view
  # ================================================================================
  # DEPENDÊNCIAS: Este recipe requer os seguintes datasets:
  #     1. new_multihorizon_components_abcxyz (previsões de componentes)
  #     2. produtos (cadastro mestre de produtos)
  #     3. df_produto_fornecedor (relacionamento produto-fornecedor com custos)
  # ================================================================================
  #
  # PROPÓSITO: Enriquecer previsões de componentes com descrições, grupos, fornecedores,
  #            custos e metadados calculados para consumo em visualizações frontend.
  #
  # INPUTS:
  #   - new_multihorizon_components_abcxyz (previsões base por componente e mês)
  #   - produtos (descrições, grupos, preços, KanBan, origem)
  #   - df_produto_fornecedor (fornecedores, custos, moedas)
  #
  # OUTPUT:
  #   - forecast_components_enriched (dataset completo para frontend)
  #
  # FILTROS APLICADOS:
  #   1. left join com produtos (linha 56): Mantém todas as previsões mesmo sem produto cadastrado
  #      Exemplo: df_enriched = df_forecast_components.merge(
  #                   df_produtos[[...]], left_on="Component", right_on="B1_COD", how="left")
  #      * Consequência: Componentes sem cadastro aparecem com campos de produto nulos
  #
  #   2. left join com produto_fornecedor (linha 80): Mantém componentes sem fornecedor
  #      Exemplo: df_enriched = df_enriched.merge(
  #                   df_produto_fornecedor[[...]], left_on="Component", right_on="PRODUTO", how="left")
  #      * Consequência: Componentes sem fornecedor terão COD_FORNE, FORNEC_NOM nulos
  #
  #   3. COD_FORNE.fillna("MTE") (linha 97): Preenche fornecedores ausentes com fabricação interna
  #      Exemplo: df_enriched["COD_FORNE"] = df_enriched["COD_FORNE"].fillna("MTE")
  #      * Consequência: Componentes sem fornecedor externo são marcados como "MTE" (fabricação interna)
  #
  #   4. FORNEC_NOM.fillna("MTE") (linha 98): Preenche nome de fornecedor ausente
  #      Exemplo: df_enriched["FORNEC_NOM"] = df_enriched["FORNEC_NOM"].fillna("MTE")
  #      * Consequência: Consistência entre código e nome de fornecedor para fabricação interna
  #
  #   5. drop columns ["B1_COD", "PRODUTO"] (linha 101): Remove colunas duplicadas após merge
  #      Exemplo: df_enriched = df_enriched.drop(columns=["B1_COD", "PRODUTO"], errors="ignore")
  #      * Consequência: Mantém apenas "Component" como identificador único, remove duplicatas
  #
  #   6. col in df_enriched.columns (linha 118): Valida existência de coluna antes de converter tipo
  #      Exemplo: for col in categorical_cols:
  #                   if col in df_enriched.columns: df_enriched[col] = df_enriched[col].astype("category")
  #      * Consequência: Evita erros se alguma coluna esperada não existir no dataset
  #
  # LÓGICA:
  #   FASE 1 - CARREGAMENTO (linhas 49-52):
  #     1. Carrega new_multihorizon_components_abcxyz (previsões base)
  #     2. Carrega produtos (cadastro mestre)
  #     3. Carrega df_produto_fornecedor (relacionamentos e custos)
  #
  #   FASE 2 - ENRIQUECIMENTO COM PRODUTOS (linhas 55-76):
  #     1. Merge left com produtos usando Component = B1_COD
  #     2. Adiciona 12 colunas de produto:
  #        - B1_DESC (descrição), B1_GRUPO (código do grupo), NOM_GRUP (nome do grupo)
  #        - ORIGEM (nacional/importado), CURVA (classificação ABC)
  #        - PRECO_VENDA_BRL/USD/EUR (preços de venda em 3 moedas)
  #        - MULTIPLO_COMPRA (lote mínimo), KANBAN_MIN, KANBAN_MAX
  #        - B1_ATIVO (ativo/inativo), DTA_CADASTRO (data de criação)
  #
  #   FASE 3 - ENRIQUECIMENTO COM FORNECEDOR (linhas 79-92):
  #     1. Merge left com df_produto_fornecedor usando Component = PRODUTO
  #     2. Adiciona 6 colunas de fornecedor:
  #        - COD_FORNE (código do fornecedor), FORNEC_NOM (nome do fornecedor)
  #        - COD_FABRI (código do fabricante), MOEDA (moeda do custo)
  #        - CUSTO_PRODUTO (custo unitário)
  #
  #   FASE 4 - TRATAMENTO DE NULOS (linhas 95-101):
  #     1. Preenche COD_FORNE nulo com "MTE" (fabricação interna)
  #     2. Preenche FORNEC_NOM nulo com "MTE"
  #     3. Remove colunas duplicadas: B1_COD, PRODUTO
  #
  #   FASE 5 - CONVERSÃO DE TIPOS (linhas 104-119):
  #     1. Converte DATA_PEDIDO e base_date para datetime64
  #     2. Converte consumption_predicted_month para float64
  #     3. Converte 9 colunas para category (otimização de memória):
  #        - Component, B1_GRUPO, NOM_GRUP, ORIGEM, CURVA
  #        - B1_ATIVO, COD_FORNE, FORNEC_NOM, MOEDA
  #
  #   FASE 6 - COLUNAS CALCULADAS (linhas 122-134):
  #     1. forecast_horizon_months = (DATA_PEDIDO.year - base_date.year) × 12 + 
  #                                   (DATA_PEDIDO.month - base_date.month)
  #     2. forecast_month_str = DATA_PEDIDO.strftime("%Y-%m")
  #     3. base_month_str = base_date.strftime("%Y-%m")
  #     4. consumption_rounded = round(consumption_predicted_month, 2)
  #
  #   FASE 7 - ORDENAÇÃO (linhas 137-141):
  #     1. Ordena por: Component ASC, base_date ASC, DATA_PEDIDO ASC
  #     2. Reset de índice para sequência contínua
  #
  # EXEMPLO COMPLETO:
  #   INPUT:
  #     new_multihorizon_components_abcxyz:
  #       Component="ABC123", DATA_PEDIDO=2025-05-01, base_date=2025-01-01, 
  #       consumption_predicted_month=1250.5678
  #     
  #     produtos:
  #       B1_COD="ABC123", B1_DESC="Rolamento", B1_GRUPO="0101", NOM_GRUP="ROLAMENTOS",
  #       ORIGEM="IMP", CURVA="A", PRECO_VENDA_USD=15.50, MULTIPLO_COMPRA=10
  #     
  #     df_produto_fornecedor:
  #       PRODUTO="ABC123", COD_FORNE="5001", FORNEC_NOM="Supplier XYZ",
  #       MOEDA="USD", CUSTO_PRODUTO=12.30
  #   
  #   PROCESSAMENTO:
  #   → Merge produtos: adiciona descrição="Rolamento", grupo="0101", preço=15.50
  #   → Merge fornecedor: adiciona COD_FORNE="5001", FORNEC_NOM="Supplier XYZ", custo=12.30
  #   → Conversões: Component→category, DATA_PEDIDO→datetime, consumption→float64
  #   → Cálculos:
  #     - forecast_horizon_months = (2025-05 - 2025-01) = 4 meses
  #     - forecast_month_str = "2025-05"
  #     - base_month_str = "2025-01"
  #     - consumption_rounded = 1250.57
  #   
  #   OUTPUT:
  #   {
  #     "Component": "ABC123",
  #     "DATA_PEDIDO": "2025-05-01",
  #     "base_date": "2025-01-01",
  #     "consumption_predicted_month": 1250.5678,
  #     "B1_DESC": "Rolamento",
  #     "B1_GRUPO": "0101",
  #     "NOM_GRUP": "ROLAMENTOS",
  #     "COD_FORNE": "5001",
  #     "FORNEC_NOM": "Supplier XYZ",
  #     "MOEDA": "USD",
  #     "CUSTO_PRODUTO": 12.30,
  #     "forecast_horizon_months": 4,
  #     "forecast_month_str": "2025-05",
  #     "consumption_rounded": 1250.57
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

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())


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
# Required standard library imports
import pandas as pd  # For DataFrame operations

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 1: Carregar datasets
df_forecast_components = Helpers.getEntityData(context, 'new_multihorizon_components_abcxyz')
df_produtos = Helpers.getEntityData(context, 'produtos')
df_produto_fornecedor = Helpers.getEntityData(context, 'df_produto_fornecedor')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 2: Merge com produtos (adicionar descrição, grupo, preços)
df_enriched = df_forecast_components.merge(
    df_produtos[[
        "B1_COD",
        "B1_DESC",
        "B1_GRUPO",
        "NOM_GRUP",
        "ORIGEM",
        "CURVA",
        "PRECO_VENDA_BRL",
        "PRECO_VENDA_USD",
        "PRECO_VENDA_EUR",
        "MULTIPLO_COMPRA",
        "KANBAN_MIN",
        "KANBAN_MAX",
        "B1_ATIVO",
        "DTA_CADASTRO"
    ]],
    left_on="Component",
    right_on="B1_COD",
    how="left"
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 3: Merge com fornecedor
df_enriched = df_enriched.merge(
    df_produto_fornecedor[[
        "PRODUTO",
        "COD_FORNE",
        "FORNEC_NOM",
        "COD_FABRI",
        "MOEDA",
        "CUSTO_PRODUTO"
    ]],
    left_on="Component",
    right_on="PRODUTO",
    how="left"
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 4: Tratamento de valores nulos e limpeza
# Preencher valores ausentes de fornecedor com "MTE" (fabricação interna)
df_enriched["COD_FORNE"] = df_enriched["COD_FORNE"].fillna("MTE")
df_enriched["FORNEC_NOM"] = df_enriched["FORNEC_NOM"].fillna("MTE")

# Remover colunas duplicadas (B1_COD e PRODUTO são duplicatas de Component)
df_enriched = df_enriched.drop(columns=["B1_COD", "PRODUTO"], errors="ignore")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 5: Conversão de tipos para otimização
# Colunas de data
df_enriched["DATA_PEDIDO"] = pd.to_datetime(df_enriched["DATA_PEDIDO"])
df_enriched["base_date"] = pd.to_datetime(df_enriched["base_date"])

# Colunas numéricas
df_enriched["consumption_predicted_month"] = df_enriched["consumption_predicted_month"].astype("float64")

# Colunas categóricas (economiza memória - importante para datasets grandes)
categorical_cols = [
    "Component", "B1_GRUPO", "NOM_GRUP", "ORIGEM", "CURVA",
    "B1_ATIVO", "COD_FORNE", "FORNEC_NOM", "MOEDA"
]
for col in categorical_cols:
    if col in df_enriched.columns:
        df_enriched[col] = df_enriched[col].astype("category")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 6: Adicionar colunas úteis para o frontend
# Coluna de forecast horizon (quantos meses à frente da base_date)
df_enriched["forecast_horizon_months"] = (
    (df_enriched["DATA_PEDIDO"].dt.year - df_enriched["base_date"].dt.year) * 12 +
    (df_enriched["DATA_PEDIDO"].dt.month - df_enriched["base_date"].dt.month)
)

# Colunas de string de data (facilita filtros no frontend)
df_enriched["forecast_month_str"] = df_enriched["DATA_PEDIDO"].dt.strftime("%Y-%m")
df_enriched["base_month_str"] = df_enriched["base_date"].dt.strftime("%Y-%m")

# Adicionar coluna de consumo arredondado (para exibição limpa)
df_enriched["consumption_rounded"] = df_enriched["consumption_predicted_month"].round(2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 7: Ordenar para melhor performance em queries
df_enriched = df_enriched.sort_values(
    ["Component", "base_date", "DATA_PEDIDO"],
    ignore_index=True
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ETAPA 8: Salvar dataset enriquecido
Helpers.save_output_dataset(
    context=context,
    output_name='forecast_components_enriched',
    data_frame=df_enriched
)