# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports (Mantenha todos os imports necessários no topo)
from collections import defaultdict
import logging
import datetime
import numpy as np
import pandas as pd
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE


def clean_whitespace(df, column):
    """Remove espaços em branco no início e fim de uma coluna de string."""
    # Garante que a coluna é do tipo string antes de aplicar .str.strip()
    return df[column].astype(str).str.strip()


def filter_t_components(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Filtra linhas onde a coluna especificada começa com 'T' seguida por um ou mais dígitos.

    Exemplo: 'T123', 'T9' são mantidos; 'TA', 't1' ou '1T' são removidos.

    Retorna:
    --------
    pd.DataFrame: DataFrame com registros que começam com 'T' + dígitos
    """
    mask = df[column].astype(str).str.match(r'^T\d+')
    return df[mask].copy()


def remove_double_dot_lines(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove linhas em que a coluna selecionada contém exatamente dois pontos ".".

    Exemplo removido: T9708.50.030

    Retorna:
    --------
    pd.DataFrame: DataFrame com registros que NÃO têm exatamente dois pontos
    """
    mask = df[column].astype(str).str.count(r'\.') != 2
    return df[mask].copy()


def fill_all_missing_periods(df, date_column, product_column, freq="D", fill_dict=None):
    """
    Preenche períodos ausentes em um DataFrame, aplicando métodos de preenchimento por grupo de produto.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada.
        date_column (str): Nome da coluna de datas.
        product_column (str): Nome da coluna de produtos.
        freq (str): Frequência do período (ex: "D", "M", "W").
        fill_dict (dict): Dicionário com o formato {coluna: metodo_ou_valor}.
                          Exemplo: {"VALOR UNITARIO R$": "ffill", "ESTOQUE": 0}
    """
    # [A sua função original fill_all_missing_periods vai aqui, sem alteração]
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    date_range = pd.date_range(min_date, max_date, freq=freq)
    product_codes = df[product_column].unique().tolist()

    all_combinations = pd.MultiIndex.from_product(
        [product_codes, date_range],
        names=[product_column, date_column]
    )
    new_df = pd.DataFrame(index=all_combinations).reset_index()

    merged_df = pd.merge(
        new_df, df, on=[product_column, date_column], how='left')

    # Aplica os preenchimentos conforme o dicionário
    if fill_dict:
        for col, method in fill_dict.items():
            if col not in merged_df.columns:
                continue  # ignora colunas inexistentes
            if method == "ffill":
                merged_df[col] = merged_df.groupby(product_column)[
                    col].fillna(method="ffill")
            elif method == "bfill":
                merged_df[col] = merged_df.groupby(product_column)[
                    col].fillna(method="bfill")
            else:
                merged_df[col] = merged_df[col].fillna(method)

    # Preenche o restante com 0
    merged_df = merged_df.fillna(0)

    return merged_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Produtos nao entregues pela exportacao


df_products_not_delivered_exportation = Helpers.getEntityData(
    context, "products_not_delivered_exportation")
df_products_not_delivered_exportation = df_products_not_delivered_exportation.dropna(
    subset=['DESCRICAO', 'QUANT_BO'])
df_products_not_delivered_exportation['PRODUTO'] = df_products_not_delivered_exportation['PRODUTO'].astype(
    str).str.strip()
df_products_not_delivered_exportation['DESCRICAO'] = df_products_not_delivered_exportation['DESCRICAO'].astype(
    str).str.strip()
df_products_not_delivered_exportation['QUANT_BO'] = pd.to_numeric(
    df_products_not_delivered_exportation['QUANT_BO'].astype(str).str.strip(), errors='coerce')
df_products_not_delivered_exportation['DT_ENTREGA'] = pd.to_datetime(
    df_products_not_delivered_exportation['DT_ENTREGA'].astype(str).str.strip(), format='%d/%m/%Y', errors='coerce')
Helpers.save_output_dataset(context=context, output_name="df_products_not_delivered_exportation",
                            data_frame=df_products_not_delivered_exportation)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Estrutura do Produto (BOM)
df_estrutura_produto = Helpers.getEntityData(context, 'estrutura_produto')

# Aplica a função universal de filtragem
df_estrutura_produto = filter_t_components(
    df_estrutura_produto, 'COD_COMPONENTE')

# 2. Aplica a função de remoção de linhas com dois pontos
df_estrutura_produto = remove_double_dot_lines(
    df_estrutura_produto, 'COD_COMPONENTE')

Helpers.save_output_dataset(
    context=context, output_name='df_estrutura_produto', data_frame=df_estrutura_produto)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Produto Fornecedor
df_produto_fornecedor = Helpers.getEntityData(context, "produto_fornecedor")
df_produto_fornecedor["PRODUTO"] = clean_whitespace(
    df_produto_fornecedor, "PRODUTO")

# Aplica a função universal de filtragem
df_produto_fornecedor = filter_t_components(df_produto_fornecedor, 'PRODUTO')

# 2. Aplica a função de remoção de linhas com dois pontos
df_produto_fornecedor = remove_double_dot_lines(
    df_produto_fornecedor, 'PRODUTO')

# 3. Apenas produtos com preco maior que 0
df_produto_fornecedor = df_produto_fornecedor[df_produto_fornecedor['CUSTO_PRODUTO'] > 0]

# --- 1. Limpeza e Padronização de COD_FORNE e COD_FABRI ---
df_produto_fornecedor["COD_FORNE"] = (
    df_produto_fornecedor["COD_FORNE"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.replace("nan", "", regex=False)
)

df_produto_fornecedor["COD_FABRI"] = (
    df_produto_fornecedor["COD_FABRI"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.replace("nan", "", regex=False)
)

# 3. Definição das colunas de String
string_cols = ["PRODUTO", "COD_FORNE", "COD_FABRI",
               "FORNEC_NOM", "MOEDA", "COD_PROD_FOR"]

# 4. Aplica o tipo string (pd.StringDtype) nas colunas definidas
df_produto_fornecedor[string_cols] = df_produto_fornecedor[string_cols].astype(
    "string")

# 5. Conversão e arredondamento
float_cols = ["CUSTO_PRODUTO", "PTAX"]
df_produto_fornecedor[float_cols] = df_produto_fornecedor[float_cols].apply(
    pd.to_numeric, errors="coerce").round(2)


Helpers.save_output_dataset(
    context=context, output_name='df_produto_fornecedor', data_frame=df_produto_fornecedor)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 📜 Histórico de Inventário e Backcasting (Formato Longo)

# --- Importa
df_estoque_atual = Helpers.getEntityData(context, 'estoque_atual')
df_estoque_atual["CODIGO_PI"] = clean_whitespace(df_estoque_atual, "CODIGO_PI")

# Aplica a função universal de filtragem
df_estoque_atual = filter_t_components(df_estoque_atual, 'CODIGO_PI')

# 2. Aplica a função de remoção de linhas com dois pontos
df_estoque_atual = remove_double_dot_lines(df_estoque_atual, 'CODIGO_PI')

# --- Importa
df_movimentacao_estoque = Helpers.getEntityData(
    context, 'movimentacao_estoque')
df_movimentacao_estoque["B1_COD"] = clean_whitespace(
    df_movimentacao_estoque, "B1_COD")

# Aplica a função universal de filtragem
df_movimentacao_estoque = filter_t_components(
    df_movimentacao_estoque, 'B1_COD')

# 2. Aplica a função de remoção de linhas com dois pontos
df_movimentacao_estoque = remove_double_dot_lines(
    df_movimentacao_estoque, 'B1_COD')

# Manipulacao
set_estoque_atual = df_estoque_atual["CODIGO_PI"].unique()

# --- Pré-processamento e Preenchimento de Dias Ausentes ---
df_movimentacao_estoque["DTA_LANCAMENTO"] = pd.to_datetime(
    df_movimentacao_estoque["DTA_LANCAMENTO"])

df_movimentacao_estoque_alldays = fill_all_missing_periods(
    df_movimentacao_estoque,
    "DTA_LANCAMENTO",
    "B1_COD",
    freq="D",
    fill_dict={"QTD_LANCADA": 0}
)

min_date = df_movimentacao_estoque_alldays["DTA_LANCAMENTO"].min()
max_date = df_movimentacao_estoque_alldays["DTA_LANCAMENTO"].max()
df_inventory_histories_base = pd.DataFrame(  # Renomeado para evitar conflito
    index=pd.date_range(start=min_date, end=max_date, freq="D")
)

# --- Backcasting (Cálculo Retroativo) ---
estoque_por_part = dict(tuple(df_estoque_atual.groupby("CODIGO_PI")))
movimentacoes_por_part = dict(
    tuple(df_movimentacao_estoque_alldays.groupby("B1_COD")))
new_inventory_histories = defaultdict(dict)

for part_number in set_estoque_atual:
    df_part_number = estoque_por_part.get(part_number)
    df_deltas = movimentacoes_por_part.get(part_number)

    if df_part_number is None or df_part_number.empty:
        continue

    if df_deltas is None or df_deltas.empty:
        last_stock = df_part_number["QTD_TOT_EST"].iloc[-1]
        for day in df_inventory_histories_base.index:
            new_inventory_histories[day][part_number] = last_stock
        continue

    df_deltas_reversed = df_deltas.sort_values(
        by="DTA_LANCAMENTO", ascending=False)
    stock = df_part_number["QTD_TOT_EST"].iloc[-1]
    unique_dates = df_deltas_reversed["DTA_LANCAMENTO"].unique()

    for day in df_inventory_histories_base.index[::-1]:
        if day < min_date:
            continue

        if day in unique_dates:
            delta = df_deltas_reversed[df_deltas_reversed["DTA_LANCAMENTO"]
                                       == day]["QTD_LANCADA"].sum()
            stock -= delta

        new_inventory_histories[day][part_number] = stock

# --- Finalização do DataFrame de Histórico (Cria Pivotado, DERRETE para Longo) ---

# 1. Cria o DataFrame pivotado (Este passo é necessário para usar o resultado do backcasting)
df_inventory_histories_pivot = pd.DataFrame.from_dict(
    new_inventory_histories, orient="index").sort_index()

# 2. Renomeia colunas (mantida a lógica de substituir "_" por "/")
df_inventory_histories_pivot.columns = df_inventory_histories_pivot.columns.astype(
    str).str.replace("_", "/")
df_inventory_histories_pivot = df_inventory_histories_pivot.reset_index(
).rename(columns={'index': 'date'})

# 3. DESPIVOTAMENTO
df_inventory_histories = df_inventory_histories_pivot.melt(
    id_vars=['date'],                      # Coluna a ser mantida (Datas)
    # Nova coluna para os nomes dos componentes (T123, T456)
    var_name='Componente',
    value_name='QTD_ESTOQUE'               # Nova coluna para os valores de estoque
)

# 4. Limpeza final e tipo
df_inventory_histories['Componente'] = df_inventory_histories['Componente'].astype(
    str)

Helpers.save_output_dataset(
    context=context, output_name='new_inventory_histories2', data_frame=df_inventory_histories)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧩 Compilação e Classificação de Componentes (Metadados)
df_produtos = Helpers.getEntityData(context, 'produtos')
df_sobdemanda = Helpers.getEntityData(context, 'produtos_sob_demanda')
df_ativos = Helpers.getEntityData(context, 'produto_fornecedor_ativo')
df_excecoes = Helpers.getEntityData(
    context, 'excecoes_produtos_sem_fornecedores')
df_parametro_fornecedor = Helpers.getEntityData(
    context, 'parametro_fornecedor')

# --- Limpeza e Marcação ---
# (Manter a lógica de remoção de duplicatas e criação de colunas de flag)
df_ativos = df_ativos.drop_duplicates(
    subset=['Cod_Produto']).assign(Ativo=True)
df_excecoes = df_excecoes.drop_duplicates(
    subset=['Cod_Produto']).assign(Excecao=True)
df_sobdemanda = df_sobdemanda.drop_duplicates(
    subset=['Cod_Produto']).assign(Sob_Demanda=True)

# --- Junção dos Metadados ---
df_compiled = df_ativos.merge(df_excecoes, how="outer", on="Cod_Produto")
df_compiled = df_compiled.merge(df_sobdemanda, how="outer", on="Cod_Produto")
df_compiled = df_compiled.fillna(
    {"Ativo": False, "Excecao": False, "Sob_Demanda": False})

# --- Adicionar Histórico de Estoque e Regras de Negócio ---
components_inventory_histories = df_inventory_histories['Componente'].unique()
df_compiled["In_Inventory_Histories"] = df_compiled["Cod_Produto"].isin(
    components_inventory_histories)

df_compiled = df_compiled.merge(df_produtos, how="left", left_on=[
                                "Cod_Produto"], right_on=["B1_COD"])

# Aplicação de regras de negócio
df_compiled.loc[df_compiled["B1_ATIVO"] == "N", "Excecao"] = True
df_compiled.loc[df_compiled["Sob_Demanda"] == True, "Ativo"] = False
df_compiled.loc[df_compiled["Excecao"] == True, "Ativo"] = False

df_compiled["Cod_Produto"] = clean_whitespace(df_compiled, "Cod_Produto")

# Aplica a função universal de filtragem
df_compiled = filter_t_components(df_compiled, 'Cod_Produto')

# 2. Aplica a função de remoção de linhas com dois pontos
df_compiled = remove_double_dot_lines(df_compiled, 'Cod_Produto')

df_compiled["Cod_Fornecedor"] = (
    df_compiled["Cod_Fornecedor"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.replace("nan", "", regex=False)
)

Helpers.save_output_dataset(
    context=context, output_name='new_compiled_components2', data_frame=df_compiled)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🧱 Preparação de Dados - Vendas
df_vendas = Helpers.getEntityData(context, "vendas")

# Manter apenas componentes que comecam com a letra T
df_vendas = filter_t_components(df_vendas, 'B1_COD_PP')

# 2. Aplica a função de remoção de linhas com dois pontos
df_vendas = remove_double_dot_lines(df_vendas, 'B1_COD_PP')

Helpers.save_output_dataset(
    context=context, output_name='df_vendas', data_frame=df_vendas)