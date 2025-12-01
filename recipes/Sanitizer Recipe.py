# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Required imports (Mantenha todos os imports necessários no topo)
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
import pandas as pd
import numpy as np
import datetime
import logging
from collections import defaultdict

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def criar_df_sem_match(df_excluidos, etapa_nome, motivo):
    """
    Adiciona colunas de controle ao dataframe de registros excluídos.
    
    Parâmetros:
    -----------
    df_excluidos : DataFrame
        DataFrame com os registros que foram excluídos nesta etapa
    etapa_nome : str
        Nome da etapa/script (ex: "01_filtro_inicial", "02_validacao_estoque")
    motivo : str
        Descrição do motivo da exclusão
    
    Retorna:
    --------
    DataFrame com colunas adicionais: Etapa, Motivo, Data_Exclusao
    """
    import datetime
    
    df_sem_match = df_excluidos.copy()
    df_sem_match['Etapa'] = etapa_nome
    df_sem_match['Motivo'] = motivo
    df_sem_match['Data_Exclusao'] = datetime.datetime.now()
    
    return df_sem_match

def clean_whitespace(df, column):
    """Remove espaços em branco no início e fim de uma coluna de string."""
    # Garante que a coluna é do tipo string antes de aplicar .str.strip()
    return df[column].astype(str).str.strip()
    
def filter_t_components(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtra linhas onde a coluna especificada começa com 'T' seguida por um ou mais dígitos.
    
    Exemplo: 'T123', 'T9' são mantidos; 'TA', 't1' ou '1T' são removidos.
    
    Retorna:
    --------
    tuple: (df_mantidos, df_excluidos)
        - df_mantidos: DataFrame com registros que começam com 'T' + dígitos
        - df_excluidos: DataFrame com registros que NÃO começam com 'T' + dígitos
    """
    # Criar máscara para identificar linhas que atendem o critério
    mask = df[column].astype(str).str.match(r'^T\d+')
    
    df_mantidos = df[mask].copy()
    df_excluidos = df[~mask].copy()
    
    return df_mantidos, df_excluidos


def remove_double_dot_lines(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove linhas em que a coluna selecionada contém exatamente dois pontos ".".
    
    Exemplo removido: T9708.50.030
    
    Retorna:
    --------
    tuple: (df_mantidos, df_excluidos)
        - df_mantidos: DataFrame com registros que NÃO têm exatamente dois pontos
        - df_excluidos: DataFrame com registros que têm exatamente dois pontos
    """
    # Criar máscara para identificar linhas que NÃO têm exatamente 2 pontos
    mask = df[column].astype(str).str.count(r'\.') != 2
    
    df_mantidos = df[mask].copy()
    df_excluidos = df[~mask].copy()
    
    return df_mantidos, df_excluidos

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

    merged_df = pd.merge(new_df, df, on=[product_column, date_column], how='left')

    # Aplica os preenchimentos conforme o dicionário
    if fill_dict:
        for col, method in fill_dict.items():
            if col not in merged_df.columns:
                continue  # ignora colunas inexistentes
            if method == "ffill":
                merged_df[col] = merged_df.groupby(product_column)[col].fillna(method="ffill")
            elif method == "bfill":
                merged_df[col] = merged_df.groupby(product_column)[col].fillna(method="bfill")
            else:
                merged_df[col] = merged_df[col].fillna(method)

    # Preenche o restante com 0
    merged_df = merged_df.fillna(0)

    return merged_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## 🧱 Preparação de Dados - Produtos nao entregues pela exportacao

df_products_not_delivered_exportation = Helpers.getEntityData(context, "products_not_delivered_exportation")
df_products_not_delivered_exportation = df_products_not_delivered_exportation.dropna(subset=['DESCRICAO', 'QUANT_BO'])
df_products_not_delivered_exportation['PRODUTO'] = df_products_not_delivered_exportation['PRODUTO'].astype(str).str.strip()
df_products_not_delivered_exportation['DESCRICAO'] = df_products_not_delivered_exportation['DESCRICAO'].astype(str).str.strip()
df_products_not_delivered_exportation['QUANT_BO'] = pd.to_numeric(df_products_not_delivered_exportation['QUANT_BO'].astype(str).str.strip(), errors='coerce')
df_products_not_delivered_exportation['DT_ENTREGA'] = pd.to_datetime(df_products_not_delivered_exportation['DT_ENTREGA'].astype(str).str.strip(), format='%d/%m/%Y', errors='coerce')
Helpers.save_output_dataset(context=context, output_name="df_products_not_delivered_exportation", data_frame=df_products_not_delivered_exportation)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## 🧱 Preparação de Dados - Estrutura do Produto (BOM)
df_estrutura_produto = Helpers.getEntityData(context, 'estrutura_produto')

# Aplica a função universal de filtragem
df_estrutura_produto, df_excluidos_1A = filter_t_components(df_estrutura_produto, 'COD_COMPONENTE')
df_excluidos_1A['origem'] = 'df_estrutura_produto'
df_excluidos_1A['Motivo'] = 'Component não começa com T seguido de dígitos'
df_excluidos_1A = df_excluidos_1A.rename(columns={'COD_COMPONENTE': 'Cod_component'})


# 2. Aplica a função de remoção de linhas com dois pontos
df_estrutura_produto, df_excluidos_1B = remove_double_dot_lines(df_estrutura_produto, 'COD_COMPONENTE')
df_excluidos_1B['origem'] = 'df_estrutura_produto'
df_excluidos_1B['Motivo'] = 'Component com dois . em seu nome'
df_excluidos_1B = df_excluidos_1B.rename(columns={'COD_COMPONENTE': 'Cod_component'})


df_excluidos_1 = pd.concat([df_excluidos_1A, df_excluidos_1B])

Helpers.save_output_dataset(context=context, output_name='df_estrutura_produto', data_frame=df_estrutura_produto)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## 🧱 Preparação de Dados - Produto Fornecedor 
df_produto_fornecedor = Helpers.getEntityData(context, "produto_fornecedor")
df_produto_fornecedor["PRODUTO"] = clean_whitespace(df_produto_fornecedor, "PRODUTO")

# Aplica a função universal de filtragem
df_produto_fornecedor, df_excluidos_2A = filter_t_components(df_produto_fornecedor, 'PRODUTO')
df_excluidos_2A['origem'] = 'df_produto_fornecedor'
df_excluidos_2A['Motivo'] = 'Component não começa com T seguido de dígitos'
df_excluidos_2A = df_excluidos_2A.rename(columns={'PRODUTO': 'Cod_component'})


# 2. Aplica a função de remoção de linhas com dois pontos
df_produto_fornecedor, df_excluidos_2B = remove_double_dot_lines(df_produto_fornecedor, 'PRODUTO')
df_excluidos_2B['origem'] = 'df_produto_fornecedor'
df_excluidos_2B['Motivo'] = 'Component com dois . em seu nome'
df_excluidos_2B = df_excluidos_2B.rename(columns={'PRODUTO': 'Cod_component'})

df_excluidos_2 = pd.concat([df_excluidos_2A, df_excluidos_2B])

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
string_cols = ["PRODUTO", "COD_FORNE", "COD_FABRI", "FORNEC_NOM", "MOEDA", "COD_PROD_FOR"]

# 4. Aplica o tipo string (pd.StringDtype) nas colunas definidas
df_produto_fornecedor[string_cols] = df_produto_fornecedor[string_cols].astype("string")

# 5. Conversão e arredondamento
float_cols = ["CUSTO_PRODUTO", "PTAX"]
df_produto_fornecedor[float_cols] = df_produto_fornecedor[float_cols].apply(pd.to_numeric, errors="coerce").round(2)

Helpers.save_output_dataset(context=context, output_name='df_produto_fornecedor', data_frame=df_produto_fornecedor)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## 📜 Histórico de Inventário e Backcasting (Formato Longo)

# --- Importa
df_estoque_atual = Helpers.getEntityData(context, 'estoque_atual')
df_estoque_atual["CODIGO_PI"] = clean_whitespace(df_estoque_atual, "CODIGO_PI")

# Aplica a função universal de filtragem
df_estoque_atual, df_excluidos_3A = filter_t_components(df_estoque_atual, 'CODIGO_PI')
df_excluidos_3A['origem'] = 'df_estoque_atual'
df_excluidos_3A['Motivo'] = 'Component não começa com T seguido de dígitos'
df_excluidos_3A = df_excluidos_3A.rename(columns={'CODIGO_PI': 'Cod_component'})


# 2. Aplica a função de remoção de linhas com dois pontos
df_estoque_atual, df_excluidos_3B = remove_double_dot_lines(df_estoque_atual, 'CODIGO_PI')
df_excluidos_3B['origem'] = 'df_estoque_atual'
df_excluidos_3B['Motivo'] = 'Component com dois . em seu nome'
df_excluidos_3B = df_excluidos_3B.rename(columns={'CODIGO_PI': 'Cod_component'})

df_excluidos_3 = pd.concat([df_excluidos_3A, df_excluidos_3B])

# --- Importa
df_movimentacao_estoque = Helpers.getEntityData(context, 'movimentacao_estoque')
df_movimentacao_estoque["B1_COD"] = clean_whitespace(df_movimentacao_estoque, "B1_COD")

# Aplica a função universal de filtragem
df_movimentacao_estoque, df_excluidos_4A = filter_t_components(df_movimentacao_estoque, 'B1_COD')
df_excluidos_4A['origem'] = 'df_movimentacao_estoque'
df_excluidos_4A['Motivo'] = 'Component não começa com T seguido de dígitos'
df_excluidos_4A = df_excluidos_4A.rename(columns={'B1_COD': 'Cod_component'})

# 2. Aplica a função de remoção de linhas com dois pontos
df_movimentacao_estoque, df_excluidos_4B = remove_double_dot_lines(df_movimentacao_estoque, 'B1_COD')
df_excluidos_4B['origem'] = 'df_movimentacao_estoque'
df_excluidos_4B['Motivo'] = 'Component com dois . em seu nome'
df_excluidos_4B = df_excluidos_4B.rename(columns={'B1_COD': 'Cod_component'})

df_excluidos_4 = pd.concat([df_excluidos_4A, df_excluidos_4B])

## Manipulacao
set_estoque_atual = df_estoque_atual["CODIGO_PI"].unique()

# --- Pré-processamento e Preenchimento de Dias Ausentes ---
df_movimentacao_estoque["DTA_LANCAMENTO"] = pd.to_datetime(df_movimentacao_estoque["DTA_LANCAMENTO"])

df_movimentacao_estoque_alldays = fill_all_missing_periods(
    df_movimentacao_estoque, 
    "DTA_LANCAMENTO", 
    "B1_COD", 
    freq="D", 
    fill_dict={"QTD_LANCADA": 0} 
)

min_date = df_movimentacao_estoque_alldays["DTA_LANCAMENTO"].min()
max_date = df_movimentacao_estoque_alldays["DTA_LANCAMENTO"].max()
df_inventory_histories_base = pd.DataFrame( # Renomeado para evitar conflito
    index=pd.date_range(start=min_date, end=max_date, freq="D")
)

# --- Backcasting (Cálculo Retroativo) ---
estoque_por_part = dict(tuple(df_estoque_atual.groupby("CODIGO_PI")))
movimentacoes_por_part = dict(tuple(df_movimentacao_estoque_alldays.groupby("B1_COD")))
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

    df_deltas_reversed = df_deltas.sort_values(by="DTA_LANCAMENTO", ascending=False)
    stock = df_part_number["QTD_TOT_EST"].iloc[-1]
    unique_dates = df_deltas_reversed["DTA_LANCAMENTO"].unique()

    for day in df_inventory_histories_base.index[::-1]:
        if day < min_date:
            continue
            
        if day in unique_dates:
            delta = df_deltas_reversed[df_deltas_reversed["DTA_LANCAMENTO"] == day]["QTD_LANCADA"].sum()
            stock -= delta
            
        new_inventory_histories[day][part_number] = stock

# --- Finalização do DataFrame de Histórico (Cria Pivotado, DERRETE para Longo) ---

# 1. Cria o DataFrame pivotado (Este passo é necessário para usar o resultado do backcasting)
df_inventory_histories_pivot = pd.DataFrame.from_dict(new_inventory_histories, orient="index").sort_index()

# 2. Renomeia colunas (mantida a lógica de substituir "_" por "/")
df_inventory_histories_pivot.columns = df_inventory_histories_pivot.columns.astype(str).str.replace("_", "/")
df_inventory_histories_pivot = df_inventory_histories_pivot.reset_index().rename(columns={'index': 'date'})

# 3. DESPIVOTAMENTO
df_inventory_histories = df_inventory_histories_pivot.melt(
    id_vars=['date'],                      # Coluna a ser mantida (Datas)
    var_name='Componente',                 # Nova coluna para os nomes dos componentes (T123, T456)
    value_name='QTD_ESTOQUE'               # Nova coluna para os valores de estoque
)

# 4. Limpeza final e tipo
df_inventory_histories['Componente'] = df_inventory_histories['Componente'].astype(str)

Helpers.save_output_dataset(context=context, output_name='new_inventory_histories2', data_frame=df_inventory_histories)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## 🧩 Compilação e Classificação de Componentes (Metadados)
df_produtos = Helpers.getEntityData(context, 'produtos')
df_sobdemanda = Helpers.getEntityData(context, 'produtos_sob_demanda')
df_ativos = Helpers.getEntityData(context, 'produto_fornecedor_ativo')
df_excecoes = Helpers.getEntityData(context, 'excecoes_produtos_sem_fornecedores')
df_parametro_fornecedor = Helpers.getEntityData(context, 'parametro_fornecedor')

# --- Limpeza e Marcação ---
# (Manter a lógica de remoção de duplicatas e criação de colunas de flag)
df_ativos = df_ativos.drop_duplicates(subset=['Cod_Produto']).assign(Ativo=True)
df_excecoes = df_excecoes.drop_duplicates(subset=['Cod_Produto']).assign(Excecao=True)
df_sobdemanda = df_sobdemanda.drop_duplicates(subset=['Cod_Produto']).assign(Sob_Demanda=True)

# --- Junção dos Metadados ---
df_compiled = df_ativos.merge(df_excecoes, how="outer", on="Cod_Produto")
df_compiled = df_compiled.merge(df_sobdemanda, how="outer", on="Cod_Produto")
df_compiled = df_compiled.fillna({"Ativo": False, "Excecao": False, "Sob_Demanda": False})

# --- Adicionar Histórico de Estoque e Regras de Negócio ---
components_inventory_histories = df_inventory_histories['Componente'].unique()
df_compiled["In_Inventory_Histories"] = df_compiled["Cod_Produto"].isin(components_inventory_histories)

df_compiled = df_compiled.merge(df_produtos, how="left", left_on=["Cod_Produto"], right_on=["B1_COD"])

# Aplicação de regras de negócio
df_compiled.loc[df_compiled["B1_ATIVO"] == "N", "Excecao"] = True
df_compiled.loc[df_compiled["Sob_Demanda"] == True, "Ativo"] = False
df_compiled.loc[df_compiled["Excecao"] == True, "Ativo"] = False

df_compiled["Cod_Produto"] = clean_whitespace(df_compiled, "Cod_Produto")

# Aplica a função universal de filtragem
df_compiled, df_excluidos_5A = filter_t_components(df_compiled, 'Cod_Produto')
df_excluidos_5A['origem'] = 'df_compiled (df_excecoes + df_sobdemanda)'
df_excluidos_5A['Motivo'] = 'Component não começa com T seguido de dígitos'
df_excluidos_5A = df_excluidos_5A.rename(columns={'Cod_Produto': 'Cod_component'})

# 2. Aplica a função de remoção de linhas com dois pontos
df_compiled, df_excluidos_5B = remove_double_dot_lines(df_compiled, 'Cod_Produto')
df_excluidos_5B['origem'] = 'df_compiled (df_excecoes + df_sobdemanda)'
df_excluidos_5B['Motivo'] = 'Component com dois . em seu nome'
df_excluidos_5B = df_excluidos_5B.rename(columns={'Cod_Produto': 'Cod_component'})

df_excluidos_5 = pd.concat([df_excluidos_5A, df_excluidos_5B])

df_compiled["Cod_Fornecedor"] = (
    df_compiled["Cod_Fornecedor"]
    .astype(str)                              
    .str.replace(r"\.0$", "", regex=True)     
    .str.replace("nan", "", regex=False)      
)

Helpers.save_output_dataset(context=context, output_name='new_compiled_components2', data_frame=df_compiled)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## 🧱 Preparação de Dados - Vendas
df_vendas = Helpers.getEntityData(context, "vendas")

### Manter apenas componentes que comecam com a letra T
df_vendas, df_excluidos_7A = filter_t_components(df_vendas, 'B1_COD_PP')
df_excluidos_7A['origem'] = 'df_vendas'
df_excluidos_7A['Motivo'] = 'Component não começa com T seguido de dígitos'

# 2. Aplica a função de remoção de linhas com dois pontos
df_vendas, df_excluidos_7B = remove_double_dot_lines(df_vendas, 'B1_COD_PP')
df_excluidos_7B['origem'] = 'df_vendas'
df_excluidos_7B['Motivo'] = 'Component com dois . em seu nome'

df_excluidos_7 = pd.concat([df_excluidos_7A, df_excluidos_7B]).rename(columns={'B1_COD_PP':'Cod_component'})

Helpers.save_output_dataset(context=context, output_name='df_vendas', data_frame=df_vendas)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Oraganizando os dados excluidos
df_excluidos_1 =  df_excluidos_1[['Cod_component', 'origem', 'Motivo']]
df_excluidos_2 = df_excluidos_2[['Cod_component', 'origem', 'Motivo']]
df_excluidos_3 = df_excluidos_3[['Cod_component', 'origem', 'Motivo']]
df_excluidos_4 = df_excluidos_4[['Cod_component', 'origem', 'Motivo']]
df_excluidos_5 = df_excluidos_5[['Cod_component', 'origem', 'Motivo']]
# df_excluidos_6 = df_excluidos_6[['Cod_component', 'origem', 'Motivo']]
df_excluidos_7 = df_excluidos_7[['Cod_component', 'origem', 'Motivo']]


dfs_excluidos = [
    df_excluidos_1,
    df_excluidos_2,
    df_excluidos_3,
    df_excluidos_4,
    df_excluidos_5,
    # df_excluidos_6,
    df_excluidos_7,
]

# Filtrar apenas os que não estão vazios
dfs_excluidos = [df for df in dfs_excluidos if len(df) > 0]

# Concatenar se houver dataframes
if dfs_excluidos:
    df_sem_match_atual = pd.concat(dfs_excluidos, ignore_index=True)
    
df_sem_match_atual

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
Helpers.save_output_dataset(context=context, output_name='df_sem_match_atual_1', data_frame=df_sem_match_atual)

