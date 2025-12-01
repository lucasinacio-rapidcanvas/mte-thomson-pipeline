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
from utils.rc.client.auth import AuthClient
from utils.rc.client.requests import Requests
from utils.rc.dtos.user import User

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pandas as pd
import numpy as np
from scipy.stats import norm
import math
import os
import itertools
import datetime
from dateutil.relativedelta import relativedelta
from typing import List
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

str_token = Helpers.get_user_token(context)
Requests.setToken(str_token)

# Constantes
DAYS_BEFORE_SHIPMENT = 15

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Important functions
def print_df_info(df_name, df):
    """Imprime informações básicas de um DataFrame"""
    print(f"\n{'='*70}")
    print(f"📊 DataFrame: {df_name}")
    print(f"{'='*70}")
    print(f"   Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"   Colunas: {list(df.columns)}")
    if len(df) > 0:
        print(f"   Primeiras colunas: {df.columns[:5].tolist()}")
    print(f"{'='*70}\n")
    
def get_in_transit_orders(df_pedidos_pendentes: pd.DataFrame, df_produtos: pd.DataFrame, df_artifact=pd.DataFrame()):
    """Processa pedidos em trânsito"""
    today = datetime.datetime.now()

    df_produtos = df_produtos[df_produtos["B1_ATIVO"]=='S']
    df_produtos = df_produtos[["B1_COD","B1_GRUPO"]]
    df_pedidos_pendentes = df_pedidos_pendentes.merge(df_produtos, left_on="PRODUTO", right_on="B1_COD", how="left")

    df_pedidos_pendentes = df_pedidos_pendentes.replace("nan", None)
    df_pedidos_pendentes["DT. Prod. Real"] = ""
    df_pedidos_pendentes["OBS 1"] = ""
    df_pedidos_pendentes["OBS 2"] = ""

    df_pedidos_pendentes["DT. Prod. Prev."] = (
        df_pedidos_pendentes["EMBARQUE_PREVISTO_PO"] - pd.Timedelta(DAYS_BEFORE_SHIPMENT, unit="D")
    )
    
    df_pedidos_pendentes["Status Pedido"] = np.where(
        df_pedidos_pendentes["PROCESSO"].isna(), "Negociação", "Transito"
    )
    
    df_pedidos_pendentes["Alerta"] = np.where(
        (today > df_pedidos_pendentes["DT. Prod. Prev."]) & (pd.isna(df_pedidos_pendentes['DT. Prod. Real'])), 
        "Prod. Atrasada", 
        "Em Progresso"
    )

    col_names = {
        "Alerta":"Alerta",
        "DATA_SI":"Data SI",
        "NUMERO_SI":"SI",
        "B1_GRUPO":"Grupo",
        "PRODUTO":"MTE",
        "CODIGO_X_MTE": "MTE X",
        "QTDE_NAO_ENTREGUE":"Qtd",
        "ENTREGA_PREVISTA_PO": "DT. Entrega PO",
        "CHEG_PORTO_ETA_15": "Entrega Prevista",
        "FORNEC_NOM": "Fornecedor",
        "COD_PROD_FOR": "Cod Prod. Forn.",
        "PROFORMA": "Proforma",
        "PEDIDO": "PO",
        "INVOICE": "Invoice",
        "PROCESSO": "Código Embarque",
        "CONFIRMACAO_PEDIDO": "Conf. PO",
        "DT. Prod. Prev.": "DT. Prod. Prev.",
        "DT. Prod. Real": "DT. Prod. Real",
        "EMBARQUE_EFET": "Data Embarque",
        "CHEG_PORTO_ETA": "Data Prevista Chegada Porto",
        "Status Pedido": "Status Pedido",
        "OBS 1": "OBS 1",
        "OBS 2": "OBS 2"
    }

    df_pedidos_pendentes.rename(columns=col_names, inplace=True)
    df_pedidos_pendentes = df_pedidos_pendentes[col_names.values()]
    
    if not df_artifact.empty:
        df_artifact['key'] = df_artifact['MTE'].astype(str) + df_artifact['PO'].astype(str)
        df_pedidos_pendentes['key'] = df_pedidos_pendentes['MTE'].astype(str) + df_pedidos_pendentes['PO'].astype(str)
        df_artifact['Data Prod Artifact'] = df_artifact["DT. Prod. Real"]
        df_artifact = df_artifact[['key', 'Data Prod Artifact']]
        df_pedidos_pendentes = pd.merge(df_pedidos_pendentes, df_artifact, on='key', how='left')
        df_pedidos_pendentes["DT. Prod. Real"] = df_pedidos_pendentes['Data Prod Artifact']
        df_pedidos_pendentes = df_pedidos_pendentes.drop(columns=['key', 'Data Prod Artifact'])
        
        df_pedidos_pendentes['Alerta'] = np.where(
            pd.notna(df_pedidos_pendentes['DT. Prod. Real']),
            np.where(
                pd.to_datetime(df_pedidos_pendentes["DT. Prod. Real"], format="%d/%m/%Y") > df_pedidos_pendentes['DT. Prod. Prev.'],
                "Produzido após prazo",
                "Produzido"
            ),
            np.where(
                today > df_pedidos_pendentes["DT. Prod. Prev."], 
                "Prod. Atrasada",
                "Em Progresso"
            )
        )
        
    # Manter DT. Prod. Prev. como datetime para uso interno
    # Formatar apenas para exibição quando necessário
    df_pedidos_pendentes["Data SI"] = df_pedidos_pendentes["Data SI"].dt.strftime("%d-%b-%Y")
    df_pedidos_pendentes["DT. Entrega PO"] = df_pedidos_pendentes["DT. Entrega PO"].dt.strftime("%d-%b-%Y")
    df_pedidos_pendentes["Entrega Prevista"] = df_pedidos_pendentes["Entrega Prevista"].dt.strftime("%d-%b-%Y")
    df_pedidos_pendentes["Data Embarque"] = df_pedidos_pendentes["Data Embarque"].dt.strftime("%d-%b-%Y")
    df_pedidos_pendentes["Data Prevista Chegada Porto"] = df_pedidos_pendentes["Data Prevista Chegada Porto"].dt.strftime("%d-%b-%Y")
    
    return df_pedidos_pendentes

def min_order_value_warning(df_main, df_faturamento_minimo):
    """Retorna a lista dos fornecedores que atingiram o mínimo no faturamento"""
    supplier_list = df_main["Supp Cod"].unique()
    all_suppliers_reached = []
    
    for supplier in supplier_list:
        df_order_supplier = df_main[df_main["Supp Cod"] == supplier]
        if not df_order_supplier.empty:
            if 'Final_order' in df_order_supplier.columns and 'Cost' in df_order_supplier.columns:
                total = (df_order_supplier['Final_order'] * df_order_supplier['Cost']).sum()
            else:
                total = 0
                
            try:
                col_supplier = None
                if 'Supp Code' in df_faturamento_minimo.columns:
                    col_supplier = 'Supp Code'
                else:
                    possible_cols = [col for col in df_faturamento_minimo.columns 
                                   if 'cod' in col.lower() or 'supp' in col.lower()]
                    if possible_cols:
                        col_supplier = possible_cols[0]
                    else:
                        continue
                
                df_faturamento_minimo[col_supplier] = df_faturamento_minimo[col_supplier].astype('str')
                aux_fat_min = df_faturamento_minimo[df_faturamento_minimo[col_supplier] == str(supplier)]
                
                if not aux_fat_min.empty:
                    aux_fat_min = aux_fat_min.reset_index(drop=True)
                    
                    fat_min_col = None
                    if 'Fatur.Min.' in aux_fat_min.columns:
                        fat_min_col = 'Fatur.Min.'
                    elif 'Faturamento_M_nimo' in aux_fat_min.columns:
                        fat_min_col = 'Faturamento_M_nimo'
                    else:
                        continue
                        
                    fat_min = aux_fat_min.loc[0, fat_min_col]
                    if total >= fat_min:
                        all_suppliers_reached.append(supplier)
                        
            except Exception as e:
                logger.warning(f"Erro ao processar fornecedor {supplier}: {e}")
                continue

    return all_suppliers_reached

def calculate_projected_level(usage_date, lt, rp, current_level, Final_order, demand, df_intransit):
    """
    Calcula o nível de estoque projetado ao longo do tempo.
    
    Retorna:
        tuple: (DataFrame com histórico, total de vendas perdidas)
    """
    first_date = usage_date + pd.Timedelta(1, unit="D")
    arrival_date_current_order = usage_date + pd.Timedelta(lt, unit="D")
    final_date = usage_date + pd.Timedelta(lt + rp, unit="D")

    demand_per_day = demand / (lt + rp) if (lt + rp) > 0 else 0
    date_range = pd.date_range(first_date, final_date, freq="D")
    projected_level = current_level
    total_lost = 0
    
    df_intransit = df_intransit.copy()
    df_intransit["DT. Entrega PO"] = pd.to_datetime(
        df_intransit["DT. Entrega PO"], 
        format="%d-%b-%Y",
        errors='coerce'
    )
    
    history_records = []
    
    for date in date_range:
        arrivals = df_intransit[df_intransit["DT. Entrega PO"] == date]["Qtd"].sum()
        projected_level += arrivals
        
        if date == arrival_date_current_order:
            projected_level += Final_order
        
        if projected_level >= demand_per_day:
            projected_level -= demand_per_day
        else:
            total_lost += demand_per_day - projected_level
            projected_level = 0
        
        history_records.append({
            "Date": date, 
            "Projected level": projected_level
        })
    
    df_projected_level_history = pd.DataFrame(history_records)
    
    return df_projected_level_history, total_lost

def move_column(df, col_to_move, reference_col, position='after'):
    """Move uma coluna para próximo de outra coluna de referência"""
    cols = list(df.columns)
    cols.remove(col_to_move)
    index = cols.index(reference_col)
    
    if position == 'after':
        cols.insert(index + 1, col_to_move)
    elif position == 'before':
        cols.insert(index, col_to_move)
    
    return df[cols]

def get_sales_last_months(df_vendas_raw):
    df_vendas = df_vendas_raw[['B1_COD_PP', 'DATA', 'QTD_VENDA_NACIONAL', 'QTD_VENDA_EXPORTACAO']]
    df_vendas.columns = ['Component', 'Data', 'QTD_VENDA_NACIONAL', 'QTD_VENDA_EXPORTACAO']
    df_vendas['Qtd'] = df_vendas['QTD_VENDA_NACIONAL'] + df_vendas['QTD_VENDA_EXPORTACAO']
    df_vendas['Periodo'] = df_vendas['Data'].dt.to_period('M').astype(str)
    df_vendas_periodo = df_vendas.groupby(['Component', 'Periodo'])['Qtd'].sum().reset_index()
    today = pd.Timestamp.today()
    meses = pd.period_range(end=today - pd.offsets.MonthBegin(1), periods=6, freq='M').strftime('%Y-%m').tolist()
    df_vendas_periodo = df_vendas_periodo[df_vendas_periodo['Periodo'].isin(meses)]
    df_pivot = df_vendas_periodo.pivot_table(
        index='Component',
        columns='Periodo',
        values='Qtd'
    )
    df_pivot = df_pivot.reindex(columns=meses, fill_value=0)
    for col in meses:
        try:
            month_num = int(col[-2:])
            new_name = f"Sales-M{month_num}"
            df_pivot.rename(columns={col: new_name}, inplace=True)
        except:
            continue
    
    df_vendas_periodo  = df_pivot.reset_index()
    df_vendas_periodo  = df_vendas_periodo .fillna(0)

    return df_vendas_periodo

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Carregar DataFrames de histórico de pedidos
df_historico_pedidos_realizados = Helpers.getEntityData(context, "df_historico_pedidos_realizados")
print_df_info("df_historico_pedidos_realizados", df_historico_pedidos_realizados)
df_historico_pedidos_realizados['MES'] = pd.to_datetime(df_historico_pedidos_realizados['MES'])

df_historico_pedidos_previstos = Helpers.getEntityData(context, "df_historico_pedidos_previstos")
print_df_info("df_historico_pedidos_previstos", df_historico_pedidos_previstos)
df_historico_pedidos_previstos['MES'] = pd.to_datetime(df_historico_pedidos_previstos['MES'])

# df_produtos
df_produtos = Helpers.getEntityData(context, "produtos")
print_df_info("df_produtos", df_produtos)

# df_pedidos_pendentes
df_pedidos_pendentes = Helpers.getEntityData(context, "pedidos_pendentes")
print_df_info("df_pedidos_pendentes", df_pedidos_pendentes)

dtypes = {
    "DATA_SI": "datetime64[ns]",
    "ENTREGA_PREVISTA_PO": "datetime64[ns]",
    "CHEG_PORTO_ETA": "datetime64[ns]",
    "CHEG_PORTO_ETA_15": "datetime64[ns]",
    "EMBARQUE_PREVISTO_PO": "datetime64[ns]",
    "EMBARQUE_EFET": "datetime64[ns]",
}
df_pedidos_pendentes = df_pedidos_pendentes.astype(dtypes)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Inicializa a lista de auditoria
lista_dfs_sem_match = []

# df_inventory_histories
df_inventory_histories = Helpers.getEntityData(context, "new_inventory_histories2")
print_df_info("df_inventory_histories", df_inventory_histories)

# Garantir tipos corretos
df_inventory_histories['date'] = pd.to_datetime(df_inventory_histories['date'])
df_inventory_histories['Componente'] = df_inventory_histories['Componente'].astype(str)
df_inventory_histories['QTD_ESTOQUE'] = pd.to_numeric(df_inventory_histories['QTD_ESTOQUE'], errors='coerce')

# --- FILTRO 1: Remover linhas com dados inválidos ---
mask_inv_invalid = (
    df_inventory_histories['date'].isna() | 
    df_inventory_histories['Componente'].isna() | 
    (df_inventory_histories['Componente'] == 'nan')
)

if mask_inv_invalid.sum() > 0:
    df_removed_inv = df_inventory_histories[mask_inv_invalid].copy()
    df_audit_inv = df_removed_inv[['Componente']].rename(columns={'Componente': 'Cod_component'})
    df_audit_inv['origem'] = 'df_inventory_histories'
    df_audit_inv['motivo'] = 'Data ou Componente nulo/invalido'
    lista_dfs_sem_match.append(df_audit_inv)

df_inventory_histories = df_inventory_histories[~mask_inv_invalid]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Obter informações globais
last_available_date = df_inventory_histories['date'].max()
first_available_date = df_inventory_histories['date'].min()
available_components = set(df_inventory_histories['Componente'].unique())

print(f"✅ Histórico de estoque carregado (formato LONG):")
print(f"   - Registros: {len(df_inventory_histories):,}")
print(f"   - Período: {first_available_date.date()} a {last_available_date.date()}")
print(f"   - Componentes: {len(available_components)}")

# df_faturamento_minimo
df_faturamento_minimo = Helpers.getEntityData(context, "faturamento_minimo")
print_df_info("df_faturamento_minimo", df_faturamento_minimo)

df_faturamento_minimo.columns = ['Fornecedor', 'Supp Code', 'Fatur.Min.']
df_faturamento_minimo['Supp Code'] = df_faturamento_minimo['Supp Code'].astype('str')
df_faturamento_minimo['Supp Code'] = df_faturamento_minimo['Supp Code'].str.replace(',', '')

# df_monthly_portalvendas
df_monthly_portalvendas = Helpers.getEntityData(context, "new_monthly_portalvendas")
print_df_info("df_monthly_portalvendas", df_monthly_portalvendas)

dtypes = {"COD_MTE_COMP": "category", "MONTH": "datetime64[ns]", "QTDE_PEDIDA": "int"}
df_monthly_portalvendas = df_monthly_portalvendas.astype(dtypes)
df_monthly_portalvendas.rename(columns={"MONTH": "DATA_PEDIDO"}, inplace=True)

# df_main
df_main = Helpers.getEntityData(context, "main")
print_df_info("df_main", df_main)

df_main['LT+RP'] = df_main['LT'] + df_main['RP']

# df_vendas_raw
df_vendas_raw = Helpers.getEntityData(context, "vendas")
print_df_info("df_vendas_raw", df_vendas_raw)

# df_new_register
df_new_register = Helpers.getEntityData(context, "produtos")
print_df_info("df_new_register", df_new_register)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Adaptar para usar df_historico_pedidos_previstos
first_base_month = df_historico_pedidos_previstos["MES"].min()
last_base_month = df_historico_pedidos_previstos["MES"].max()
           
available_base_months = pd.date_range(start=first_base_month, end=last_base_month, freq="MS").to_list()
available_base_months.append("Last available")

base_month = "Last available"
if base_month == "Last available":
    base_month = last_base_month
    usage_date = df_inventory_histories['date'].max()
else:
    usage_date = base_month + pd.offsets.MonthEnd(0)

base_date = base_month + pd.offsets.MonthEnd(0)
days_since_base = (usage_date - base_date).days

if base_date not in df_inventory_histories['date'].values:
    base_date = df_inventory_histories['date'].max()

print(f"📅 Data base selecionada: {base_date.date()}")
print(f"📅 Data de uso: {usage_date.date()}")
print(f"📅 Dias desde a base: {days_since_base}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 🔧 Limpeza e preparação do df_main
print("🔧 Preparando df_main...")

# 1. RESOLVER DUPLICAÇÃO DE COLUNAS Component/Cod_X
if 'Component' in df_main.columns and 'Cod_X' in df_main.columns:
    print("   ⚠️ DETECTADA DUPLICAÇÃO: 'Component' e 'Cod_X' existem simultaneamente")
    if df_main['Component'].equals(df_main['Cod_X']):
        print("   ✅ Colunas são idênticas - removendo 'Cod_X'")
        df_main = df_main.drop(columns=['Cod_X'])
    else:
        print("   ⚠️ Colunas são DIFERENTES! Mantendo 'Component' e removendo 'Cod_X'")
        df_main = df_main.drop(columns=['Cod_X'])
elif 'Cod_X' in df_main.columns and 'Component' not in df_main.columns:
    print("   Renomeando 'Cod_X' → 'Component'")
    df_main = df_main.rename(columns={'Cod_X': 'Component'})
elif 'Component' in df_main.columns:
    print("   ✅ Apenas 'Component' existe (OK)")
else:
    raise ValueError("❌ Nem 'Component' nem 'Cod_X' encontradas!")

# 2. Garantir que Component é string
df_main['Component'] = df_main['Component'].astype(str)

# 3. FILTRAGEM COM AUDITORIA
print("   Filtrando dados...")
initial_rows = len(df_main)

# --- FILTRO 2: Fornecedor Nulo ---
mask_supp_na = df_main["Supp_Cod"].isna()

if mask_supp_na.sum() > 0:
    df_removed_supp = df_main[mask_supp_na].copy()
    df_audit_supp = df_removed_supp[['Component']].rename(columns={'Component': 'Cod_component'})
    df_audit_supp['origem'] = 'df_main_preprocessing'
    df_audit_supp['motivo'] = 'Supp_Cod (Codigo Fornecedor) nulo'
    lista_dfs_sem_match.append(df_audit_supp)

df_main = df_main[~mask_supp_na]
print(f"   Após remover Supp_Cod nulos: {len(df_main)} linhas")

# --- FILTRO 3: Formato de Componente Inválido ---
mask_bad_fmt = df_main["Component"].str.count(r"\.") >= 2

if mask_bad_fmt.sum() > 0:
    df_removed_fmt = df_main[mask_bad_fmt].copy()
    df_audit_fmt = df_removed_fmt[['Component']].rename(columns={'Component': 'Cod_component'})
    df_audit_fmt['origem'] = 'df_main_preprocessing'
    df_audit_fmt['motivo'] = 'Formato invalido (contem 2 ou mais pontos)'
    lista_dfs_sem_match.append(df_audit_fmt)

df_main = df_main[~mask_bad_fmt]
print(f"   Após filtrar pontos: {len(df_main)} linhas")

# 4. Limpeza de valores
df_main = df_main.replace("None", None)

# Remover colunas Order_sug se existirem
cols_to_remove = ['Order_sug', 'Order_sug_v2', 'Final_order']
for col in cols_to_remove:
    if col in df_main.columns:
        df_main = df_main.drop(columns=[col])
        print(f"   ✅ Coluna '{col}' removida")

# Garantir colunas numéricas essenciais
df_main["Demand_(LT+RP)"] = df_main["Demand_(LT+RP)"].fillna(0).astype(int)
df_main["LT"] = df_main["LT"].fillna(0).astype(int)
df_main["RP"] = df_main["RP"].fillna(0).astype(int)

print(f"✅ df_main preparado: {len(df_main)} linhas finais")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# CONSOLIDAÇÃO DO DATAFRAME DE AUDITORIA (df_sem_match)

if len(lista_dfs_sem_match) > 0:
    df_sem_match = pd.concat(lista_dfs_sem_match, ignore_index=True)
else:
    df_sem_match = pd.DataFrame(columns=['Cod_component', 'origem', 'motivo'])

df_sem_match = df_sem_match[['Cod_component', 'origem', 'motivo']].drop_duplicates()
Helpers.save_output_dataset(context=context, output_name='df_sem_match_atual_7', data_frame=df_sem_match)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Processar base_month e base_date se existirem
if 'base_month' in df_main.columns and 'base_date' in df_main.columns:
    base_month_main = df_main["base_month"].max()
    base_date_main = df_main["base_date"].max()
    df_main = df_main[df_main["base_month"] == base_month_main]
    df_main = df_main.drop(columns=["base_month", "base_date"])
    print(f"✅ Filtrado por base_month: {base_month_main}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Mover coluna Component para o início
if 'Component' in df_main.columns:
    component_col = df_main.pop('Component')
    df_main.insert(0, 'Component', component_col)
    print("✅ Coluna 'Component' movida para o início")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Renomear colunas e preparar estrutura
df_main.columns = df_main.columns.str.replace("_", " ")

# Remover colunas antigas de Order sug se ainda existirem
cols_to_drop = ['Total Stock Sales 12M', 'Order sug Sales 12M', 'Lost 12M', 'Order sug', 'Order sug v2']
existing_cols_to_drop = [col for col in cols_to_drop if col in df_main.columns]
if existing_cols_to_drop:
    df_main = df_main.drop(existing_cols_to_drop, axis=1)
    print(f"✅ Colunas removidas: {existing_cols_to_drop}")

# Adicionar coluna Obs se não existir
if 'Obs' not in df_main.columns:
    # Inserir após a primeira coluna que contém "Stock" ou no final
    stock_cols = [col for col in df_main.columns if 'Stock' in col]
    if stock_cols:
        ref_column_index = df_main.columns.get_loc(stock_cols[0])
        df_main.insert(ref_column_index + 1, 'Obs', '')
    else:
        df_main['Obs'] = ''

# Garantir tipos corretos
if 'Cost' in df_main.columns:
    df_main['Cost'] = df_main['Cost'].astype('float64').round(2)
if 'Total Cost' in df_main.columns:
    df_main['Total Cost'] = df_main['Total Cost'].astype('float64').round(2)
if 'Currency' in df_main.columns:
    df_main['Currency'] = df_main['Currency'].replace('nan', '')
if '% Export 12M' in df_main.columns:
    df_main['% Export 12M'] = df_main['% Export 12M'].round(5)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Identificar componentes com histórico
components_in_main = set(df_main['Component'].unique())
components_with_history = components_in_main.intersection(available_components)
components_without_history = components_in_main - available_components

print(f"\n📦 Status dos componentes:")
print(f"   ✅ Com histórico: {len(components_with_history)}")
print(f"   ⚠️ Sem histórico: {len(components_without_history)}")
print(f"   Taxa de cobertura: {len(components_with_history)/len(components_in_main)*100:.1f}%")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# OBTER SUGESTÕES DO MODELO A PARTIR DO HISTÓRICO DE PEDIDOS
# ============================================================================

min_month = df_historico_pedidos_previstos['MES'].min()
df_historico_pedidos_previstos = df_historico_pedidos_previstos[df_historico_pedidos_previstos['MES']==min_month]
df_historico_pedidos_previstos = df_historico_pedidos_previstos[['COMPONENT', 'COD_FORNE', 'QUANT_PREDITA']].rename(
    columns = {
        'COMPONENT':'Component',
        'COD_FORNE':'Supp Cod',
        'QUANT_PREDITA':'Final_order'
    })

df_main['Supp Cod'] = df_main['Supp Cod'].astype(int).astype(str)
df_main = pd.merge(
    df_main, 
    df_historico_pedidos_previstos, 
    on=['Component', 'Supp Cod'],
    how='left'
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Merging transit for the next months into df_main
df_pending_orders = get_in_transit_orders(df_pedidos_pendentes, df_produtos)
df_pending_orders['DT. Entrega PO'] = pd.to_datetime(df_pending_orders['DT. Entrega PO'], format="%d-%b-%Y", errors='coerce')
df_pending_orders['year_month'] = df_pending_orders['DT. Entrega PO'].dt.strftime('%Y-%m')
df_transit_next_months = df_pending_orders.groupby(['MTE', 'year_month'])['Qtd'].sum().reset_index()
df_transit_next_months = df_transit_next_months.pivot_table(index='MTE', columns='year_month', values='Qtd')
df_transit_next_months.columns = ['Transit ' + col for col in df_transit_next_months.columns]
df_transit_next_months = df_transit_next_months.reset_index()      
df_transit_next_months = df_transit_next_months.iloc[:, list([0]) + list(range(-4, 0))]

df_main = pd.merge(df_main, df_transit_next_months, left_on='Component', right_on='MTE', how='left')
df_main = df_main.drop(columns='MTE')

cols_to_fill = [col for col in df_main.columns if col.startswith("Sales") or col.startswith("Transit")]
df_main[cols_to_fill] = df_main[cols_to_fill].fillna(0)
df_main["Supp Cod"] = df_main["Supp Cod"].astype(str).fillna("-")
cols_to_replace = [col for col in ["Supplier", "ABC"] if col in df_main.columns]
df_main[cols_to_replace] = df_main[cols_to_replace].fillna("-")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Date of registration and first sale
df_new_register = df_new_register[['B1_COD', 'DTA_CADASTRO']]

df_first_sale = df_vendas_raw[['B1_COD_PP', 'DATA']]
df_first_sale = df_first_sale.groupby('B1_COD_PP')['DATA'].min().reset_index()
df_new_products = pd.merge(df_new_register, df_first_sale, left_on='B1_COD', right_on='B1_COD_PP', how='left')
df_new_products.rename(columns={'DATA': 'first_sale_date'}, inplace=True)

one_year_ago = datetime.datetime.now() - datetime.timedelta(days=365)
four_years_ago = datetime.datetime.now() - datetime.timedelta(days=365*4)

def calculate_new_product(row):
    if row['DTA_CADASTRO'] < four_years_ago:
        return False
    else:
        if pd.isna(row['first_sale_date']):
            return True
        elif row['first_sale_date'] >= one_year_ago:
            return True
        else:
            return False

df_new_products['NewProduct'] = df_new_products.apply(calculate_new_product, axis=1)

df_main = pd.merge(df_main, df_new_products, left_on='Component', right_on='B1_COD', how='left')
df_main = df_main.drop(['B1_COD', 'B1_COD_PP', 'DTA_CADASTRO', 'first_sale_date'], axis=1)

# Get exceptions
df_exceptions = Helpers.getEntityData(context, "excecoes_produtos_sem_fornecedores") 
df_main['IsException'] = df_main['Component'].apply(lambda x: x in df_exceptions['Cod_Produto'].values)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Classificação ABC
df_main = df_main.sort_values(by='Sales 12M', ascending=False)

df_main['Participacao'] = (df_main['Sales 12M']/df_main['Sales 12M'].fillna(0).sum()*100).round(2)
df_main['Participacao Acumulada'] = df_main['Participacao'].cumsum().round(2)

conditions = [
    df_main['Participacao Acumulada'] < 80,
    df_main['Participacao Acumulada'] < 90,
    df_main['Participacao Acumulada'] < 95,
    df_main['Participacao Acumulada'] > 95
]

choices = ['A', 'B', 'C', 'D']

df_main['New ABC'] = np.select(conditions, choices, default=np.nan)
df_main['check abc'] = df_main['ABC'] == df_main['New ABC']

df_main['Venda mensal'] = np.rint(df_main['Sales 12M']/12)
df_main['Venda mensal'] = df_main['Venda mensal'].fillna(0)

df_main['Alcance - Estoque Atual'] = (
    (df_main["Stock"].fillna(0)) /
    (df_main["Venda mensal"].replace(0, np.nan))
).round(0).fillna(0).astype(int)

df_main['Alcance - Estoque Total'] = (
    (df_main["Total Stock"].fillna(0)) /
    (df_main["Venda mensal"].replace(0, np.nan))
).round(0).fillna(0).astype(int)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# LÓGICA DE SUGESTÃO DE COMPRA
# ============================================================================
print("\n🎯 Calculando sugestões (Modelo e Baseline separados)...")

# --- 1. Definir cobertura mínima por classe ---
df_main['cobertura_minima_em_meses'] = np.where(
    df_main['New ABC'].isin(['A', 'B']), 7, 5
)

# --- 2. Calcular BASELINE TRADICIONAL ---
print("\n📊 Calculando Final_order baseline...")
venda_mensal_safe = df_main["Venda mensal"].replace(0, np.nan).fillna(1)

baseline_raw = (
    df_main['cobertura_minima_em_meses'] * venda_mensal_safe
) - df_main["Total Stock"].fillna(0)

# Arredondar baseline para múltiplo de 10
df_main['Final_order baseline'] = np.ceil(baseline_raw / 10) * 10
df_main['Final_order baseline'] = df_main['Final_order baseline'].fillna(0).astype(int)

# Garantir que baseline não seja negativo
df_main['Final_order baseline'] = df_main['Final_order baseline'].clip(lower=0)

# --- 3. Calcular FINAL_ORDER DO MODELO ---
df_main['Final_order'] = df_main['Final_order'].fillna(0).astype(int)

# --- 4. Aplicar FLAG ao MODELO ---
alcance_float = (df_main["Total Stock"].fillna(0)) / venda_mensal_safe
df_main['Flag'] = np.where(
    (alcance_float.notna()) & (alcance_float > df_main['cobertura_minima_em_meses']),
    "Não Comprar",
    "Comprar"
)

# Zerar APENAS o Final_order (modelo) se Flag = "Não Comprar"
df_main['Final_order'] = np.where(
    df_main['Flag'] == "Não Comprar",
    0,
    df_main['Final_order']
).astype(int)

# --- 5. Rastreabilidade ---
df_main['Origem Sugestão'] = np.where(
    df_main['Final_order'] == 0,
    'Flag: Não Comprar',
    'Modelo ML (100%)'
)

# --- 6. Limpar coluna temporária ---
df_main = df_main.drop(columns=['cobertura_minima_em_meses'])

# --- 7. Calcular cobertura ---
df_main['Cobertura'] = (df_main['Total Stock'] / venda_mensal_safe).round(2)
df_main['Cobertura'] = df_main['Cobertura'].fillna(0)

df_main['Nova cobertura'] = np.rint(
    (df_main['Total Stock'] + df_main['Final_order']) / venda_mensal_safe
)
df_main['Nova cobertura'] = df_main['Nova cobertura'].astype(float).fillna(0)

df_main['Alcance - Estoque total + Novo pedido'] = (
    (df_main["Total Stock"].fillna(0) + df_main['Final_order'].fillna(0)) /
    venda_mensal_safe
).round(0).fillna(0).astype(int)

cols_to_fix = ['Final_order', 'Final_order baseline']

for col in cols_to_fix:
    # 1. Garante que NaNs sejam 0 para evitar erros
    # 2. Aplica a fórmula: (Valor / 5) -> Arredonda -> * 5
    # 3. Converte para inteiro
    df_main[col] = df_main[col].fillna(0)
    df_main[col] = (df_main[col] / 5).round() * 5
    df_main[col] = df_main[col].astype(int)

print("\n✅ Sugestões calculadas!")
print(f"   - Final_order: 100% do modelo ML")
print(f"   - Final_order baseline: Para comparação")

print("\n--- Distribuição das Origens das Sugestões ---")
print(df_main['Origem Sugestão'].value_counts())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Aplicar teto de segurança APENAS ao Final_order (modelo)
print("\n🔒 Aplicando teto de segurança ao Final_order (modelo)...")

df_main['Sales 12M'] = pd.to_numeric(df_main['Sales 12M'], errors='coerce').fillna(0)
teto_anual_arredondado = np.floor(df_main['Sales 12M'] / 10) * 10

df_main['Obs'] = df_main['Obs'].astype(str).fillna('')

# Aplicar cap ao Final_order (modelo)
cond_capped = (df_main['Final_order'] > teto_anual_arredondado)
df_main.loc[cond_capped, 'Obs'] = (
    df_main.loc[cond_capped, 'Obs']
    .str.strip()
    .add(' [Cap Final]')
)
df_main['Final_order'] = np.minimum(df_main['Final_order'], teto_anual_arredondado).astype(int)

# Aplicar cap ao baseline também (para comparação justa)
cond_capped_bl = (df_main['Final_order baseline'] > teto_anual_arredondado)
df_main.loc[cond_capped_bl, 'Obs'] = (
    df_main.loc[cond_capped_bl, 'Obs']
    .str.strip()
    .add(' [Cap BL]')
)
df_main['Final_order baseline'] = np.minimum(
    df_main['Final_order baseline'], 
    teto_anual_arredondado
).astype(int)

df_main['Obs'] = df_main['Obs'].str.strip()
df_main = df_main.drop(columns=['Obs'])

print(f"✅ Teto aplicado:")
print(f"   - Final_order limitado: {cond_capped.sum()}")
print(f"   - Baseline limitado: {cond_capped_bl.sum()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Verificar sugestões muito altas ou muito baixas
avg_demand = (df_main['Sales 12M'] + df_main['Unfulfilled 12M'])/12
df_main['Check Suggestion'] = np.where(
    (avg_demand / 12 < 10) | 
    (df_main['Final_order'] > (12 * avg_demand)) | 
    (df_main['Final_order'] < (avg_demand / 10)),
    False, 
    True
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Garantir que não há valores negativos
df_main['Final_order'] = df_main['Final_order'].apply(lambda x: max(0, x))
df_main['Final_order baseline'] = df_main['Final_order baseline'].apply(lambda x: max(0, x))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Verificar faturamento mínimo
default_list = min_order_value_warning(df_main, df_faturamento_minimo)
df_main['Min Order Value Warning'] = df_main['Supp Cod'].isin(default_list)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Comparação Modelo vs Baseline
print("\n📊 Análise: Modelo vs Baseline...")

df_main['Diferença (Modelo - Baseline)'] = df_main['Final_order'] - df_main['Final_order baseline']
df_main['Diferença Abs'] = df_main['Diferença (Modelo - Baseline)'].abs()

# Categorizar concordância
conditions = [
    df_main['Final_order'] == df_main['Final_order baseline'],
    df_main['Final_order'] > df_main['Final_order baseline'],
    df_main['Final_order'] < df_main['Final_order baseline']
]
choices = [
    '[Model == Baseline]',
    '[Model > Baseline]',
    '[Baseline > Model]'
]
df_main['Comparação'] = np.select(conditions, choices, default='')

# Alertas visuais de diferença
alert_conditions = [
    df_main['Diferença Abs'] > 900,
    df_main['Diferença Abs'] > 400
]
alert_icons = ['🔴', '🟡']
df_main['Alerta (Diferença: Modelo - Baseline)'] = np.select(alert_conditions, alert_icons, default='🟢')

print("\n--- Distribuição de Concordância ---")
print(df_main['Comparação'].value_counts())

print("\n--- Alertas de Diferença ---")
print(df_main['Alerta (Diferença: Modelo - Baseline)'].value_counts())

print(f"\n📈 Estatísticas da Diferença:")
print(f"   - Média: {df_main['Diferença (Modelo - Baseline)'].mean():.2f}")
print(f"   - Mediana: {df_main['Diferença (Modelo - Baseline)'].median():.2f}")
print(f"   - Diferença absoluta média: {df_main['Diferença Abs'].mean():.2f}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calcular alerta baseado na Model Suggestion vs demanda média
df_main['Model vs Demand Ratio'] = (
    df_main['Final_order'] / df_main['Venda mensal']
).replace([np.inf, -np.inf], np.nan).round(2)

conditions = [
    df_main['Model vs Demand Ratio'] > 10,  # 🔴 Sugestão muito alta
    df_main['Model vs Demand Ratio'] > 5   # 🟡 Sugestão alta
]
icons = ['🔴', '🟡']
df_main['Alerta (Model vs Demand Ratio)'] = np.select(conditions, icons, default='🟢')
df_main['Model vs Demand Ratio'] = df_main['Model vs Demand Ratio'].fillna(0)
#df_main[['Component', 'Final_order baseline', 'Final_order', 'Alert Diferença']].sort_values(['Component', 'Final_order baseline'], ascending=[True, False])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_vendas_periodo = get_sales_last_months(df_vendas_raw)
cols_sales = [col for col in df_vendas_periodo.columns if col.startswith('Sales-M')]
df_vendas_periodo[cols_sales] = df_vendas_periodo[cols_sales].fillna(0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calcular estatísticas de vendas mensais (mantido igual)
df_main = pd.merge(df_main, df_vendas_periodo, on='Component', how='left')
sales_cols = df_main.filter(like="Sales-M")

df_main["Média"] = sales_cols.mean(axis=1, skipna=True).round(2)
df_main["DP"] = sales_cols.std(axis=1, ddof=1, skipna=True).round(2)

cv = df_main["DP"] / df_main["Média"]
cv = cv.replace([np.inf, -np.inf], np.nan)
df_main["CV"] = cv.round(2)

cond_cv = [
    df_main["CV"].le(0.2),
    df_main["CV"].le(0.5)
]
choice_cv = ["BAIXO", "MÉDIO"]
df_main["CV Flag"] = np.select(cond_cv, choice_cv, default="ALTO")

cond_ns = [
    df_main["CV"].le(0.2),
    df_main["CV"].le(0.5)
]
choice_ns = [0.975, 0.95]
df_main["Nivel de Servico"] = np.select(cond_ns, choice_ns, default=0.90).astype(float).round(2)

valid_ns = df_main["Nivel de Servico"].between(1e-12, 1 - 1e-12)
df_main["Valor crítico da normal (Z)"] = np.where(
    valid_ns, norm.ppf(df_main["Nivel de Servico"]), np.nan
).round(2)

z_round = df_main["Valor crítico da normal (Z)"].round(2)
cond_z = [
    z_round.lt(1.28),
    z_round.le(1.28),
    z_round.le(1.64),
    z_round.le(1.96),
    z_round.le(2.33)
]
choice_z = [
    "abaixo de 90%",
    "90% de chance de não faltar",
    "95% de chance de não faltar",
    "97,5% de chance de não faltar",
    "99% de chance de não faltar"
]
df_main["Z Flag"] = np.select(cond_z, choice_z, default="Nível de serviço fora da faixa")

lt_rt_term = (df_main["LT"] + df_main["RP"]) / 30.0
lt_rt_term = lt_rt_term.clip(lower=0)
df_main["Sigma no período"] = (df_main["DP"] * np.sqrt(lt_rt_term)).replace([np.inf, -np.inf], np.nan).round(2)

df_main["Demanda Média no Periodo"] = ((df_main["Média"] / 30.0) * (df_main["LT"] + df_main["RP"])).round(2)

df_main["Estoque de Segurança"] = (
    df_main["Valor crítico da normal (Z)"] * df_main["DP"] * np.sqrt(lt_rt_term)
).round(2)

df_main["ROP"] = (df_main["Sigma no período"] + df_main["Demanda Média no Periodo"]).round(2)

df_main["Order Up To"] = (df_main["ROP"] + (df_main["Média"] / 30.0) * df_main["RP"]).round(2)

df_main["Service Level"] = np.select(
    [
        df_main["CV Flag"].eq("BAIXO"),
        df_main["CV Flag"].eq("MÉDIO")
    ],
    ["DEMANDA PREVISÍVEL", "MÉDIA VOLATILIDADE"],
    default="ALTA VOLATILIDADE"
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Funções auxiliares para conversão de tipos
def _safe_to_numeric(df, cols, as_int=False, round_ndec=None):
    """Converte colunas para numérico"""
    present = [c for c in cols if c in df.columns]
    if not present:
        return
    df[present] = df[present].apply(pd.to_numeric, errors="coerce")
    if round_ndec is not None:
        df[present] = df[present].round(round_ndec)
    if as_int:
        df[present] = df[present].astype("Int64")

def _clip_negatives(df, cols):
    """Zera valores negativos"""
    for c in cols:
        if c in df.columns:
            df.loc[df[c] < 0, c] = 0

# Colunas de string
string_cols = [
    "Component", "Cod X", "Description", "Group code", "Group name",
    "Supplier", "ABC", "Obs", "Comparação"
]
present_str = [c for c in string_cols if c in df_main.columns]
if present_str:
    df_main[present_str] = df_main[present_str].fillna("").astype(str)
    df_main[present_str] = df_main[present_str].apply(lambda s: s.str.strip())

if "Group code" in df_main.columns:
    df_main["Group code"] = df_main["Group code"].str.replace("'", "", regex=False)

# Colunas inteiras (COM Final_order baseline)
int_cols = [
    "Supp Cod", "Stock", "Transit", "Inspection", "Reserved", "Total Stock",
    "Sales 12M", "Unfulfilled 12M", "KanBan Min", "KanBan Max",
    "Final_order", "Final_order baseline",  # ← AMBOS mantidos
    "Alert", "Safety stock", "RP", "LT", "LT+RP", "Inventory level",
    "Demand (LT+RP)", "Venda mensal", "Diferença (Modelo - Baseline)", "Diferença Abs"
]
_safe_to_numeric(df_main, int_cols, as_int=True)

if "Supp Cod" in df_main.columns:
    df_main["Supp Cod"] = df_main["Supp Cod"].astype(str)

# Colunas float
float_cols = ["% Export 12M", "Cost", "Total Cost", "Model Suggestion", "Model vs Demand Ratio"]
_safe_to_numeric(df_main, float_cols, as_int=False, round_ndec=2)

# Colunas dinâmicas inteiras
transit_cols = [c for c in df_main.columns if c.startswith("Transit ")]
sales_m_cols = [c for c in df_main.columns if c.startswith("Sales-M")]
dyn_int_cols = transit_cols + sales_m_cols
_safe_to_numeric(df_main, dyn_int_cols, as_int=True)

# Recalcular Total Cost (baseado no Final_order do modelo)
if {"Cost", "Final_order"}.issubset(df_main.columns):
    cost_num = pd.to_numeric(df_main["Cost"], errors="coerce")
    fo_num = pd.to_numeric(df_main["Final_order"], errors="coerce")
    df_main["Total Cost"] = (cost_num * fo_num).round(2)
else:
    if "Total Cost" not in df_main.columns:
        df_main["Total Cost"] = np.nan

# Zerar negativos
_clip_negatives(df_main, [
    "Stock", "Transit", "Inspection", "Reserved", "Total Stock", 
    "Total Cost", "Sales 12M", "Unfulfilled 12M"
])

coluna_para_mover = df_main.pop('Final_order baseline')
posicao_nova = df_main.columns.get_loc('Final_order') + 1
df_main.insert(posicao_nova, 'Final_order baseline', coluna_para_mover)

print("✅ Tipos de dados atualizados")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ============================================================================
# CÁLCULO DE RUPTURA PROJETADA (SIMULAÇÃO DIA A DIA)
# ============================================================================
print("\n🔮 Iniciando simulação de estoque dia-a-dia...")
transit_dict = dict(tuple(df_pending_orders[df_pending_orders['MTE'].isin(df_main['Component'])].groupby('MTE')))

# Definir função wrapper para aplicar no DataFrame
def simulate_stock_row(row):
    component = row['Component']
    
    # Se não tiver LT ou RP, assume 0 para evitar erro de divisão
    lt = int(row['LT']) if pd.notna(row['LT']) else 0
    rp = int(row['RP']) if pd.notna(row['RP']) else 0
    
    # Demanda total no período de cobertura (LT+RP)
    # Se a coluna 'Demand (LT+RP)' não estiver confiável, calculamos na hora usando Venda Mensal
    if 'Venda mensal' in row and row['Venda mensal'] > 0:
        daily_demand = row['Venda mensal'] / 30
        demand_period_total = daily_demand * (lt + rp)
    else:
        demand_period_total = 0

    if demand_period_total == 0:
        return 0.0 # Sem demanda, sem perda

    current_stock = int(row['Total Stock']) if pd.notna(row['Total Stock']) else 0
    final_order = int(row['Final_order']) if pd.notna(row['Final_order']) else 0
    
    # Busca trânsito específico deste componente
    df_transit_comp = transit_dict.get(component, pd.DataFrame(columns=["DT. Entrega PO", "Qtd"]))
    
    # Data de início (Hoje ou a data base do relatório)
    # Assumindo 'today' como data de uso, ou use a variável 'usage_date' definida anteriormente no script
    sim_date = datetime.datetime.now() 
    
    try:
        # Chama a função original calculate_projected_level
        # Nota: A função retorna (DataFrame Histórico, Total Lost)
        _, total_lost = calculate_projected_level(
            usage_date=sim_date,
            lt=lt,
            rp=rp,
            current_level=current_stock,
            Final_order=final_order,
            demand=demand_period_total, 
            df_intransit=df_transit_comp
        )
        return total_lost
    except Exception:
        return 0.0

# 3. Aplicar ao df_main
# Aviso: Isso pode levar alguns segundos/minutos dependendo do tamanho do df_main
print("Executando simulação para cada componente...")
df_main['Venda Perdida Projetada (Unid)'] = df_main.apply(simulate_stock_row, axis=1)

# Calcular valor financeiro da perda
if 'Cost' in df_main.columns:
    df_main['Venda Perdida Projetada ($)'] = (df_main['Venda Perdida Projetada (Unid)'] * df_main['Cost']).round(2)

print("✅ Simulação concluída.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 4. Criar alerta de Ruptura
df_main['Alerta Ruptura'] = np.where(
    df_main['Venda Perdida Projetada (Unid)'] > 0,
    '🔴 Risco de Ruptura',
    '🟢 Estoque Saudável'
)

print(df_main['Alerta Ruptura'].value_counts())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Estatísticas finais
print("\n📊 Estatísticas Finais:")
print(f"\n🤖 MODELO (Final_order):")
print(f"   - Total de componentes: {len(df_main)}")
print(f"   - Componentes com sugestão > 0: {(df_main['Final_order'] > 0).sum()}")
print(f"   - Média: {df_main['Final_order'].mean():.2f}")
print(f"   - Valor total: ${df_main['Total Cost'].sum():,.2f}")

print(f"\n📐 BASELINE (Final_order baseline):")
print(f"   - Componentes com sugestão > 0: {(df_main['Final_order baseline'] > 0).sum()}")
print(f"   - Média: {df_main['Final_order baseline'].mean():.2f}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Salvar resultado final
print("\n💾 Salvando df_main...")
Helpers.save_output_dataset(context=context, output_name='df_main', data_frame=df_main)
print("✅ df_main salvo com sucesso!")

print(f"\n✨ Processamento concluído!")
print(f"   📦 DataFrame final: {df_main.shape[0]} linhas × {df_main.shape[1]} colunas")