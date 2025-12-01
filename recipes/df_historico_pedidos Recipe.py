# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from utils.notebookhelpers.helpers import Helpers
from utils.dtos.templateOutputCollection import TemplateOutputCollection
from utils.dtos.templateOutput import TemplateOutput
from utils.dtos.templateOutput import OutputType
from utils.dtos.templateOutput import ChartType
from utils.dtos.variable import Metadata
from utils.rcclient.commons.variable_datatype import VariableDatatype
from utils.dtos.templateOutput import FileType
from utils.dtos.rc_ml_model import RCMLModel
from utils.libutils.vectorStores.utils import VectorStoreUtils

context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import logging
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Dict, Tuple, Union, Optional, Callable
from functools import wraps
import holidays
import calendar
import regex as re
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer, mean_absolute_error, mean_squared_error, 
    r2_score, mean_absolute_percentage_error, median_absolute_error
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# FUNÇÕES AUXILIARES (MÉTRICAS E UTILIDADES)
# -------------------------------------------------------------------------------- 
def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    ape = np.abs(y_true - y_pred) / y_true
    
    if np.isscalar(ape):
        if np.isfinite(ape):
            return ape
        else:
            return 1
    else:
        ape[~np.isfinite(ape)] = 1
    return np.mean(ape)

def get_metrics(df_actual, df_prediction, index=None):
    """Calcula métricas de performance"""
    if index is not None:
        df_actual = df_actual.loc[index]
        df_prediction = df_prediction.loc[index]
    
    metrics_dict = {
        "MAE": mean_absolute_error(df_actual, df_prediction),
        "MSE": mean_squared_error(df_actual, df_prediction),
        "RMSE": np.sqrt(mean_squared_error(df_actual, df_prediction)),
        "R2": r2_score(df_actual, df_prediction),
        "Total Error": np.sum(df_actual - df_prediction),
        "Percentage error": np.sum(df_actual-df_prediction)/np.sum(df_actual),
        "MAPE": mean_absolute_percentage_error(df_actual, df_prediction),
        "Median Absolute Error": median_absolute_error(df_actual, df_prediction),
    }
    return metrics_dict

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# 1. CARREGAMENTO E TRATAMENTO INICIAL
# -------------------------------------------------------------------------------- 
df_historico_pedidos = Helpers.getEntityData(context, 'df_historico_pedidos').rename(columns={
    'PRODUTO': 'COMPONENT'
})

# Obtendo valor unitario
df_historico_pedidos['PRECO_UNIT'] = df_historico_pedidos['PRECO_TOTAL']/df_historico_pedidos['QUANTIDADE']

# Tratamento de Lead Time
df_historico_pedidos['LEAD_TIME'] = df_historico_pedidos['LEAD_TIME_DIAS'] / 30

# Preenchimento de Lead Times vazios (Hierarquia: Comp+Forn -> Forn -> Global -> 3)
media_comp_fornec = np.ceil(df_historico_pedidos.groupby(['COMPONENT', 'COD_FORNE'])['LEAD_TIME'].transform('mean'))
df_historico_pedidos['LEAD_TIME'] = df_historico_pedidos['LEAD_TIME'].fillna(media_comp_fornec)

media_fornec = np.ceil(df_historico_pedidos.groupby('COD_FORNE')['LEAD_TIME'].transform('mean'))
df_historico_pedidos['LEAD_TIME'] = df_historico_pedidos['LEAD_TIME'].fillna(media_fornec)

media_global = np.ceil(df_historico_pedidos['LEAD_TIME'].mean())
df_historico_pedidos['LEAD_TIME'] = df_historico_pedidos['LEAD_TIME'].fillna(media_global)

# Limpeza e Conversão
df_historico_pedidos['LEAD_TIME'] = df_historico_pedidos['LEAD_TIME'].replace([np.inf, -np.inf], 1)
df_historico_pedidos['LEAD_TIME'] = np.ceil(df_historico_pedidos['LEAD_TIME']).astype(int)

# Caso o lead time seja menor que 1 mes (algo raro, optamos por usar o lead time médio daquele fornecedor, mas se esse valor for NaN, então
# usamos um lead time de 3 meses
df_historico_pedidos.loc[
    df_historico_pedidos['LEAD_TIME'] == 1,
    'LEAD_TIME'
] = media_fornec.fillna(3)

# Tornando as colunas datas
df_historico_pedidos['DATA_SI'] = pd.to_datetime(df_historico_pedidos['DATA_SI'])
df_historico_pedidos['MES'] = df_historico_pedidos['DATA_SI'].dt.to_period('M').dt.to_timestamp()

# Mantendo apenas uma observacao por "PEDIDO", "COD_FORNE", "COMPONENT", "DATA_SI"
df_historico_pedidos = (
    df_historico_pedidos
    .sort_values("PRECO_TOTAL", ascending=False)
    .drop_duplicates(subset=["MES", "COD_FORNE", "COMPONENT", "DATA_SI"], keep="first")
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# 1.2. PADRONIZAÇÃO MENSAL DO HISTÓRICO
# -------------------------------------------------------------------------------- 
print("📅 Agregando dados mensalmente e preenchendo lacunas...")

# Agregar mensalmente
df_monthly = df_historico_pedidos.groupby(['COMPONENT', 'COD_FORNE', 'MES']).agg({
    'QUANTIDADE': 'sum',
    'LEAD_TIME': 'max' 
}).reset_index()

# Garantir continuidade temporal (Cross Join)
unique_pairs = df_monthly[['COMPONENT', 'COD_FORNE']].drop_duplicates()
all_months = pd.date_range(df_monthly['MES'].min(), df_monthly['MES'].max(), freq='MS')
df_dates = pd.DataFrame({'MES': all_months})

unique_pairs['key'] = 1
df_dates['key'] = 1
df_full_idx = pd.merge(unique_pairs, df_dates, on='key').drop('key', axis=1)

# Merge final
df_final = pd.merge(df_full_idx, df_monthly, on=['COMPONENT', 'COD_FORNE', 'MES'], how='left')

# Preencher Nulos
df_final['QUANTIDADE'] = df_final['QUANTIDADE'].fillna(0)
df_final['LEAD_TIME'] = df_final.groupby(['COMPONENT', 'COD_FORNE'])['LEAD_TIME'].ffill().bfill().fillna(1).astype(int)

df_final = df_final.sort_values(['COMPONENT', 'COD_FORNE', 'MES'])

print(f"✅ Base mensal padronizada: {df_final.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_vendas = Helpers.getEntityData(context, 'df_vendas')
df_vendas['QUANTIDADE'] = df_vendas['QTD_VENDA_NACIONAL'] + df_vendas['QTD_VENDA_EXPORTACAO']
df_vendas['MES'] = df_vendas['DATA'].dt.to_period('M').dt.to_timestamp()
df_vendas = df_vendas[['B1_COD_PP', 'MES', 'QUANTIDADE']].rename(columns = {'B1_COD_PP': 'COMPONENT'})
df_vendas = df_vendas.groupby(['COMPONENT', 'MES']).agg({
    'QUANTIDADE': 'sum' 
}).reset_index().sort_values(['COMPONENT',  'MES'])
df_vendas

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# 3. CRIAÇÃO DO TARGET HÍBRIDO (INTERNO + EXTERNO/VENDAS)
# ==================================================================================
print("\n🎯 Criando TARGET (Prioridade: Real Futuro Interno > Real Futuro Vendas)...")

df_final['TARGET'] = np.nan 
lead_times_unicos = df_final['LEAD_TIME'].unique()

# A) FASE 1: Tentar preencher com dados REAIS da própria base (Shift Negativo no df_final)
print("\n   Fase 1: Preenchendo com dados reais futuros (interno)...")
for lt in lead_times_unicos:
    if pd.isna(lt) or lt <= 0: 
        continue
    lt_meses = int(lt)
    
    # Busca venda real no futuro do dataset atual
    shift_futuro = df_final.groupby(['COMPONENT', 'COD_FORNE'])['QUANTIDADE'].shift(-lt_meses)
    
    mask = (df_final['LEAD_TIME'] == lt)
    df_final.loc[mask, 'TARGET'] = shift_futuro[mask]

n_nans_antes = df_final['TARGET'].isna().sum()
print(f"   TARGETs vazios após dados internos: {n_nans_antes}")

# B) FASE 2: Preencher NaNs restantes usando o df_vendas
if n_nans_antes > 0:
    print("\n   Fase 2: Preenchendo com dados do df_vendas...")
    
    # ✅ PREPARAR DF_VENDAS
    df_vendas_clean = df_vendas.copy()
    # Apenas garante que a quantidade no df_vendas não seja NaN, mas não altera o Target ainda
    df_vendas_clean['QUANTIDADE'] = df_vendas_clean['QUANTIDADE'].fillna(0)
    
    # Calcular DATA_ALVO_NECESSARIA
    df_final['DATA_ALVO_NECESSARIA'] = [
        m + pd.DateOffset(months=int(lt)) 
        for m, lt in zip(df_final['MES'], df_final['LEAD_TIME'])
    ]
    
    print(f"      Período necessário: {df_final['DATA_ALVO_NECESSARIA'].min()} até {df_final['DATA_ALVO_NECESSARIA'].max()}")
    print(f"      Período disponível (df_vendas): {df_vendas_clean['MES'].min()} até {df_vendas_clean['MES'].max()}")
    
    # Merge com df_vendas
    df_merged = pd.merge(
        df_final,
        df_vendas_clean[['COMPONENT', 'MES', 'QUANTIDADE']], 
        left_on=['COMPONENT', 'DATA_ALVO_NECESSARIA'],      
        right_on=['COMPONENT', 'MES'],                      
        how='left',
        suffixes=('', '_vendas')
    )
    
    # Preencher TARGET onde ainda está vazio
    mask_nan = df_final['TARGET'].isna()
    
    # Identificar nome correto da coluna após o merge
    if 'QUANTIDADE_vendas' in df_merged.columns:
        coluna_alvo = 'QUANTIDADE_vendas'
    else:
        coluna_alvo = 'QUANTIDADE' 

    # Atribui o valor encontrado. Se não achou match, df_merged[...] será NaN, mantendo o Target como NaN.
    df_final.loc[mask_nan, 'TARGET'] = df_merged.loc[mask_nan, coluna_alvo]
    
    # Limpar colunas auxiliares
    df_final = df_final.drop(columns=['DATA_ALVO_NECESSARIA'])
    
    n_nans_depois = df_final['TARGET'].isna().sum()
    print(f"   TARGETs vazios após df_vendas: {n_nans_depois}")
    print(f"   ✅ Recuperamos {n_nans_antes - n_nans_depois} linhas!")
    
    # ✅ DIAGNÓSTICO (Sem preencher com 0)
    if n_nans_depois > 0:
        print(f"\n   ⚠️ DIAGNÓSTICO DOS {n_nans_depois} NANs RESTANTES (Serão mantidos como NaN):")
        
        df_nans = df_final[df_final['TARGET'].isna()].copy()
        df_nans['DATA_ALVO'] = [
            m + pd.DateOffset(months=int(lt)) 
            for m, lt in zip(df_nans['MES'], df_nans['LEAD_TIME'])
        ]
        
        # Verificar se são datas fora do range do df_vendas
        min_vendas = df_vendas_clean['MES'].min()
        max_vendas = df_vendas_clean['MES'].max()
        
        fora_range = (df_nans['DATA_ALVO'] < min_vendas) | (df_nans['DATA_ALVO'] > max_vendas)
        n_fora = fora_range.sum()
        
        print(f"      Datas ALVO fora do range do df_vendas: {n_fora}")
        
        if n_fora > 0:
            print(f"      Exemplo de datas fora do range:")
            print(df_nans[fora_range][['COMPONENT', 'MES', 'LEAD_TIME', 'DATA_ALVO']].head(5))
        
        # Verificar componentes sem dados no df_vendas
        components_sem_vendas = set(df_nans['COMPONENT']) - set(df_vendas_clean['COMPONENT'])
        if components_sem_vendas:
            print(f"\n      COMPONENTs ausentes no df_vendas: {len(components_sem_vendas)}")
            print(f"      Exemplos: {list(components_sem_vendas)[:5]}")

        # 🛑 AQUI ESTAVA O FILTRO DE ZEROS QUE FOI REMOVIDO 🛑
        print(f"\n      ⚠️ NENHUM preenchimento com 0 foi realizado. Mantendo NaNs originais.")

else:
    print("   ✅ Todos os TARGETs preenchidos com dados internos!")

print(f"\n✅ TARGET criado com sucesso!")
print(f"   Total de linhas: {len(df_final):,}")
print(f"   TARGETs válidos: {df_final['TARGET'].notna().sum():,}")
print(f"   TARGETs NaN (Mantidos): {df_final['TARGET'].isna().sum()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# CLASSIFICAÇÃO ABC (PARETO DE VOLUME DE COMPRAS)
# -------------------------------------------------------------------------------- 
print("\n" + "="*80)
print("📊 CLASSIFICAÇÃO ABC (ANÁLISE DE PARETO)")
print("="*80)

# Agregar volume total de compras por COMPONENT (últimos 12 meses para classificação)
last_12_months = df_final['MES'].max() - pd.DateOffset(months=12)
df_abc = df_final[df_final['MES'] >= last_12_months].groupby('COMPONENT')['QUANTIDADE'].sum().reset_index()
df_abc = df_abc.sort_values('QUANTIDADE', ascending=False).reset_index(drop=True)

# Calcular Pareto
df_abc['cumsum'] = df_abc['QUANTIDADE'].cumsum()
total_volume = df_abc['QUANTIDADE'].sum()
df_abc['cumsum_pct'] = (df_abc['cumsum'] / total_volume) * 100

def get_abc_class(cumsum_pct):
    if cumsum_pct <= 80: return 'A'
    elif cumsum_pct <= 95: return 'B'
    else: return 'C'

df_abc['CLASSE_ABC'] = df_abc['cumsum_pct'].apply(get_abc_class)

# Criar mapa COMPONENT -> Classe ABC
product_class_map = df_abc.set_index('COMPONENT')['CLASSE_ABC'].to_dict()

# Aplicar ao dataset
df_final['CLASSE_ABC'] = df_final['COMPONENT'].map(product_class_map).fillna('C')

print(f"\n   📊 Distribuição de COMPONENTs por Classe ABC:")
for classe in ['A', 'B', 'C']:
    n_components = (df_abc['CLASSE_ABC'] == classe).sum()
    volume = df_abc[df_abc['CLASSE_ABC'] == classe]['QUANTIDADE'].sum()
    pct_volume = (volume / total_volume) * 100
    print(f"      Classe {classe}: {n_components:,} COMPONENTs ({pct_volume:.1f}% do volume)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# 4. CRIAÇÃO DAS FEATURES DE HISTÓRICO (LAGS)
# ==================================================================================
print("\n⏪ Criando colunas de histórico anterior (Lags 1 a 12)...")

cols_lags = []
for i in range(1, 13):
    col_name = f'HISTORICO_VENDAS_LAG{i}'
    cols_lags.append(col_name)
    # Shift Positivo (i) pega o valor passado
    df_final[col_name] = df_final.groupby(['COMPONENT', 'COD_FORNE'])['QUANTIDADE'].shift(i)

print(f"   ✅ {len(cols_lags)} LAGs criados")
print(f"   Exemplo de colunas: {cols_lags[:3]}")

# Verificar
print(f"\n   🔍 Verificação:")
print(f"      LAG1 tem {df_final['HISTORICO_VENDAS_LAG1'].notna().sum():,} valores válidos")
print(f"      LAG12 tem {df_final['HISTORICO_VENDAS_LAG12'].notna().sum():,} valores válidos")

# ==================================================================================
# 5. CRIAÇÃO DAS FEATURES DE SAZONALIDADE
# ==================================================================================
df_final['MES_NUM'] = df_final['MES'].dt.month
df_final['TRIMESTRE'] = df_final['MES'].dt.quarter
df_final['ANO'] = df_final['MES'].dt.year

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# CRIAÇÃO DE FEATURES ADICIONAIS
# -------------------------------------------------------------------------------- 
print("\n" + "="*80)
print("🛠️ CRIAÇÃO DE FEATURES ADICIONAIS")
print("="*80)

df_final = df_final.sort_values(['COMPONENT', 'COD_FORNE', 'MES'])

# 1. Diferenças (usando os lags que já existem)
print("   Criando diferenças...")
df_final['diff_1'] = df_final['QUANTIDADE'] - df_final['HISTORICO_VENDAS_LAG1']
df_final['diff_2'] = df_final['HISTORICO_VENDAS_LAG1'] - df_final['HISTORICO_VENDAS_LAG2']
df_final['diff_3'] = df_final['HISTORICO_VENDAS_LAG2'] - df_final['HISTORICO_VENDAS_LAG3']

# 2. Rolling Statistics (usando os lags existentes)
print("   Criando rolling statistics...")

# Rolling Mean (janelas de 3, 6, 12)
df_final['rolling_mean_3'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].mean(axis=1)
df_final['rolling_mean_6'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                        'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].mean(axis=1)
df_final['rolling_mean_12'] = df_final[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].mean(axis=1)

# Rolling Std (janelas de 3, 6, 12)
df_final['rolling_std_3'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].std(axis=1)
df_final['rolling_std_6'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                       'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].std(axis=1)
df_final['rolling_std_12'] = df_final[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].std(axis=1)

# Rolling Max (janelas de 3, 6, 12)
df_final['rolling_max_3'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].max(axis=1)
df_final['rolling_max_6'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                       'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].max(axis=1)
df_final['rolling_max_12'] = df_final[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].max(axis=1)

# Rolling Min (janelas de 3, 6, 12)
df_final['rolling_min_3'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].min(axis=1)
df_final['rolling_min_6'] = df_final[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                       'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].min(axis=1)
df_final['rolling_min_12'] = df_final[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].min(axis=1)

# 3. Lag do lead time
print("   Criando lag de lead time...")
df_final['lead_time_lag_1'] = df_final.groupby(['COMPONENT', 'COD_FORNE'])['LEAD_TIME'].shift(1)

# 4. Features temporais
print("   Criando features temporais...")
df_final['month'] = df_final['MES'].dt.month
df_final['quarter'] = df_final['MES'].dt.quarter
df_final['quarter_start'] = df_final['MES'].dt.is_quarter_start.astype(int)
df_final['quarter_end'] = df_final['MES'].dt.is_quarter_end.astype(int)

# One-Hot para mês
df_final = pd.get_dummies(df_final, columns=['month'], prefix='month')

# 5. Limpar valores infinitos e NaNs
print("   Limpando valores infinitos e NaNs...")
numeric_cols = df_final.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df_final[col] = df_final[col].replace([np.inf, -np.inf], np.nan)
    
# Preencher NaNs nas rolling statistics (quando não há dados suficientes)
rolling_cols = [col for col in df_final.columns if 'rolling_' in col]
for col in rolling_cols:
    df_final[col] = df_final[col].fillna(0)

max_date = df_final['MES'].max()
df_final = df_final[df_final['MES'] < max_date]
print(f"   ✅ Features criadas: {len(df_final.columns)} colunas totais")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# PREPARAÇÃO PARA MODELAGEM
# -------------------------------------------------------------------------------- 
print("\n" + "="*80)
print("📐 PREPARAÇÃO PARA MODELAGEM")
print("="*80)


df_model_data = df_final.copy()

print(f"   Dataset para modelagem: {df_model_data.shape}")

# Definir features (excluindo colunas de identificação e TARGET)
ignore_cols = [
    'COMPONENT', 'COD_FORNE', 'MES', 'TARGET', 'DATA_SI', 'CLASSE_ABC', 'QUANTIDADE'
]
ignore_cols = [c for c in ignore_cols if c in df_model_data.columns]

# CRÍTICO: Filtrar apenas colunas numéricas (excluir datetime)
all_possible_features = [c for c in df_model_data.columns if c not in ignore_cols]

# Verificar tipo de cada coluna
features = []
excluded_features = []

for col in all_possible_features:
    dtype = df_model_data[col].dtype
    # Aceitar apenas tipos numéricos (int, float, bool)
    if dtype in ['int64', 'float64', 'int32', 'float32', 'bool', 'uint8', 'int8']:
        features.append(col)
    else:
        excluded_features.append(col)

print(f"\n   ✅ Features numéricas selecionadas: {len(features)}")

if excluded_features:
    print(f"   ⚠️ Features excluídas (não-numéricas): {len(excluded_features)}")
    print(f"      Exemplos: {excluded_features[:5]}")

# Contar features por tipo
historico_features_count = len([f for f in features if 'HISTORICO_VENDAS_LAG' in f])
outras_features_count = len(features) - historico_features_count

print(f"\n   📊 Breakdown de Features:")
print(f"      📈 Histórico (LAGS): {historico_features_count}")
print(f"      🔧 Outras: {outras_features_count}")
print(f"      📊 TOTAL: {len(features)}")

print(f"\n   🔍 Verificação de LAGs:")
lag_cols = [c for c in features if 'HISTORICO_VENDAS_LAG' in c]
for lag_col in lag_cols[:3]:  # Primeiros 3
    n_nans = df_model_data[lag_col].isna().sum()
    print(f"      {lag_col}: {n_nans} NaNs ({n_nans/len(df_model_data)*100:.1f}%)")

# Split temporal (80/20)
months = sorted(df_model_data['MES'].unique())
split_idx = int(len(months) * 0.8)
train_months = months[:split_idx]
test_months = months[split_idx:]

df_train = df_model_data[df_model_data['MES'].isin(train_months)].copy()
df_test = df_model_data[df_model_data['MES'].isin(test_months)].copy()

print(f"\n   📅 Split Temporal:")
print(f"      Treino: {len(train_months)} meses ({train_months[0]} até {train_months[-1]})")
print(f"      Teste:  {len(test_months)} meses ({test_months[0]} até {test_months[-1]})")
print(f"      Registros Treino: {len(df_train):,}")
print(f"      Registros Teste:  {len(df_test):,}")

# Após o split:
print(f"\n   📊 Distribuição temporal:")
print(f"      Treino - Meses: {len(train_months)}")
for mes in train_months[:3]:
    n = len(df_train[df_train['MES'] == mes])
    print(f"         {pd.to_datetime(mes).strftime('%Y-%m')}: {n:,} linhas")
print(f"      ...")
print(f"      Teste - Meses: {len(test_months)}")
for mes in test_months[:3]:
    n = len(df_test[df_test['MES'] == mes])
    print(f"         {pd.to_datetime(mes).strftime('%Y-%m')}: {n:,} linhas")

# Verificar se há dados suficientes
print(f"\n   🔍 Verificação de Dados:")
print(f"      TARGET com NaN (treino): {df_train['TARGET'].isna().sum()}")
print(f"      TARGET com NaN (teste): {df_test['TARGET'].isna().sum()}")

# Preencher NaNs remanescentes nas features
print(f"\n   🧹 Preenchendo NaNs nas features...")
df_train[features] = df_train[features].fillna(0)
df_test[features] = df_test[features].fillna(0)

# DIAGNÓSTICO: Verificar tipos antes do treinamento
print(f"\n   🔬 Diagnóstico de Tipos:")
non_numeric = []
for col in features:
    if df_train[col].dtype not in ['int64', 'float64', 'int32', 'float32', 'bool', 'uint8', 'int8']:
        non_numeric.append((col, df_train[col].dtype))

if non_numeric:
    print(f"      ⚠️ ATENÇÃO: {len(non_numeric)} features com tipo não-numérico:")
    for col, dtype in non_numeric[:10]:
        print(f"         - {col}: {dtype}")
    
    # Remover essas features
    features = [f for f in features if f not in [c for c, _ in non_numeric]]
    print(f"      ✅ Features removidas. Total restante: {len(features)}")
else:
    print(f"      ✅ Todas as {len(features)} features são numéricas")

print(f"\n   ✅ Dados preparados para treinamento!")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# FUNÇÃO DE TREINO ABC-XGBOOST
# -------------------------------------------------------------------------------- 
def train_abc_models(df_train, df_test, features, product_class_map, user_param_grid=None, n_iter_search=30):
    """
    Treina um modelo XGBoost separado para cada classe ABC.
    """
    
    print("\n" + "="*80)
    print("🚀 TREINAMENTO DE MODELOS XGBOOST SEPARADOS POR CLASSE ABC")
    print("="*80)
    
    if user_param_grid is None:
        user_param_grid = {}

    # Classificar treino e teste
    classes_train = df_train['COMPONENT'].map(product_class_map).fillna('C').values
    classes_test = df_test['COMPONENT'].map(product_class_map).fillna('C').values
    
    n_A = (classes_train == 'A').sum()
    n_B = (classes_train == 'B').sum()
    n_C = (classes_train == 'C').sum()
    
    print(f"\n📊 Distribuição de LINHAS de treino:")
    print(f"    Classe A: {n_A:,} linhas ({n_A/len(classes_train)*100:.1f}%)")
    print(f"    Classe B: {n_B:,} linhas ({n_B/len(classes_train)*100:.1f}%)")
    print(f"    Classe C: {n_C:,} linhas ({n_C/len(classes_train)*100:.1f}%)")
    print()

    # Preparar dados
    X_train = df_train[features].fillna(0).astype('float64')
    y_train = df_train["TARGET"].fillna(0).astype('float64')
    X_test = df_test[features].fillna(0).astype('float64')
    y_test = df_test["TARGET"].fillna(0).astype('float64')

    abc_models = {
        'models': {}, 
        'feature_names': features, 
        'product_class_map': product_class_map
    }
    
    y_pred_train = np.zeros(len(y_train))
    y_pred_test = np.zeros(len(y_test))

    # Parâmetros default
    default_xgb_params = {
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.03,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 5,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'gamma': 0,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'n_jobs': -1
    }

    # Treinar modelo para cada classe
    for classe in ['A', 'B', 'C']:
        print("-" * 60)
        print(f"🧠 Treinando Modelo para CLASSE {classe}")
        print("-" * 60)
        
        mask_train = (classes_train == classe)
        X_train_classe = X_train[mask_train]
        y_train_classe = y_train[mask_train]
        
        if len(X_train_classe) == 0:
            print(f"    ⚠️ Sem dados de treino para Classe {classe}. Pulando.")
            abc_models['models'][classe] = None
            continue
            
        model_classe = None
        
        # GridSearch ou treino padrão
        if classe in user_param_grid:
            print(f"    🔍 Executando RandomizedSearchCV para Classe {classe}...")
            param_grid = user_param_grid[classe]
            
            base_model = xgb.XGBRegressor(n_jobs=-1, random_state=42)

            tscv = TimeSeriesSplit(n_splits=3, max_train_size=12, test_size=3)

            search = RandomizedSearchCV(
                base_model, 
                param_grid, 
                scoring=make_scorer(mape, greater_is_better=False), 
                cv=tscv, 
                n_iter=n_iter_search, 
                n_jobs=-1, 
                verbose=1,
                random_state=42
            )
            
            try:
                search.fit(X_train_classe, y_train_classe)
                model_classe = search.best_estimator_
                print(f"    ✅ Melhores parâmetros: {search.best_params_}")
            except Exception as e:
                print(f"    ❌ ERRO no RandomizedSearchCV: {e}")
                print("    Voltando para parâmetros padrão...")
                model_classe = xgb.XGBRegressor(**default_xgb_params)
                model_classe.fit(X_train_classe, y_train_classe)
        else:
            print(f"    ⚙️  Treinando Classe {classe} com parâmetros padrão...")
            model_classe = xgb.XGBRegressor(**default_xgb_params)
            model_classe.fit(X_train_classe, y_train_classe)
            
        # Armazenar modelo
        abc_models['models'][classe] = model_classe
        
        # Prever no treino
        pred_train_classe = model_classe.predict(X_train_classe)
        y_pred_train[mask_train] = pred_train_classe
        
        # Prever no teste
        mask_test = (classes_test == classe)
        X_test_classe = X_test[mask_test]
        if not X_test_classe.empty:
            pred_test_classe = model_classe.predict(X_test_classe)
            y_pred_test[mask_test] = pred_test_classe
            
        # Métricas de treino da classe
        mae_train_classe = mean_absolute_error(y_train_classe, pred_train_classe)
        mape_train_classe = mape(y_train_classe, pred_train_classe)
        print(f"    📈 Classe {classe} (Treino) - MAE: {mae_train_classe:.2f} | MAPE: {mape_train_classe:.4f}")

    # Métricas finais
    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_test = np.maximum(y_pred_test, 0)
    
    y_pred_train = np.nan_to_num(y_pred_train, nan=y_train.mean())
    y_pred_test = np.nan_to_num(y_pred_test, nan=y_train.mean())

    print("\n" + "="*80)
    print("📊 MÉTRICAS FINAIS (COMBINADAS)")
    print("="*80)
    
    for classe in ['A', 'B', 'C']:
        mask_train = (classes_train == classe)
        mask_test = (classes_test == classe)
        
        if mask_train.sum() > 0:
            mae_train = mean_absolute_error(y_train[mask_train], y_pred_train[mask_train])
            mape_train = mape(y_train[mask_train], y_pred_train[mask_train])
            wmape_train = wmape(y_train[mask_train], y_pred_train[mask_train])
            print(f"    TREINO Classe {classe} - MAE: {mae_train:.2f} | MAPE: {mape_train:.4f} | WMAE: {wmape_train:.4f}")
            
        if mask_test.sum() > 0:
            mae_test = mean_absolute_error(y_test[mask_test], y_pred_test[mask_test])
            mape_test = mape(y_test[mask_test], y_pred_test[mask_test])
            wmape_test = wmape(y_test[mask_test], y_pred_test[mask_test])
            print(f"    TESTE  Classe {classe} - MAE: {mae_test:.2f} | MAPE: {mape_test:.4f} | WMAE: {wmape_test:.4f}")
            
    print("-" * 40)
    mae_geral_train = mean_absolute_error(y_train, y_pred_train)
    mae_geral_test = mean_absolute_error(y_test, y_pred_test)
    wmape_geral_train = wmape(y_train, y_pred_train)
    wmape_geral_test = wmape(y_test, y_pred_test)
    print(f"    GERAL (Treino) - MAE: {mae_geral_train:.2f} | WMAE: {wmape_geral_train:.4f}")
    print(f"    GERAL (Teste)  - MAE: {mae_geral_test:.2f} | WMAE: {wmape_geral_test:.4f}")
    print("="*80)
    
    return (
        abc_models,
        y_train.values,
        y_pred_train,
        y_test.values,
        y_pred_test
    )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# GRIDS DE HIPERPARÂMETROS
# -------------------------------------------------------------------------------- 
grid_A = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.6, 0.7],
    "colsample_bytree": [0.6, 0.7],
    "min_child_weight": [15, 20, 25],
    "gamma": [1, 5],
    "reg_alpha": [1, 10],
    "reg_lambda": [1, 5]
}

grid_B = {
    "n_estimators": [150, 250, 350],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.03],
    "subsample": [0.6, 0.7],
    "colsample_bytree": [0.6, 0.8],
    "min_child_weight": [10, 15, 20],
    "gamma": [0.5, 1],
    "reg_alpha": [0.1, 1.0],
    "reg_lambda": [0.1, 1.0]
}

grid_C = {
    "n_estimators": [50, 100],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.7, 0.8],
    "min_child_weight": [10, 20]
}

user_defined_param_grid = {
    'A': grid_A,
    'B': grid_B,
    'C': grid_C
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# TREINAMENTO
# -------------------------------------------------------------------------------- 
ensemble_model, y_train, y_pred_train, y_test, y_pred_test = train_abc_models(
    df_train,
    df_test,
    features,
    product_class_map,
    user_param_grid=user_defined_param_grid,
    n_iter_search=30
)

print("\n" + "="*80)
print(f"✅ TREINAMENTO ABC CONCLUÍDO")
print(f"Modelos disponíveis: {list(ensemble_model['models'].keys())}")
print(f"Total de features utilizadas: {len(features)}")
print("="*80)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# ANÁLISE DE IMPORTÂNCIA DAS FEATURES
# ==================================================================================
print("\n" + "="*80)
print("📊 ANÁLISE DE IMPORTÂNCIA DAS FEATURES")
print("="*80)

for classe in ['A', 'B', 'C']:
    model = ensemble_model['models'].get(classe)
    if model is None:
        continue
    
    print(f"\n🔍 Classe {classe}:")
    
    # Obter importâncias
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top 10 geral
    print(f"\n   Top 10 Features Gerais:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    # Features de vendas mais importantes
    vendas_importance = feature_importance[feature_importance['feature'].str.contains('QUANT_PRED_VENDAS_MODELO')]
    if not vendas_importance.empty:
        print(f"\n   Top 5 Features de Vendas Futuras:")
        for idx, row in vendas_importance.head(5).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- 
# CRIAR DATAFRAMES DE PREDIÇÕES
# -------------------------------------------------------------------------------- 
print("\n" + "="*80)
print("📊 CRIANDO DATAFRAMES DE PREDIÇÕES")
print("="*80)

# Treino
df_pred_train = df_train[['COMPONENT', 'COD_FORNE', 'MES', 'TARGET']].copy()
df_pred_train['PREDITO'] = y_pred_train
df_pred_train['TIPO'] = 'treino'
df_pred_train['CLASSE_ABC'] = df_pred_train['COMPONENT'].map(product_class_map).fillna('C')

# Teste
df_pred_test = df_test[['COMPONENT', 'COD_FORNE', 'MES', 'TARGET']].copy()
df_pred_test['PREDITO'] = y_pred_test
df_pred_test['TIPO'] = 'teste'
df_pred_test['CLASSE_ABC'] = df_pred_test['COMPONENT'].map(product_class_map).fillna('C')

# Combinar
df_predictions_compras = pd.concat([df_pred_train, df_pred_test], ignore_index=True)

# Reordenar
cols_order = [
    'COMPONENT', 'COD_FORNE', 'MES', 'TIPO', 'CLASSE_ABC',
    'TARGET', 'PREDITO'
]
df_predictions_compras = df_predictions_compras[cols_order]
df_predictions_compras['PREDITO'] = np.ceil(df_predictions_compras['PREDITO'])

print(f"\n   ✅ Predições de Compras: {df_predictions_compras.shape}")
print(f"   Período: {df_predictions_compras['MES'].min()} a {df_predictions_compras['MES'].max()}")
print(f"   COMPONENTs únicos: {df_predictions_compras['COMPONENT'].nunique()}")
print(f"   Fornecedores únicos: {df_predictions_compras['COD_FORNE'].nunique()}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# FUNÇÃO PARA FORECAST
# ==================================================================================

def recursive_prediction(df_base, model, n_months, base_month, product_class_map):
    """
    Previsão recursiva multi-HORIZONte.
    
    Args:
        df_base: df_final (JÁ TEM todas as features calculadas)
        model: Modelo treinado (ensemble ABC)
        n_months: Número de meses a prever
        base_month: Último mês com dados reais
        product_class_map: Mapeamento COMPONENT -> Classe ABC
    """
    
    print(f"🔮 Previsão recursiva iniciada")
    print(f"    Base: {base_month}")
    print(f"    HORIZONte: {n_months} meses")
    
    # Copiar base completa
    df_work = df_base.copy()
    df_work['MES'] = pd.to_datetime(df_work['MES'])
    
    # Lista para output
    all_predictions = []
    
    # Mês atual
    current_month = base_month
    
    # Loop de previsão
    for HORIZON in range(1, n_months + 1):
        target_month = current_month + pd.DateOffset(months=1)
        
        print(f"\n    📅 HORIZONte {HORIZON}/{n_months}: Prevendo {target_month.strftime('%Y-%m')}")
        
        # 1. Filtrar linhas do mês atual (base para previsão)
        df_current = df_work[df_work['MES'] == current_month].copy()
        
        if len(df_current) == 0:
            print(f"         ⚠️  Sem dados para {current_month.strftime('%Y-%m')}")
            current_month = target_month
            continue
        
        # 2. Garantir que tem histórico suficiente
        df_current = df_current.dropna(subset=['HISTORICO_VENDAS_LAG12'])
        
        if len(df_current) == 0:
            print(f"         ⚠️  Sem histórico suficiente")
            current_month = target_month
            continue
        
        # 3. Garantir que CLASSE_ABC existe
        if 'CLASSE_ABC' not in df_current.columns:
            df_current['CLASSE_ABC'] = df_current['COMPONENT'].map(product_class_map).fillna('C')
        
        # 4. Garantir que todas as features do modelo existem
        for feat in model['feature_names']:
            if feat not in df_current.columns:
                df_current[feat] = 0
        
        # 5. Preencher NaNs
        df_current[model['feature_names']] = df_current[model['feature_names']].fillna(0)
        
        # 6. FAZER PREVISÃO
        try:
            predictions = predict_ensemble(model, df_current)
            predictions = np.maximum(predictions, 0)
            
            print(f"         ✅ {len(predictions):,} previsões | Volume: {predictions.sum():,.0f}")
            
        except Exception as e:
            print(f"         ❌ Erro: {e}")
            raise
        
        # 7. CRIAR NOVA LINHA PARA O MÊS FUTURO
        df_next = df_current.copy()
        df_next['MES'] = target_month
        
        # 8. SHIFTAR OS LAGS (a quantidade atual vira LAG1)
        df_next['HISTORICO_VENDAS_LAG12'] = df_next['HISTORICO_VENDAS_LAG11']
        df_next['HISTORICO_VENDAS_LAG11'] = df_next['HISTORICO_VENDAS_LAG10']
        df_next['HISTORICO_VENDAS_LAG10'] = df_next['HISTORICO_VENDAS_LAG9']
        df_next['HISTORICO_VENDAS_LAG9'] = df_next['HISTORICO_VENDAS_LAG8']
        df_next['HISTORICO_VENDAS_LAG8'] = df_next['HISTORICO_VENDAS_LAG7']
        df_next['HISTORICO_VENDAS_LAG7'] = df_next['HISTORICO_VENDAS_LAG6']
        df_next['HISTORICO_VENDAS_LAG6'] = df_next['HISTORICO_VENDAS_LAG5']
        df_next['HISTORICO_VENDAS_LAG5'] = df_next['HISTORICO_VENDAS_LAG4']
        df_next['HISTORICO_VENDAS_LAG4'] = df_next['HISTORICO_VENDAS_LAG3']
        df_next['HISTORICO_VENDAS_LAG3'] = df_next['HISTORICO_VENDAS_LAG2']
        df_next['HISTORICO_VENDAS_LAG2'] = df_next['HISTORICO_VENDAS_LAG1']
        df_next['HISTORICO_VENDAS_LAG1'] = df_current['QUANTIDADE'].values
        
        # 9. ATUALIZAR QUANTIDADE COM A PREVISÃO
        df_next['QUANTIDADE'] = predictions
        
        # 10. RECALCULAR FEATURES DERIVADAS
        
        # Diffs
        df_next['diff_1'] = df_next['QUANTIDADE'] - df_next['HISTORICO_VENDAS_LAG1']
        df_next['diff_2'] = df_next['HISTORICO_VENDAS_LAG1'] - df_next['HISTORICO_VENDAS_LAG2']
        df_next['diff_3'] = df_next['HISTORICO_VENDAS_LAG2'] - df_next['HISTORICO_VENDAS_LAG3']
        
        # Rolling means
        df_next['rolling_mean_3'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].mean(axis=1)
        df_next['rolling_mean_6'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                              'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].mean(axis=1)
        df_next['rolling_mean_12'] = df_next[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].mean(axis=1)
        
        # Rolling stds
        df_next['rolling_std_3'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].std(axis=1)
        df_next['rolling_std_6'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                             'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].std(axis=1)
        df_next['rolling_std_12'] = df_next[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].std(axis=1)
        
        # Rolling max
        df_next['rolling_max_3'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].max(axis=1)
        df_next['rolling_max_6'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                             'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].max(axis=1)
        df_next['rolling_max_12'] = df_next[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].max(axis=1)
        
        # Rolling min
        df_next['rolling_min_3'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3']].min(axis=1)
        df_next['rolling_min_6'] = df_next[['HISTORICO_VENDAS_LAG1', 'HISTORICO_VENDAS_LAG2', 'HISTORICO_VENDAS_LAG3', 
                                             'HISTORICO_VENDAS_LAG4', 'HISTORICO_VENDAS_LAG5', 'HISTORICO_VENDAS_LAG6']].min(axis=1)
        df_next['rolling_min_12'] = df_next[[f'HISTORICO_VENDAS_LAG{i}' for i in range(1, 13)]].min(axis=1)
        
        # Lead time lag
        df_next['lead_time_lag_1'] = df_current['LEAD_TIME'].values
        
        # Features temporais
        df_next['quarter'] = df_next['MES'].dt.quarter
        df_next['quarter_start'] = df_next['MES'].dt.is_quarter_start.astype(int)
        df_next['quarter_end'] = df_next['MES'].dt.is_quarter_end.astype(int)
        
        # One-hot do mês (zerar todos e ativar o correto)
        target_month_num = target_month.month
        for m in range(1, 13):
            df_next[f'month_{m}'] = int(m == target_month_num)
        
        # Limpar infinitos e NaNs
        numeric_cols = df_next.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_next[col] = df_next[col].replace([np.inf, -np.inf], np.nan)
        
        rolling_cols = [col for col in df_next.columns if 'rolling_' in col]
        for col in rolling_cols:
            df_next[col] = df_next[col].fillna(0)
        
        # 11. ADICIONAR AO HISTÓRICO
        df_work = pd.concat([df_work, df_next], ignore_index=True)
        
        # 12. GUARDAR PREVISÃO PARA OUTPUT
        df_pred = df_next[['COMPONENT', 'COD_FORNE', 'MES', 'QUANTIDADE', 'LEAD_TIME', 'CLASSE_ABC']].copy()
        all_predictions.append(df_pred)
        
        # 13. AVANÇAR MÊS
        current_month = target_month
    
    # Combinar previsões
    if not all_predictions:
        print("\n⚠️  Nenhuma previsão gerada!")
        return pd.DataFrame()
    
    df_forecast = pd.concat(all_predictions, ignore_index=True)
    
    print(f"\n✅ Previsão concluída!")
    print(f"    Registros: {len(df_forecast):,}")
    print(f"    COMPONENTs: {df_forecast['COMPONENT'].nunique()}")
    print(f"    Volume total: {df_forecast['QUANTIDADE'].sum():,.0f}")
    
    return df_forecast


def predict_ensemble(ensemble_model, df_features):
    """
    Faz previsões usando o modelo ABC treinado.
    """
    
    features = ensemble_model['feature_names']
    X = df_features[features].fillna(0).astype('float64')
    
    predictions = np.zeros(len(df_features))
    
    for classe in ['A', 'B', 'C']:
        mask = (df_features['CLASSE_ABC'] == classe)
        if mask.sum() == 0:
            continue
        
        X_classe = X.loc[mask]
        model_classe = ensemble_model['models'][classe]
        
        if model_classe is None:
            predictions[mask.values] = 0.0
        else:
            pred_classe = model_classe.predict(X_classe)
            predictions[mask.values] = pred_classe
    
    return np.maximum(predictions, 0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# EXECUTAR PREVISÃO RECURSIVA (12 MESES)
# ==================================================================================
print("\n" + "="*80)
print("🔮 INICIANDO PREVISÃO RECURSIVA MULTI-HORIZONTE")
print("="*80)

n_HORIZONs = 12
base_date = df_final['MES'].max()

print(f"\n📅 Configuração:")
print(f"    Data base: {base_date}")
print(f"    HORIZONte: {n_HORIZONs} meses")
print(f"    COMPONENTs: {df_final['COMPONENT'].nunique()}")
print(f"    Fornecedores: {df_final['COD_FORNE'].nunique()}")

# Executar
df_forecast_compras = recursive_prediction(
    df_base=df_final,
    model=ensemble_model,
    n_months=n_HORIZONs,
    base_month=base_date,
    product_class_map=product_class_map
)

print("\n" + "="*80)
print("✅ PREVISÃO CONCLUÍDA!")
print("="*80)

if not df_forecast_compras.empty:
    print(f"\n📊 Resumo do Forecast:")
    print(f"    Total de registros: {len(df_forecast_compras):,}")
    print(f"    COMPONENTs únicos: {df_forecast_compras['COMPONENT'].nunique()}")
    print(f"    Fornecedores únicos: {df_forecast_compras['COD_FORNE'].nunique()}")
    print(f"    Período: {df_forecast_compras['MES'].min()} até {df_forecast_compras['MES'].max()}")
    print(f"    Volume total previsto: {df_forecast_compras['QUANTIDADE'].sum():,.0f}")
    
    # Estatísticas por classe
    print("\n📈 Distribuição do Forecast por Classe ABC:")
    for classe in ['A', 'B', 'C']:
        df_classe = df_forecast_compras[df_forecast_compras['CLASSE_ABC'] == classe]
        if len(df_classe) > 0:
            volume = df_classe['QUANTIDADE'].sum()
            pct = volume / df_forecast_compras['QUANTIDADE'].sum() * 100
            n_components = df_classe['COMPONENT'].nunique()
            
            print(f"    Classe {classe}:")
            print(f"        COMPONENTs: {n_components}")
            print(f"        Volume: {volume:,.0f} ({pct:.1f}%)")
    
    # Amostra
    print("\n📋 Primeiras previsões:")
    print(df_forecast_compras.head(10))
else:
    print("\n⚠️  Nenhuma previsão foi gerada. Verifique os dados de entrada.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def expandir_custo_por_mes(df, data_final):
    """
    Expande MES para cada COMPONENT + COD_FORNE até 'data_final',
    criando linhas faltantes e preenchendo MOEDA e PRECO_UNIT
    com forward-fill (valor do mês anterior).
    
    Parâmetros:
        df : DataFrame contendo colunas:
             COMPONENT, COD_FORNE, MES, MOEDA, PRECO_UNIT
        data_final : str ou datetime
             Exemplo: '2025-12-01'
    
    Retorno:
        DataFrame expandido.
    """

    # Garantir datetime
    df = df.copy()
    df['MES'] = pd.to_datetime(df['MES'])
    data_final = pd.to_datetime(data_final)

    def expandir_grupo(g):
        # Ordena por MES
        g = g.sort_values('MES')

        # Se houver MES duplicado no mesmo COMPONENT+COD_FORNE, mantém só a última
        g = g.drop_duplicates(subset='MES', keep='last')

        # Define o range do primeiro MES até data_final
        full_idx = pd.date_range(
            g['MES'].min(), 
            data_final, 
            freq='MS'  # Month Start
        )

        # Coloca MES como índice e expande
        g = g.set_index('MES').reindex(full_idx)
        g.index.name = 'MES'

        # Preenche COMPONENT e COD_FORNE (fixos no grupo)
        g['COMPONENT'] = g['COMPONENT'].ffill().bfill()
        g['COD_FORNE'] = g['COD_FORNE'].ffill().bfill()

        # Preenche MOEDA e PRECO_UNIT com valores do mês anterior
        g['MOEDA'] = g['MOEDA'].ffill()
        g['PRECO_UNIT'] = g['PRECO_UNIT'].ffill()

        return g

    # Aplica a expansão por grupo
    df_expanded = (
        df
        .groupby(['COMPONENT', 'COD_FORNE'], group_keys=False)
        .apply(expandir_grupo)
        .reset_index()
    )

    return df_expanded


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_forecast_compras['QUANTIDADE'] = np.ceil(df_forecast_compras['QUANTIDADE'])
max_month = df_forecast_compras['MES'].min() + pd.DateOffset(months=2)
df_hist_recent = df_historico_pedidos[['COMPONENT', 'COD_FORNE', 'MES', 'MOEDA', 'PRECO_UNIT']].sort_values(['COMPONENT', 'COD_FORNE', 'MES'], ascending=[True, True, False]).drop_duplicates(subset=['COMPONENT', 'COD_FORNE', 'MES'], keep='first')
df_hist_recent = expandir_custo_por_mes(
    df_historico_pedidos[['COMPONENT', 'COD_FORNE', 'MES', 'MOEDA', 'PRECO_UNIT']],
    data_final=max_month
)

# 5) Merge no df_forecast_compras
df_forecast_compras = df_forecast_compras.merge(
    df_hist_recent,
    on=['COMPONENT', 'COD_FORNE', 'MES'],
    how='left'
)
df_forecast_compras = df_forecast_compras[~df_forecast_compras['PRECO_UNIT'].isna()] 
df_forecast_compras = df_forecast_compras[[*df_forecast_compras.drop(columns='CLASSE_ABC').columns, 'CLASSE_ABC']]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# COMPENSAÇÃO INTELIGENTE - MOVIMENTAÇÃO PROPORCIONAL DE QUANTIDADES
# ==================================================================================
def verificar_e_compensar_faturamento_minimo(df_forecast, df_faturamento_minimo, n_meses_output=3):
    """
    Compensa faturamento movendo COMPONENTES INTEIROS.
    Trabalha com PRECO_UNIT e calcula PRECO_TOTAL = QUANTIDADE * PRECO_UNIT.
    """
    
    print("\n" + "="*80)
    print(f"💰 COMPENSAÇÃO INTELIGENTE V3 - OUTPUT: {n_meses_output} MESES")
    print("="*80)
    
    # 1. PREPARAR DADOS
    df_work = df_forecast.copy()
    df_work['MES'] = pd.to_datetime(df_work['MES'])
    df_work['COD_FORNE'] = df_work['COD_FORNE'].astype(str)
    
    # ✅ CALCULAR PRECO_TOTAL inicial
    if 'PRECO_TOTAL' not in df_work.columns:
        df_work['PRECO_TOTAL'] = (df_work['QUANTIDADE'] * df_work['PRECO_UNIT']).round(2)
    
    meses_unicos = sorted(df_work['MES'].unique())
    meses_output = meses_unicos[:n_meses_output]
    
    print(f"\n📅 Output: {n_meses_output} meses | Compensação: {len(meses_unicos)} meses")
    
    # 2. IDENTIFICAR COLUNAS FATURAMENTO
    col_map = {col.lower(): col for col in df_faturamento_minimo.columns}
    
    col_forn_fat = None
    for p in ['cod_forne', 'codforne', 'cod_fornecedor', 'fornecedor', 'supp_code']:
        if p in col_map:
            col_forn_fat = col_map[p]
            break
    
    col_fat = None
    for p in ['faturamento_minimo', 'faturamentominimo', 'faturamento', 'fatur.min.']:
        if p in col_map:
            col_fat = col_map[p]
            break
    
    if not col_forn_fat or not col_fat:
        print("⚠️ ERRO: Colunas não encontradas")
        return None, None
    
    # 3. MAPA DE FATURAMENTO
    df_fat = df_faturamento_minimo[[col_forn_fat, col_fat]].copy()
    df_fat = df_fat.rename(columns={col_forn_fat: 'COD_FORNE', col_fat: 'FATURAMENTO_MINIMO'})
    df_fat['COD_FORNE'] = df_fat['COD_FORNE'].astype(str)
    faturamento_map = df_fat.set_index('COD_FORNE')['FATURAMENTO_MINIMO'].to_dict()
    
    # 4. PRIORIDADES ABC
    prioridade_abc = {'A': 1, 'B': 2, 'C': 3}
    
    # 5. COMPENSAÇÃO INTELIGENTE
    print("\n📊 Aplicando compensação (componentes inteiros)...")
    
    fornecedores = df_work['COD_FORNE'].unique()
    movimentos = []
    indices_para_remover = []
    
    for fornecedor in fornecedores:
        fat_min = faturamento_map.get(fornecedor, 0)
        if fat_min == 0:
            continue
        
        df_forn = df_work[df_work['COD_FORNE'] == fornecedor].copy()
        meses_forn = sorted(df_forn['MES'].unique())
        
        # Processar apenas os N primeiros meses
        for mes_atual in meses_forn[:n_meses_output]:
            
            # Calcular valor atual
            mask_atual = (df_work['COD_FORNE'] == fornecedor) & (df_work['MES'] == mes_atual)
            valor_atual = df_work.loc[mask_atual, 'PRECO_TOTAL'].sum()
            
            if valor_atual >= fat_min:
                continue
            
            deficit = fat_min - valor_atual
            
            # Buscar em meses futuros
            meses_futuros = [m for m in meses_forn if m > mes_atual]
            
            for mes_futuro in meses_futuros:
                if deficit <= 0.01:
                    break
                
                # Pegar componentes do mês futuro
                mask_futuro = (df_work['COD_FORNE'] == fornecedor) & (df_work['MES'] == mes_futuro)
                df_futuro = df_work[mask_futuro].copy()
                
                if df_futuro.empty:
                    continue
                
                # 🎯 ORDENAR: ABC (A>B>C), depois por valor (maior>menor)
                df_futuro['PRIORIDADE'] = df_futuro['CLASSE_ABC'].map(prioridade_abc).fillna(3)
                df_futuro = df_futuro.sort_values(['PRIORIDADE', 'PRECO_TOTAL'], ascending=[True, False])
                
                # 🚀 MOVER COMPONENTES INTEIROS
                for idx, row in df_futuro.iterrows():
                    if deficit <= 0.01:
                        break
                    
                    qtd_total = row['QUANTIDADE']
                    preco_unit = row['PRECO_UNIT']
                    valor_total = row['PRECO_TOTAL']
                    
                    # Decidir: mover inteiro ou fazer split
                    if valor_total <= deficit + 0.01:
                        # ✅ MOVER COMPONENTE INTEIRO
                        qtd_mover = qtd_total
                        valor_mover = valor_total
                        
                        # Marcar para remover
                        indices_para_remover.append(idx)
                    else:
                        # ✅ SPLIT - calcular quantidade necessária
                        qtd_necessaria = deficit / preco_unit if preco_unit > 0 else 0
                        qtd_mover = np.ceil(qtd_necessaria)
                        qtd_mover = min(qtd_mover, qtd_total)
                        
                        # ✅ CALCULAR VALOR com 2 casas decimais
                        valor_mover = round(qtd_mover * preco_unit, 2)
                        
                        # Atualizar mês futuro (reduzir)
                        nova_qtd_futuro = qtd_total - qtd_mover
                        df_work.at[idx, 'QUANTIDADE'] = nova_qtd_futuro
                        df_work.at[idx, 'PRECO_TOTAL'] = round(nova_qtd_futuro * preco_unit, 2)
                    
                    # ADICIONAR NO MÊS ATUAL
                    mask_existe = (
                        (df_work['COD_FORNE'] == fornecedor) & 
                        (df_work['MES'] == mes_atual) & 
                        (df_work['COMPONENT'] == row['COMPONENT'])
                    )
                    
                    if mask_existe.any():
                        # Já existe - SOMAR
                        idx_existe = df_work[mask_existe].index[0]
                        nova_qtd = df_work.at[idx_existe, 'QUANTIDADE'] + qtd_mover
                        df_work.at[idx_existe, 'QUANTIDADE'] = nova_qtd
                        # ✅ RECALCULAR PRECO_TOTAL com 2 casas decimais
                        df_work.at[idx_existe, 'PRECO_TOTAL'] = round(nova_qtd * preco_unit, 2)
                    else:
                        # Criar nova linha
                        nova_linha = {
                            'COMPONENT': row['COMPONENT'],
                            'COD_FORNE': fornecedor,
                            'MES': mes_atual,
                            'QUANTIDADE': qtd_mover,
                            'LEAD_TIME': row['LEAD_TIME'],
                            'MOEDA': row['MOEDA'],
                            'PRECO_UNIT': preco_unit,
                            'PRECO_TOTAL': valor_mover,
                            'CLASSE_ABC': row['CLASSE_ABC']
                        }
                        df_work = pd.concat([df_work, pd.DataFrame([nova_linha])], ignore_index=True)
                    
                    # Registrar movimento
                    movimentos.append({
                        'COD_FORNE': fornecedor,
                        'COMPONENT': row['COMPONENT'],
                        'MES_ORIGEM': pd.to_datetime(mes_futuro).strftime('%Y-%m'),
                        'MES_DESTINO': pd.to_datetime(mes_atual).strftime('%Y-%m'),
                        'QUANTIDADE_MOVIDA': qtd_mover,
                        'VALOR_MOVIDO': valor_mover,
                        'CLASSE_ABC': row['CLASSE_ABC'],
                        'TIPO': 'INTEIRO' if qtd_mover == qtd_total else 'PARCIAL'
                    })
                    
                    deficit -= valor_mover
    
    # 6. REMOVER COMPONENTES MOVIDOS INTEIROS
    if indices_para_remover:
        df_work = df_work.drop(index=indices_para_remover).reset_index(drop=True)
    
    # 7. REMOVER QUANTIDADES ZERO
    df_work = df_work[df_work['QUANTIDADE'] > 0.01].copy()
    
    # 8. ARREDONDAR QUANTIDADES E RECALCULAR PREÇOS
    print(f"\n🔢 Arredondando quantidades...")
    df_work['QUANTIDADE'] = np.ceil(df_work['QUANTIDADE'])
    
    # ✅ RECALCULAR PRECO_TOTAL com quantidades arredondadas (2 casas decimais)
    df_work['PRECO_TOTAL'] = (df_work['QUANTIDADE'] * df_work['PRECO_UNIT']).round(2)
    
    # 9. FILTRAR APENAS OS N PRIMEIROS MESES
    print(f"\n✂️  Filtrando {n_meses_output} primeiros meses...")
    df_output = df_work[df_work['MES'].isin(meses_output)].copy()
    
    # 10. AGREGAR (caso tenha duplicatas)
    print(f"\n📊 Agregando...")
    df_output_final = df_output.groupby(
        ['COMPONENT', 'COD_FORNE', 'MES'], 
        as_index=False
    ).agg({
        'QUANTIDADE': 'sum',
        'PRECO_UNIT': 'first',  # Preço unitário permanece o mesmo
        'LEAD_TIME': 'first',
        'MOEDA': 'first',
        'CLASSE_ABC': 'first'
    })
    
    # ✅ RECALCULAR PRECO_TOTAL após agregação (2 casas decimais)
    df_output_final['QUANTIDADE'] = np.ceil(df_output_final['QUANTIDADE'])
    df_output_final['PRECO_TOTAL'] = (df_output_final['QUANTIDADE'] * df_output_final['PRECO_UNIT']).round(2)
    
    # ✅ ORDENAR COLUNAS
    df_output_final = df_output_final[[
        'COMPONENT', 'COD_FORNE', 'MES', 'QUANTIDADE', 
        'LEAD_TIME', 'MOEDA', 'PRECO_TOTAL', 'CLASSE_ABC'
    ]]
    
    df_output_final = df_output_final.sort_values(['COD_FORNE', 'MES', 'COMPONENT'])
    
    print(f"   ✅ Linhas finais: {len(df_output_final):,}")
    
    # 11. CRIAR RESUMO
    df_resumo = df_output_final.groupby(['COD_FORNE', 'MES']).agg({
        'PRECO_TOTAL': 'sum',
        'COMPONENT': 'count'
    }).reset_index()
    
    df_resumo = df_resumo.rename(columns={
        'PRECO_TOTAL': 'VALOR_FINAL',
        'COMPONENT': 'N_COMPONENTES'
    })
    
    df_resumo['COD_FORNE'] = df_resumo['COD_FORNE'].astype(str)
    df_resumo['FATURAMENTO_MINIMO'] = df_resumo['COD_FORNE'].map(faturamento_map).fillna(0)
    
    # Valores originais
    df_original = df_forecast.copy()
    df_original['MES'] = pd.to_datetime(df_original['MES'])
    df_original['COD_FORNE'] = df_original['COD_FORNE'].astype(str)
    
    # ✅ CALCULAR PRECO_TOTAL original se não existir
    if 'PRECO_TOTAL' not in df_original.columns:
        df_original['PRECO_TOTAL'] = (df_original['QUANTIDADE'] * df_original['PRECO_UNIT']).round(2)
    
    df_orig_valores = df_original[df_original['MES'].isin(meses_output)].groupby(['COD_FORNE', 'MES'])['PRECO_TOTAL'].sum().reset_index()
    df_orig_valores = df_orig_valores.rename(columns={'PRECO_TOTAL': 'VALOR_ORIGINAL'})
    
    df_resumo = pd.merge(df_resumo, df_orig_valores, on=['COD_FORNE', 'MES'], how='left')
    df_resumo['VALOR_ORIGINAL'] = df_resumo['VALOR_ORIGINAL'].fillna(0)
    df_resumo['VALOR_RECEBIDO'] = (df_resumo['VALOR_FINAL'] - df_resumo['VALOR_ORIGINAL']).clip(lower=0)
    
    # Status
    df_resumo['ATENDE_MINIMO_ORIGINAL'] = df_resumo['VALOR_ORIGINAL'] >= df_resumo['FATURAMENTO_MINIMO']
    df_resumo['ATENDE_MINIMO_FINAL'] = df_resumo['VALOR_FINAL'] >= df_resumo['FATURAMENTO_MINIMO']
    df_resumo['STATUS'] = df_resumo.apply(
        lambda r: 'Atende' if r['ATENDE_MINIMO_FINAL'] 
                  else f"Falta R${r['FATURAMENTO_MINIMO'] - r['VALOR_FINAL']:,.2f}",
        axis=1
    )
    
    df_resumo = df_resumo[[
        'COD_FORNE', 'MES', 'N_COMPONENTES',
        'VALOR_ORIGINAL', 'VALOR_RECEBIDO', 'VALOR_FINAL',
        'FATURAMENTO_MINIMO', 'ATENDE_MINIMO_ORIGINAL', 'ATENDE_MINIMO_FINAL', 'STATUS'
    ]]
    
    # 12. ESTATÍSTICAS
    print("\n" + "="*80)
    print("📊 ESTATÍSTICAS")
    print("="*80)
    print(f"   Movimentos realizados: {len(movimentos)}")
    
    n_orig = (~df_resumo['ATENDE_MINIMO_ORIGINAL']).sum()
    n_final = (~df_resumo['ATENDE_MINIMO_FINAL']).sum()
    n_resolvidos = ((~df_resumo['ATENDE_MINIMO_ORIGINAL']) & df_resumo['ATENDE_MINIMO_FINAL']).sum()
    
    print(f"   Antes: {n_orig} não atingiam | Resolvidos: {n_resolvidos} | Restam: {n_final}")
    
    if n_orig > 0:
        print(f"   Taxa de resolução: {(n_resolvidos/n_orig)*100:.1f}%")
    
    # Status por mês
    print(f"\n   🎯 Status por mês:")
    for mes in meses_output:
        mask = df_resumo['MES'] == mes
        atende = df_resumo[mask & df_resumo['ATENDE_MINIMO_FINAL']].shape[0]
        total = df_resumo[mask].shape[0]
        if total > 0:
            print(f"      {pd.to_datetime(mes).strftime('%Y-%m')}: {atende}/{total} ({atende/total*100:.1f}%)")
    
    # Tipo de movimentos
    if movimentos:
        df_mov = pd.DataFrame(movimentos)
        n_inteiros = (df_mov['TIPO'] == 'INTEIRO').sum()
        n_parciais = (df_mov['TIPO'] == 'PARCIAL').sum()
        print(f"\n   📦 Tipo de movimentos:")
        print(f"      Componentes inteiros: {n_inteiros}")
        print(f"      Parciais (split): {n_parciais}")
        
        # Por classe ABC
        print(f"\n   📦 Movimentos por classe ABC:")
        for classe in ['A', 'B', 'C']:
            mov_classe = df_mov[df_mov['CLASSE_ABC'] == classe]
            if not mov_classe.empty:
                qtd = mov_classe['QUANTIDADE_MOVIDA'].sum()
                valor = mov_classe['VALOR_MOVIDO'].sum()
                print(f"      Classe {classe}: {len(mov_classe)} movimentos | "
                      f"Qtd: {qtd:,.0f} | R$ {valor:,.2f}")
    
    print("\n" + "="*80)
    
    return df_resumo, df_output_final

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# EXECUTAR ANÁLISE DE FATURAMENTO MINIMO
# ==================================================================================
df_faturamento_minimo = Helpers.getEntityData(context, 'faturamento_minimo').rename(columns={
    'Faturamento_M_nimo': 'FATURAMENTO_MINIMO',
    'cod._Fornecedor': 'COD_FORNE',
}).drop(['Fornecedor'], axis=1)

df_analise_faturamento, df_forecast_compras = verificar_e_compensar_faturamento_minimo(
    df_faturamento_minimo=df_faturamento_minimo,
    df_forecast=df_forecast_compras
)

if df_analise_faturamento is not None:
    print("\n📋 Amostra do RESUMO:")
    print(df_analise_faturamento.head(20))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Adicionando os resultados da predicao
df_model_data = df_model_data[['COMPONENT', 'COD_FORNE', 'MES', 'LEAD_TIME', 'TARGET', 'QUANTIDADE']]
df_model_data = pd.merge(
    df_model_data, 
    df_predictions_compras[['COMPONENT', 'COD_FORNE', 'MES', 'PREDITO', 'CLASSE_ABC']], 
    on=['COMPONENT', 'COD_FORNE', 'MES'],
    how='left'
    ).rename(columns = {
    'TARGET':'QUANT_TARGET',
    'QUANTIDADE':'QUANT_REALIZADA',
    'PREDITO':'QUANT_PREDITA'
    })

# Adicionando os custos
df_model_data = pd.merge(
    df_model_data, 
    df_hist_recent[['COMPONENT', 'COD_FORNE', 'MES', 'MOEDA', 'PRECO_UNIT']], 
    on=['COMPONENT', 'COD_FORNE', 'MES'],
    how='left'
    )

# Adicionando os custos
df_model_data['PRECO_TARGET'] = df_model_data['QUANT_TARGET'] * df_model_data['PRECO_UNIT']
df_model_data['PRECO_REALIZADO'] = df_model_data['QUANT_REALIZADA'] * df_model_data['PRECO_UNIT']
df_model_data['PRECO_PREDITO'] = df_model_data['QUANT_PREDITA'] * df_model_data['PRECO_UNIT']

df_forecast_compras = df_forecast_compras.rename(columns = {
    'QUANTIDADE':'QUANT_PREDITA',
    'PRECO_TOTAL':'PRECO_PREDITO'
})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# ==================================================================================
# SALVAR RESULTADOS
# ==================================================================================
print("\n" + "="*80)
print("💾 SALVANDO RESULTADOS")
print("="*80)

# Predições de Treino/Teste
Helpers.save_output_dataset(
    context=context,
    output_name='df_historico_pedidos_realizados',
    data_frame=df_model_data
)
print("    ✅ predicoes_compras_abc_xgboost")

# Forecast 12 meses
Helpers.save_output_dataset(
    context=context,
    output_name='df_historico_pedidos_previstos',
    data_frame=df_forecast_compras
)
print("    ✅ df_forecast_compras_12m")

print("\n" + "="*80)
print("✅ PROCESSAMENTO COMPLETO!")
print("="*80)