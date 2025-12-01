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
from utils.rcclient.enums import DataSourceType
from utils.rc.dtos.global_variable import GlobalVariable
from utils.rc.dtos.project import Project
from utils.rc.client.requests import Requests

import pandas as pd
import numpy as np
import datetime
import os
import json
import logging

# Inicializa o contexto
context = Helpers.getOrCreateContext(contextId='contextId', localVars=locals())

# ------------------------------------------------------------------------------
# Funções auxiliares
def normalize_supplier_code(value):
    """Normaliza código de fornecedor para string sem .0, com tratamento de NaN/None."""
    try:
        if pd.isna(value) or value == '-' or value == '':
            return ""
        # Se for string como "2880.0", converte para "2880"
        if isinstance(value, str) and '.' in value:
            return str(int(float(value)))
        # Se for float/int, converte direto
        return str(int(float(value)))
    except (ValueError, TypeError):
        # fallback: string pura, removendo espaços
        return str(value).strip()

# Função auxiliar: Normaliza dados para JSON serializable
def normalize_for_json(value):
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        return value.fillna("").replace([np.inf, -np.inf], "").to_dict('records')
    if isinstance(value, pd.Series):
        return value.fillna("").replace([np.inf, -np.inf], "").tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value

def normalize_structure(obj):
    if isinstance(obj, dict):
        return {k: normalize_structure(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_structure(v) for v in obj]
    else:
        return normalize_for_json(obj)

def datetime_to_string_aammdd(date_input):
    """
    Converte datas para string no formato AAMMDD.
    
    Suporta:
    - datetime.datetime
    - pandas.Timestamp
    - string nos formatos: YYYY-MM-DD, DD/MM/YYYY, YYYYMMDD
    - pandas.Series com qualquer formato acima
    """
    
    # Se for Series, processa de forma vetorizada
    if isinstance(date_input, pd.Series):
        # Tenta converter com inferência automática
        dt_series = pd.to_datetime(date_input, errors='coerce', format=None, dayfirst=True)
        
        # Caso algumas datas não sejam reconhecidas, tenta manualmente os formatos
        mask_na = dt_series.isna() & date_input.notna()
        if mask_na.any():
            dt_series[mask_na] = pd.to_datetime(date_input[mask_na], errors='coerce', format='%Y%m%d')
        
        return dt_series.dt.strftime('%Y%m%d')

    # Se for único valor
    if isinstance(date_input, (datetime.datetime, pd.Timestamp)):
        return date_input.strftime('%Y%m%d')

    if isinstance(date_input, str):
        # Tenta primeiro com inferência
        try:
            date_obj = pd.to_datetime(date_input, dayfirst=True)
        except Exception:
            try:
                date_obj = datetime.datetime.strptime(date_input, '%Y%m%d')
            except Exception:
                raise ValueError(f"Formato de data não reconhecido: {date_input}")
        return date_obj.strftime('%Y%m%d')

    raise ValueError(f"Tipo de entrada não suportado: {type(date_input)}")

# ------------------------------------------------------------------------------
def min_order_value_warning(df_main_user, df_faturamento_minimo):
    """
    Retorna um dicionário onde:
    - True = NÃO atingiu o mínimo (precisa de aviso/warning)
    - False = ATINGIU o mínimo (não precisa de aviso)
    """
    supplier_list = df_main_user["Supp_Cod"].unique()
    suppliers_warning_needed = {}
    
    logging.info(f"🔍 DEBUG: Analisando {len(supplier_list)} fornecedores para faturamento mínimo")
    logging.info(f"🔍 DEBUG: Fornecedores únicos no df_main_user: {supplier_list}")
    logging.info(f"🔍 DEBUG: Colunas df_faturamento_minimo: {df_faturamento_minimo.columns.tolist()}")
    
    # Mostrar todos os códigos de fornecedor do df_faturamento_minimo
    if 'cod._Fornecedor' in df_faturamento_minimo.columns:
        fat_min_suppliers = df_faturamento_minimo['cod._Fornecedor'].unique()
        logging.info(f"🔍 DEBUG: Fornecedores no faturamento mínimo: {fat_min_suppliers}")
    
    for supplier in supplier_list:
        logging.info(f"🔍 DEBUG: ===== PROCESSANDO FORNECEDOR {supplier} =====")
        
        df_order_supplier = df_main_user[df_main_user["Supp_Cod"] == supplier]
        if not df_order_supplier.empty:
            if 'Final_order' in df_order_supplier.columns and 'Cost' in df_order_supplier.columns:
                # CORREÇÃO: Converter para numérico antes de multiplicar
                final_order = pd.to_numeric(df_order_supplier['Final_order'], errors='coerce').fillna(0)
                cost = pd.to_numeric(df_order_supplier['Cost'], errors='coerce').fillna(0)
                total = (final_order * cost).sum()
                
                logging.info(f"🔍 DEBUG: Final_order: {final_order.tolist()}")
                logging.info(f"🔍 DEBUG: Cost: {cost.tolist()}")
                logging.info(f"🔍 DEBUG: Total calculado: {total}")
            else:
                total = 0
                
            try:
                # Determinar coluna do fornecedor no df_faturamento_minimo
                if 'Supp_Code' in df_faturamento_minimo.columns:
                    col_supplier = 'Supp_Code'
                elif 'Supp Code' in df_faturamento_minimo.columns:
                    col_supplier = 'Supp Code'
                elif 'cod._Fornecedor' in df_faturamento_minimo.columns:
                    col_supplier = 'cod._Fornecedor'
                else:
                    # Tentar encontrar uma coluna que contenha 'cod' ou 'supp'
                    possible_cols = [col for col in df_faturamento_minimo.columns 
                                   if 'cod' in col.lower() or 'supp' in col.lower()]
                    if possible_cols:
                        col_supplier = possible_cols[0]
                    else:
                        logging.warning(f"🔍 DEBUG: Nenhuma coluna de fornecedor encontrada para {supplier}")
                        continue
                
                logging.info(f"🔍 DEBUG: Usando coluna '{col_supplier}' para buscar fornecedor")
                
                # Normalizar códigos para comparação
                supplier_normalized = normalize_supplier_code(supplier)
                logging.info(f"🔍 DEBUG: Fornecedor normalizado: '{supplier}' -> '{supplier_normalized}'")
                
                # Normalizar coluna de fornecedores no df_faturamento_minimo
                df_faturamento_minimo[col_supplier] = df_faturamento_minimo[col_supplier].astype('str')
                
                # Tentar múltiplas formas de busca
                search_attempts = [
                    str(supplier),  # Original
                    supplier_normalized,  # Normalizado
                    str(supplier).replace('.0', ''),  # Sem .0
                ]
                
                logging.info(f"🔍 DEBUG: Tentativas de busca: {search_attempts}")
                
                aux_fat_min = pd.DataFrame()
                for search_val in search_attempts:
                    aux_fat_min = df_faturamento_minimo[df_faturamento_minimo[col_supplier] == search_val]
                    logging.info(f"🔍 DEBUG: Buscando '{search_val}': {len(aux_fat_min)} resultados")
                    if not aux_fat_min.empty:
                        break
                
                if not aux_fat_min.empty:
                    aux_fat_min = aux_fat_min.reset_index(drop=True)
                    
                    # Determinar coluna de faturamento mínimo
                    if 'Fatur.Min.' in aux_fat_min.columns:
                        fat_min = aux_fat_min.loc[0, 'Fatur.Min.']
                        logging.info(f"🔍 DEBUG: Usando coluna 'Fatur.Min.'")
                    elif 'Faturamento_M_nimo' in aux_fat_min.columns:
                        fat_min = aux_fat_min.loc[0, 'Faturamento_M_nimo']
                        logging.info(f"🔍 DEBUG: Usando coluna 'Faturamento_M_nimo'")
                    elif 'Faturamento_Minimo' in aux_fat_min.columns:
                        fat_min = aux_fat_min.loc[0, 'Faturamento_Minimo']
                        logging.info(f"🔍 DEBUG: Usando coluna 'Faturamento_Minimo'")
                    else:
                        logging.warning(f"🔍 DEBUG: Nenhuma coluna de faturamento mínimo encontrada. Colunas disponíveis: {aux_fat_min.columns.tolist()}")
                        continue
                    
                    # Garantir que fat_min seja numérico
                    fat_min_original = fat_min
                    fat_min = pd.to_numeric(fat_min, errors='coerce')
                    if pd.isna(fat_min):
                        fat_min = 0
                    
                    logging.info(f"🔍 DEBUG: Faturamento mínimo bruto: '{fat_min_original}' -> numérico: {fat_min}")
                    
                    # CORREÇÃO DA LÓGICA:
                    # True = NÃO atingiu o mínimo (total < fat_min) = PRECISA DE AVISO
                    # False = ATINGIU o mínimo (total >= fat_min) = NÃO PRECISA DE AVISO
                    needs_warning = total < fat_min
                    suppliers_warning_needed[supplier] = needs_warning
                    
                    logging.info(f"🔍 DEBUG: ===== RESULTADO FINAL =====")
                    logging.info(f"🔍 DEBUG: Fornecedor {supplier}: Total={total}, Mínimo={fat_min}, Warning={needs_warning}")
                    logging.info(f"🔍 DEBUG: =============================")
                else:
                    logging.warning(f"🔍 DEBUG: Fornecedor {supplier} NÃO encontrado no df_faturamento_minimo")
                    
                    # Mostrar todos os fornecedores disponíveis para debug
                    available_suppliers = df_faturamento_minimo[col_supplier].unique().tolist()
                    logging.info(f"🔍 DEBUG: Fornecedores disponíveis: {available_suppliers}")
                        
            except Exception as e:
                logging.error(f"🔍 DEBUG: Erro ao processar fornecedor {supplier}: {e}")
                continue

    logging.info(f"🔍 DEBUG: ===== RESUMO WARNINGS =====")
    for supplier, warning in suppliers_warning_needed.items():
        logging.info(f"🔍 DEBUG: {supplier}: Warning = {warning}")
    logging.info(f"🔍 DEBUG: =============================")

    return suppliers_warning_needed

# ------------------------------------------------------------------------------
def create_protheus_dataframe(df_main_user, df_produto_fornecedor, forn,
                            filial="1", emissao=None, unidreq="1", codcomp="35",
                            local="1", cc="20", fabr=None, transittime=32, skip_zero=True):
    def safe_str_int(value):
        try:
            if pd.isna(value):
                return None
            return str(int(float(value)))
        except (ValueError, TypeError):
            return None

    df_main_user = df_main_user[df_main_user["Supp_Cod"] == forn]

    if not df_main_user.empty:
        # Agregar produtos duplicados
        agg_dict = {
            'Final_order': lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum(),
            'Cost': 'first',
            'Currency': 'first',
            'LT': 'first',
            'Supp_Cod': 'first'
        }

        # Incluir outras colunas que possam existir
        for col in df_main_user.columns:
            if col not in ['Component'] + list(agg_dict.keys()):
                agg_dict[col] = 'first'

        df_main_user = df_main_user.groupby('Component').agg(agg_dict).reset_index()
        df_main_user = df_main_user.set_index('Component', drop=False)

    if 'Component' not in df_main_user.index.names:
        df_main_user = df_main_user.set_index('Component', drop=False)

    protheus_columns = ["Filial", "Emissao", "UnidReq", "CodComp", "Produto", "Local",
                        "Quant", "CC", "DatPRF", "DatEmbarque", "Forn", "Fabr",
                        "Moeda", "Preco_Unit"]
    df_protheus = pd.DataFrame(columns=protheus_columns)
    current_day = datetime.datetime.now() if emissao is None else datetime.datetime.strptime(emissao, "%d/%m/%Y")

    # Prepara df_produto_fornecedor
    df_produto_fornecedor = df_produto_fornecedor.copy()
    if 'COD_FORNE' in df_produto_fornecedor.columns:
        df_produto_fornecedor["COD_FORNE"] = df_produto_fornecedor["COD_FORNE"].replace('<NA>', pd.NA)
        df_produto_fornecedor["COD_FORNE"] = pd.to_numeric(df_produto_fornecedor["COD_FORNE"], errors='coerce')

    if not pd.isna(forn):
        try:
            # Normaliza o código do fornecedor para comparação
            forn_normalized = normalize_supplier_code(forn)
            forn_float = float(forn_normalized) if forn_normalized else None

            if forn_float and 'COD_FORNE' in df_produto_fornecedor.columns:
                df_produto_fornecedor_filtered = df_produto_fornecedor[df_produto_fornecedor["COD_FORNE"] == forn_float]
            else:
                df_produto_fornecedor_filtered = pd.DataFrame(columns=df_produto_fornecedor.columns)
        except (ValueError, TypeError):
            df_produto_fornecedor_filtered = pd.DataFrame(columns=df_produto_fornecedor.columns)
    else:
        df_produto_fornecedor_filtered = pd.DataFrame(columns=df_produto_fornecedor.columns)

    for _, row in df_main_user.iterrows():
        # CORREÇÃO: Melhor conversão de Final_order
        try:
            final_order_value = row.get("Final_order", 0)
            if isinstance(final_order_value, str):
                final_order_value = final_order_value.replace(',', '.')  # Para casos com vírgula decimal
            final_order = float(pd.to_numeric(final_order_value, errors='coerce'))
            if pd.isna(final_order):
                final_order = 0
        except (ValueError, TypeError):
            final_order = 0

        if final_order == 0 and skip_zero:
            continue

        component = row["Component"]

        # Calcula datas
        try:
            lt_value = row.get("LT", 0)
            lt_days = int(float(pd.to_numeric(lt_value, errors='coerce')))
            if pd.isna(lt_days):
                lt_days = 0
        except (ValueError, TypeError):
            lt_days = 0

        datprf = current_day + datetime.timedelta(days=lt_days)
        datembarque = datprf - datetime.timedelta(days=transittime)

        data_row = {
            "Filial": filial,
            "Emissao": current_day.strftime("%Y%m%d") if emissao is None else emissao,
            "UnidReq": unidreq,
            "CodComp": codcomp,
            "Produto": component,
            "Local": local,
            "Quant": int(final_order),
            "CC": cc,
            "DatPRF": datprf.strftime("%Y%m%d"),
            "DatEmbarque": datembarque.strftime("%Y%m%d"),
            "Forn": safe_str_int(normalize_supplier_code(forn)),
        }

        # Busca fabricante
        if 'COD_FABRI' in df_produto_fornecedor_filtered.columns and 'PRODUTO' in df_produto_fornecedor_filtered.columns:
            fabri_values = df_produto_fornecedor_filtered[
                df_produto_fornecedor_filtered["PRODUTO"] == component
            ]["COD_FABRI"].values
            fabri = fabri_values[0] if len(fabri_values) > 0 else None
        else:
            fabri = None
        data_row["Fabr"] = safe_str_int(fabri)

        # Moeda e preço
        data_row["Moeda"] = str(row.get("Currency", "")) if not pd.isna(row.get("Currency")) else ""
        
        # CORREÇÃO: Melhor conversão de Cost
        try:
            cost_value = row.get("Cost", 0)
            if isinstance(cost_value, str):
                cost_value = cost_value.replace(',', '.')  # Para casos com vírgula decimal
            cost = float(pd.to_numeric(cost_value, errors='coerce'))
            if pd.isna(cost):
                cost = 0.0
            data_row["Preco_Unit"] = cost
        except (ValueError, TypeError):
            data_row["Preco_Unit"] = 0.0

        df_protheus = pd.concat([df_protheus, pd.DataFrame(data=data_row, index=[0])], ignore_index=True)

    if not df_protheus.empty:
        df_protheus = df_protheus.fillna("")

    return df_protheus

# ------------------------------------------------------------------------------
class Model(RCMLModel):
    def load(self, artifacts):
        pass

    def predict(self, model_input, context):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        def load_data(model_input):
            try:
                token = Helpers.get_user_token(context)
                Requests.setToken(token)

                df_main_user = pd.DataFrame(model_input['df_main_user'])
                df_produto_fornecedor = pd.DataFrame(model_input['df_produto_fornecedor'])
                df_faturamento_minimo = pd.DataFrame(model_input['df_faturamento_minimo'])

                # CORREÇÃO: Garantir tipos corretos nas colunas numéricas
                if 'Final_order' in df_main_user.columns:
                    df_main_user['Final_order'] = pd.to_numeric(df_main_user['Final_order'], errors='coerce').fillna(0)
                if 'Cost' in df_main_user.columns:
                    df_main_user['Cost'] = pd.to_numeric(df_main_user['Cost'], errors='coerce').fillna(0)
                if 'LT' in df_main_user.columns:
                    df_main_user['LT'] = pd.to_numeric(df_main_user['LT'], errors='coerce').fillna(0)

                logger.info(f"Dados carregados - df_main_user: {df_main_user.shape}, df_produto_fornecedor: {df_produto_fornecedor.shape}, df_faturamento_minimo: {df_faturamento_minimo.shape}")

                return df_faturamento_minimo, df_main_user, df_produto_fornecedor

            except Exception as e:
                logger.error(f"Erro ao carregar dados: {e}")
                raise

        def get_protheus(df_main_user, df_produto_fornecedor, df_faturamento_minimo, export_zeros=True, arg_codcomp=35, arg_local=1):
            colunas_finais = ['Filial', 'Emissao', 'UnidReq', 'CodComp', 'Produto', 'Local',
                            'Quant', 'CC', 'DatPRF', 'DatEmbarque', 'Forn', 'Fabr',
                            'Moeda', 'Preco_Unit']

            warning_dict = {}
            try:
                # Calcula quais fornecedores precisam de aviso
                warning_dict = min_order_value_warning(df_main_user, df_faturamento_minimo)
                
                # CORREÇÃO: Separar fornecedores que PRECISAM e NÃO PRECISAM de aviso
                fornecedores_com_warning = [k for k, v in warning_dict.items() if v == True]  # NÃO atingiram mínimo
                fornecedores_sem_warning = [k for k, v in warning_dict.items() if v == False]  # ATINGIRAM mínimo
                
                logger.info(f"✅ Fornecedores que NÃO atingiram mínimo (warning=True): {fornecedores_com_warning}")
                logger.info(f"✅ Fornecedores que ATINGIRAM mínimo (warning=False): {fornecedores_sem_warning}")
                
            except Exception as e:
                logger.warning(f"Erro ao calcular avisos de faturamento mínimo: {e}")
                warning_dict = {}

            if 'Supp_Cod' not in df_main_user.columns:
                logger.error("Coluna 'Supp_Cod' não encontrada no DataFrame principal")
                return pd.DataFrame(columns=colunas_finais)

            # Seleciona fornecedores únicos
            suppliers_from_df = df_main_user['Supp_Cod'].dropna().unique().tolist()
            supplier_selector = list(set(list(warning_dict.keys()) + suppliers_from_df))

            # Remove valores inválidos
            supplier_selector = [s for s in supplier_selector if s != '-' and not pd.isna(s) and str(s).strip() != '']

            logger.info(f"Total de fornecedores para processar: {len(supplier_selector)}")

            lista_de_dataframes = []
            for supplier in supplier_selector:
                try:
                    df_resultado_individual = create_protheus_dataframe(
                        df_main_user.copy(),
                        df_produto_fornecedor,
                        supplier,
                        codcomp=str(arg_codcomp),
                        local=str(arg_local),
                        skip_zero=not export_zeros
                    )
                    if not df_resultado_individual.empty:
                        lista_de_dataframes.append(df_resultado_individual)
                        logger.debug(f"Fornecedor {supplier}: {len(df_resultado_individual)} linhas adicionadas")
                except Exception as e:
                    logger.warning(f"Erro ao processar fornecedor {supplier}: {e}")
                    continue

            if lista_de_dataframes:
                df_protheus = pd.concat(lista_de_dataframes, ignore_index=True)
                logger.info(f"DataFrame concatenado: {len(df_protheus)} linhas")
            else:
                df_protheus = pd.DataFrame(columns=colunas_finais)
                logger.warning("Nenhum DataFrame foi criado")

            if len(df_protheus) > 0:
                df_protheus.drop_duplicates(inplace=True)
                
                try:
                    # Recalcula o dicionário de warnings
                    warning_dict = min_order_value_warning(df_main_user, df_faturamento_minimo)
                    
                    # Aplica warning=True para fornecedores que NÃO atingiram mínimo
                    def check_warning(forn_code):
                        if pd.isna(forn_code) or str(forn_code).strip() == '':
                            return False
                            
                        forn_normalized = normalize_supplier_code(forn_code)
                        
                        # Debug para cada verificação
                        logger.info(f"🔍 DEBUG: Verificando warning para Forn='{forn_code}' normalizado='{forn_normalized}'")
                        
                        # Verifica em todos os formatos possíveis
                        for supplier_key, needs_warning in warning_dict.items():
                            supplier_key_normalized = normalize_supplier_code(supplier_key)
                            logger.info(f"🔍 DEBUG: Comparando '{forn_normalized}' == '{supplier_key_normalized}' (original: '{supplier_key}')")
                            
                            if forn_normalized == supplier_key_normalized:
                                logger.info(f"🔍 DEBUG: ✅ MATCH! Retornando warning={needs_warning}")
                                return needs_warning  # Retorna True se NÃO atingiu mínimo
                        
                        logger.info(f"🔍 DEBUG: ❌ Nenhum match encontrado, retornando False")
                        return False  # Default: não precisa de warning
                    
                    df_protheus['Min Order Value Warning'] = df_protheus['Forn'].apply(check_warning)
                    
                    # Log para verificação
                    warnings_true = df_protheus['Min Order Value Warning'].sum()
                    warnings_false = len(df_protheus) - warnings_true
                    logger.info(f"✅ Warnings aplicados - True: {warnings_true}, False: {warnings_false}")
                    
                    # Debug: Mostrar exemplos
                    if len(df_protheus) > 0:
                        sample = df_protheus[['Forn', 'Produto', 'Quant', 'Preco_Unit', 'Min Order Value Warning']].head(10)
                        logger.info(f"✅ Amostra dos dados:\n{sample}")
                    
                except Exception as e:
                    logger.warning(f"Erro ao adicionar coluna de aviso: {e}")
                    df_protheus['Min Order Value Warning'] = False

                logger.info(f"DataFrame final após remover duplicatas: {len(df_protheus)} linhas")

            df_protheus = df_protheus.fillna("")
            df_protheus['Emissao'] = datetime_to_string_aammdd(df_protheus['Emissao'])
            df_protheus['DatPRF'] = datetime_to_string_aammdd(df_protheus['DatPRF'])
            df_protheus['DatEmbarque'] = datetime_to_string_aammdd(df_protheus['DatEmbarque'])

            return df_protheus

        try:
            df_faturamento_minimo, df_main_user, df_produto_fornecedor = load_data(model_input)
            
            # Por padrão export_zeros=True para incluir mais dados
            df_protheus = get_protheus(df_main_user, df_produto_fornecedor, df_faturamento_minimo, export_zeros=True)

            result = {
                'df_protheus': df_protheus.to_dict('records') if not df_protheus.empty else [],
                'success': True,
                'total_records': len(df_protheus),
                'warnings_count': {
                    'with_warning': int(df_protheus['Min Order Value Warning'].sum()) if 'Min Order Value Warning' in df_protheus.columns else 0,
                    'without_warning': len(df_protheus) - int(df_protheus['Min Order Value Warning'].sum()) if 'Min Order Value Warning' in df_protheus.columns else len(df_protheus)
                }
            }
            normalized = normalize_structure(result)
            json.dumps(normalized)  # sanity check de serialização
            return normalized

        except Exception as e:
            logger.error(f"❌ Erro na execução: {e}")
            return {
                'error': str(e),
                'df_protheus': [],
                'success': False
            }

# ------------------------------------------------------------------------------

Helpers.save_output_rc_ml_model(
    context=context,
    model_name='ps_get_protheus1',
    model_obj=Model,
    artifacts={}
)