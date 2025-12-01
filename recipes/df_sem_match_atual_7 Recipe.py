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
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Your code goes here
col_names = ["component", "origem", "motivo"]

df_sem_match1 = Helpers.getEntityData(context, 'df_sem_match_atual_1')
df_sem_match1.columns = col_names

df_sem_match2 = Helpers.getEntityData(context, 'df_sem_match_atual_2')
df_sem_match2.columns = col_names

df_sem_match3 = Helpers.getEntityData(context, 'df_sem_match_atual_3')
df_sem_match3.columns = col_names

df_sem_match4 = Helpers.getEntityData(context, 'df_sem_match_atual_4')
df_sem_match4.columns = col_names

df_sem_match5 = Helpers.getEntityData(context, 'df_sem_match_atual_5')
df_sem_match5.columns = col_names

df_sem_match6 = Helpers.getEntityData(context, 'df_sem_match_atual_6')
df_sem_match6.columns = col_names

df_sem_match7 = Helpers.getEntityData(context, 'df_sem_match_atual_7')
df_sem_match7.columns = col_names

df_sem_match = pd.concat([
    df_sem_match1,
    df_sem_match2,
    df_sem_match3,
    df_sem_match4,
    df_sem_match5,
    df_sem_match6,
    df_sem_match7
], ignore_index=True)
df_sem_match = df_sem_match.drop_duplicates().reset_index(drop=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
remover_motivos = ['Component não começa com T seguido de dígitos', 'Component com dois . em seu nome',
                   'Linha de rodape/invalida']
df_sem_match = df_sem_match[~df_sem_match['motivo'].isin(remover_motivos)]           
# df_sem_match[df_sem_match['motivo'].str.startswith('T')]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
Helpers.save_output_dataset(context=context, output_name='df_sem_match', data_frame=df_sem_match)