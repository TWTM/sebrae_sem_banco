import os
import pandas as pd
from pandasql import sqldf

def load_csv_data(folder_path: str):
    """
    Carrega todos os arquivos .csv de uma pasta específica em um dicionário de DataFrames.
    As chaves do dicionário são os nomes dos arquivos sem a extensão .csv.

    Args:
        folder_path (str): O caminho absoluto para a pasta contendo os arquivos CSV.

    Returns:
        Um dicionário no formato {'nome_da_tabela': DataFrame}.
    """
    dataframes = {}
    if not os.path.isdir(folder_path):
        # Esta verificação agora retorna um dicionário vazio e uma mensagem de erro clara.
        return None, f"Erro: O diretório especificado não foi encontrado: '{folder_path}'"

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            table_name = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            try:
                dataframes[table_name] = pd.read_csv(file_path)
            except Exception as e:
                # Se um arquivo específico falhar, retornamos o erro.
                return None, f"Erro ao carregar o arquivo {filename}: {e}"
    
    if not dataframes:
        return None, f"Nenhum arquivo CSV foi encontrado na pasta '{folder_path}'."

    return dataframes, f"Tabelas carregadas com sucesso: {list(dataframes.keys())}"


def execute_sql_on_dfs(query: str, dataframes: dict):
    """
    Executa uma consulta SQL em um dicionário de DataFrames Pandas.

    Args:
        query (str): A consulta SQL a ser executada.
        dataframes (dict): Um dicionário no formato {'nome_da_tabela': DataFrame}.

    Returns:
        Uma tupla (result_df, message).
        result_df é um DataFrame com os resultados ou None em caso de erro.
        message é uma string de status.
    """
    if not dataframes:
        return None, "Erro: Não há tabelas (DataFrames) para consultar."

    # A função lambda torna os dataframes no dicionário acessíveis para o pandasql
    pysqldf = lambda q: sqldf(q, dataframes)

    try:
        result_df = pysqldf(query)
        message = f"Consulta executada com sucesso. Foram encontrados {len(result_df)} registros."
        return result_df, message
    except Exception as e:
        error_message = f"Erro ao executar a consulta SQL: {e}"
        return None, error_message