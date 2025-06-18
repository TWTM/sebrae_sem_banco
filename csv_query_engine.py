import os
import pandas as pd
from pandasql import sqldf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv_data(folder_path='data'):
    """
    Scans a folder for .csv files and loads them into a dictionary of Pandas DataFrames.
    The dictionary keys are the filenames without the .csv extension, which will
    serve as table names.

    Returns:
        A dictionary where { 'table_name': DataFrame }.
    """
    logger.info(f"Searching for CSV files in folder: '{folder_path}'")
    dataframes = {}
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                table_name = os.path.splitext(filename)[0]
                file_path = os.path.join(folder_path, filename)
                try:
                    dataframes[table_name] = pd.read_csv(file_path)
                    logger.info(f"Successfully loaded '{filename}' as table '{table_name}'.")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
    except FileNotFoundError:
        logger.error(f"The specified folder '{folder_path}' was not found.")
        return None
    
    if not dataframes:
        logger.warning("No CSV files were found or loaded.")
        
    return dataframes

def execute_sql_on_dfs(query: str, dataframes: dict):
    """
    Executes an SQL query on a dictionary of Pandas DataFrames.

    Args:
        query (str): The SQL query to execute.
        dataframes (dict): A dictionary of {'table_name': DataFrame}.

    Returns:
        A tuple containing (result_df, message).
        result_df is a DataFrame with the query results.
        message is a string indicating success or failure.
    """
    if not dataframes:
        return pd.DataFrame(), "Error: No dataframes available to query."

    # This makes the DataFrames in the dictionary available to pandasql by their key name.
    # For example, dataframes['formularios'] can be queried as `SELECT * FROM formularios;`
    pysqldf = lambda q: sqldf(q, dataframes)

    try:
        logger.info(f"Executing SQL query: {query}")
        result_df = pysqldf(query)
        message = f"Consulta executada com sucesso. Foram encontrados {len(result_df)} registros."
        return result_df, message
    except Exception as e:
        error_message = f"Erro ao executar a consulta SQL: {e}"
        logger.error(error_message)
        return pd.DataFrame(), error_message