# --- Hot-patch for sqlite3 version issues in ChromaDB ---
# This must be at the very top of the file, before any other imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End of hot-patch ---

import streamlit as st
import json
import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_together import ChatTogether
from langchain_core.messages import SystemMessage, HumanMessage

# --- Import your modules ---
from csv_query_engine import load_csv_data, execute_sql_on_dfs
from populacao_rag import criar_documentos_de_conhecimento, criar_base_de_conhecimento_rag

# --- App Configuration ---
st.set_page_config(
    page_title="ChatBot do DPO - faça perguntas com base nos dados do One Trust - versão BETA",
    page_icon="🤖",
    layout="wide"
)

# --- LLM and Prompt Configuration ---
TOGETHER_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
PROMPT_SISTEMA = """
Você é um assistente especialista em traduzir perguntas em linguagem natural (português) para consultas SQL. 
Sua única função é gerar um código SQL funcional e otimizado com base no esquema das tabelas e no contexto fornecido. 
UTILIZE AS INFORMAÇÕES PRESENTES NO CONTEXTO PARA MONTAR A QUERY.
Siga estas regras estritamente:
1.  **Formato de Saída Obrigatório:** Sua única resposta deve ser um objeto JSON. O JSON deve conter duas chaves:
    * `"query"`: Uma string contendo o código SQL gerado.
    * `"descricao"`: Uma string com uma breve explicação em português do que o código SQL faz.
    * Exemplo de output: `{"query": "SELECT * FROM Clientes;", "descricao": "Este comando seleciona todas as colunas e registros da tabela 'Clientes'."}`
    NÃO ADICIONE MAIS NADA além do output definido.
2.  Use apenas as tabelas e colunas definidas no contexto. Os nomes das tabelas (`ot_consolidada`, `nx_org_group_classified_v2`) são os nomes a serem usados na query.
3.  **Ignore qualquer `Caminho` ou `Path` (ex: `iceberg.landing_trusted`) mencionado no contexto; use apenas o nome da tabela diretamente na query.**
4.  Analise a pergunta do usuário para identificar as colunas corretas, filtros (WHERE), agregações (COUNT, GROUP BY) e ordenações (ORDER BY).
5.  Se a pergunta for ambígua ou se for impossível gerar a consulta com o contexto fornecido, sua única resposta deve ser: `{"query": "ERRO: Impossível gerar a consulta.", "descricao": "A pergunta é ambígua ou não pode ser respondida com o contexto fornecido."}`
6.  **Buscas de Texto Robustas:** Gere consultas que sejam robustas a variações de digitação e capitalização. Para todas as cláusulas `WHERE` que filtram uma coluna de texto, aplique a função `LOWER()` à coluna e ao valor de busca. Exemplo: `LOWER(coluna) LIKE LOWER('%valor%')`.
"""
PROMPT_SISTEMA_INTERPRETADOR = """
Você vai receber um dataframe que foi convertido em markdown e a pergunta original que o usuário fez. 
Seu objetivo é repassar os itens que foram passados para você de uma forma mais limpa.
Leia esse dataframe ATENTAMENTE, SEM OMITIR DADOS, E SEM CRIAR DADOS FICTICIOS e responda a pergunta original do usuário com base nesse dataframe e somente nele

PONTOS IMPORTANTES:
1. O usuário precisa de todas as informações que vierem dentro do dataframe, não omita infromações.
2. Você deve responder a pergunta SOMENTE com os dados que foram fornecidos no dataframe, NÃO CRIE NEM SUPONHA NADA.
3. Responda SEMPRE em português do Brasil
4. Tome cuidado para não deixar passar informações importantes que estarão no prompt que você receber
"""

# --- Caching ---
@st.cache_resource
def inicializar_llm():
    try:
        api_key = st.secrets["TOGETHER_API_KEY"]
        llm = ChatTogether(model=TOGETHER_MODEL_NAME, temperature=0.0, together_api_key=api_key)
        return llm
    except Exception as e:
        st.error(f'Erro ao inicializar o modelo de linguagem da Together AI: {e}')
        st.warning("Verifique se a `TOGETHER_API_KEY` está configurada nos segredos do seu aplicativo Streamlit Cloud.")
        return None

@st.cache_resource
def inicializar_retriever(nome_diretorio_db="base_chroma_db"):
    try:
        st.write("Inicializando a base de conhecimento ChromaDB...")
        modelo_embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        db_vetorial = Chroma(persist_directory=nome_diretorio_db, embedding_function=modelo_embedding)
        retriever = db_vetorial.as_retriever(search_kwargs={"k": 3})
        st.success("Base de conhecimento pronta.")
        return retriever
    except Exception as e:
        st.error(f"Erro ao inicializar o retriever: {e}")
        return None

@st.cache_data
def carregar_dados_csv():
    """
    Localiza a pasta 'dados' de forma robusta e carrega os arquivos CSV.
    """
    # --- CORREÇÃO PRINCIPAL ---
    # Constrói o caminho absoluto para a pasta 'dados'
    # com base na localização do script atual (app.py)
    script_dir = os.path.dirname(__file__)
    folder_path = os.path.join(script_dir, "dados")
    
    dataframes, message = load_csv_data(folder_path)
    
    if dataframes is None:
        st.error(message) # Exibe a mensagem de erro detalhada do load_csv_data
        return None
    
    st.success(message) # Exibe a mensagem de sucesso
    return dataframes

# --- Main App Logic ---
def main():
    st.title("ChatBot do DPO - faça perguntas com base nos dados do One Trust - versão BETA")
    st.markdown("Faça uma pergunta em português sobre os dados dos arquivos CSV e o sistema irá gerar e executar uma consulta SQL para encontrar a resposta.")

    DB_DIR = "base_chroma_db"
    if not os.path.exists(DB_DIR):
        st.info("Base de conhecimento não encontrada. Criando pela primeira vez...")
        with st.spinner("Lendo o dicionário de dados e populando o ChromaDB..."):
            try:
                documentos = criar_documentos_de_conhecimento()
                criar_base_de_conhecimento_rag(documentos, nome_diretorio_db=DB_DIR)
                st.success("Base de conhecimento RAG criada com sucesso!")
            except Exception as e:
                st.error(f"Erro crítico ao criar a base de conhecimento RAG: {e}")
                return
    
    with st.sidebar:
        st.header("Sobre")
        st.markdown("Os dados carregados para base de conhecimento do chat, são as perguntas e respostas das Avaliações realizadas e que estão com o Status de 'Concluída' e 'Em Revisão'. Faça busca por Unidade do Sebrae, Unidade Organizacional, entre outras buscas possível. O foco do chat na versão BETA é fazer consultas básicas de quantidade de Avaliações.")
        

    # --- Initialization ---
    llm = inicializar_llm()
    retriever = inicializar_retriever(nome_diretorio_db=DB_DIR)
    dataframes = carregar_dados_csv()

    if not all([llm, retriever, dataframes]):
        st.warning("O sistema não está totalmente pronto. Verifique as mensagens de erro acima e a configuração.")
        return

    pergunta_usuario = st.text_input(
        "Qual informação você gostaria de consultar?",
        placeholder="Ex: Quantos formularios do tipo RAT existem em cada unidade organizacional no estado de São Paulo?"
    )

    if st.button("Gerar Resposta", type="primary") and pergunta_usuario:
        with st.spinner("Processando sua pergunta..."):
            
            # 1. Buscando Contexto (RAG)
            st.subheader("1. Buscando Contexto (RAG)")
            documentos_relevantes = retriever.invoke(pergunta_usuario)
            contexto_rag = "".join(f"---\nFonte: {doc.metadata.get('fonte', 'desconhecida')}\nConteúdo:\n{doc.page_content}\n" for doc in documentos_relevantes)
            with st.expander("Ver Contexto Encontrado"):
                st.text(contexto_rag)

            # 2. Gerando Consulta SQL com LLM
            st.subheader("2. Gerando Consulta SQL com LLM")
            prompt_final = f"Contexto das Tabelas:\n{contexto_rag}\n\nCom base SOMENTE no contexto acima, traduza a seguinte pergunta para SQL.\n\nPergunta do Usuário:\n{pergunta_usuario}"
            
            response = llm.invoke([SystemMessage(content=PROMPT_SISTEMA), HumanMessage(content=prompt_final)])
            resposta_json_str = response.content
            
            if not resposta_json_str:
                st.error("O modelo não retornou uma resposta.")
                return

            try:
                clean_response = resposta_json_str.strip().replace("```json", "").replace("```", "")
                resposta_obj = json.loads(clean_response)
                sql_gerado = resposta_obj.get("query")
                descricao = resposta_obj.get("descricao")
                
                st.code(sql_gerado, language='sql')
                st.info(f"**Descrição:** {descricao}")

                if not sql_gerado or "ERRO:" in sql_gerado:
                    st.error("O modelo não conseguiu gerar uma consulta SQL válida para a sua pergunta.")
                    return

            except (json.JSONDecodeError, AttributeError):
                st.error(f"O LLM retornou uma resposta em formato inválido. Resposta recebida:")
                st.code(resposta_json_str)
                return

            # 3. Executando Consulta nos Arquivos CSV
            st.subheader("3. Executando Consulta nos Arquivos CSV")
            if sql_gerado.strip().endswith(";"):
                sql_gerado = sql_gerado.strip()[:-1]

            df_resultado, mensagem = execute_sql_on_dfs(sql_gerado, dataframes)
            
            if df_resultado is not None and not df_resultado.empty:
                st.success(mensagem)
                st.dataframe(df_resultado)

                # 4. Interpretando os Resultados
                st.subheader("4. Interpretando os Resultados")
                with st.spinner("Gerando resposta final em linguagem natural..."):
                    dados_markdown = df_resultado.to_markdown()
                    interpretador_prompt = f"**PERGUNTA ORIGINAL DO USUÁRIO:**\n{pergunta_usuario}\n\n**DADOS DA CONSULTA:**\n{dados_markdown}\n\nCom base apenas nos dados acima, responda à pergunta original do usuário."
                    
                    resposta_final_obj = llm.invoke([SystemMessage(content=PROMPT_SISTEMA_INTERPRETADOR), HumanMessage(content=interpretador_prompt)])
                    st.markdown(resposta_final_obj.content)
            else:
                st.warning(f"A consulta não retornou resultados. Mensagem do sistema: {mensagem}")

if __name__ == "__main__":
    main()