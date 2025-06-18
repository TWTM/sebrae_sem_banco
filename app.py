# --- Hot-patch for sqlite3 version issues in ChromaDB ---
# This must be at the very top of the file, before any other imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End of hot-patch ---

import streamlit as st
import json
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_together import ChatTogether
from langchain_core.messages import SystemMessage, HumanMessage

# Import your existing modules
from trino_connector import TrinoService
from populacao_rag import criar_documentos_de_conhecimento, criar_base_de_conhecimento_rag

# --- App Configuration ---
st.set_page_config(
    page_title="Natural Language to SQL",
    page_icon="🤖",
    layout="wide"
)

# --- LLM and Prompt Configuration ---
TOGETHER_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

PROMPT_SISTEMA = """
Você é um assistente especialista em traduzir perguntas em linguagem natural (português) para consultas SQL. 
Sua única função é gerar um código SQL funcional e otimizado com base no esquema do banco de dados e no contexto fornecido. 
UTILIZE AS INFORMAÇÕES PRESENTES NO CONTEXTO PARA MONTAR A QUERY
Siga estas regras estritamente:
1.  **Formato de Saída Obrigatório:** Sua única resposta deve ser um objeto JSON. O JSON deve conter duas chaves:
    * `"query"`: Uma string contendo o código SQL gerado.
    * `"descricao"`: Uma string com uma breve explicação em português do que o código SQL faz.
    * Exemplo de output: `{"query": "SELECT * FROM Clientes;", "descricao": "Este comando seleciona todas as colunas e registros da tabela 'Clientes'."}`
    NÃO ADICIONE MAIS NADA além do output definido. Não quero explicações nem observações adicionais. SOMENTE A QUERY seguida pela DESCRIÇÃO como está no formato acima
2.  Use apenas as tabelas e colunas definidas no contexto. Se uma coluna ou tabela não for mencionada no contexto, você não deve usá-la.
3.  Analise a pergunta do usuário para identificar as colunas corretas, filtros (WHERE), agregações (COUNT, GROUP BY) e ordenações (ORDER BY).
4.  Se a pergunta for ambígua ou se for impossível gerar a consulta com o contexto fornecido, sua única resposta deve ser: 'ERRO: Impossível gerar a consulta.'
5.  **Buscas de Texto Robustas:** Gere consultas que sejam robustas a variações de digitação e capitalização. Para todas as cláusulas `WHERE` que filtram uma coluna de texto, siga obrigatoriamente este padrão:
    * Aplique a função `LOWER()` à coluna do banco de dados.
    * Use o operador `LIKE`.
    * Aplique a função `LOWER()` ao valor de texto fornecido pelo usuário e envolva-o com os caracteres coringa `%`.
6.  Ao criar a consulta SQL adicione o caminho antes da tabela que será pesquisada.
"""
PROMPT_SISTEMA_INTERPRETADOR = """
Você vai receber um dataframe que foi convertido em markdown e a pergunta original que o usuário fez. 
Seu objetivo é repassar os itens que foram passados para você de uma forma mais limpa.
Leia esse dataframe ATENTAMENTE, SEM OMITIR DADOS, e responda a pergunta original do usuário com base nesse dataframe

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
        st.warning(f"Verifique se a pasta '{nome_diretorio_db}' existe. Se não, use o botão na barra lateral para populá-la.", icon="👈")
        return None

@st.cache_resource
def inicializar_trino_service():
    try:
        trino_creds = {
            "host": st.secrets["trino"]["host"],
            "port": st.secrets["trino"]["port"],
            "user": st.secrets["trino"]["user"],
            "catalog": st.secrets["trino"]["catalog"],
            "schema": st.secrets["trino"]["schema"],
            "http_scheme": st.secrets["trino"]["http_scheme"],
            "keycloak_url": st.secrets["keycloak"]["url"],
            "keycloak_grant_type": st.secrets["keycloak"]["grant_type"],
            "keycloak_client_id": st.secrets["keycloak"]["client_id"],
            "keycloak_client_secret": st.secrets["keycloak"]["client_secret"],
        }
        service = TrinoService(trino_creds)
        if service.connect():
             st.success("Conexão com o Trino estabelecida com sucesso.")
             return service
        else:
            st.error("Falha ao conectar com o Trino.")
            return None
    except KeyError as e:
        st.error(f'Não foi possível carregar as credenciais. A chave {e} não foi encontrada nos segredos do Streamlit Cloud.')
        st.info("Verifique se os nomes das chaves no seu painel de segredos correspondem exatamente aos esperados pelo código.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao configurar o serviço Trino: {e}")
        return None


# --- Main App Logic ---
def main():
    st.title("🤖 Tradutor de Linguagem Natural para SQL")
    st.markdown("Faça uma pergunta em português sobre os dados e o sistema irá gerar e executar uma consulta SQL para encontrar a resposta.")

    with st.sidebar:
        st.header("Configurações")
        st.markdown("Esta aplicação usa RAG para fornecer ao modelo o contexto do schema do seu banco de dados.")
        if st.button("Popular Base de Conhecimento RAG"):
            with st.spinner("Criando documentos e populando a base de dados ChromaDB..."):
                try:
                    documentos = criar_documentos_de_conhecimento()
                    criar_base_de_conhecimento_rag(documentos)
                    st.success("Base de conhecimento RAG populada com sucesso!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao popular a base de conhecimento: {e}")

    llm = inicializar_llm()
    retriever = inicializar_retriever()
    trino_service = inicializar_trino_service()

    if not all([llm, retriever, trino_service]):
        st.warning("O sistema não está pronto. Verifique as mensagens de erro acima e a configuração dos segredos.")
        return

    pergunta_usuario = st.text_input(
        "Qual informação você gostaria de consultar?",
        placeholder="Ex: Quantos formularios do tipo RAT existem em cada unidade organizacional no estado de São Paulo?"
    )

    if st.button("Gerar Resposta", type="primary") and pergunta_usuario:
        with st.spinner("Processando sua pergunta..."):
            st.subheader("1. Buscando Contexto (RAG)")
            documentos_relevantes = retriever.invoke(pergunta_usuario)
            contexto_rag = "".join(f"---\nFonte: {doc.metadata.get('fonte', 'desconhecida')}\nConteúdo:\n{doc.page_content}\n" for doc in documentos_relevantes)
            with st.expander("Ver Contexto Encontrado"):
                st.text(contexto_rag)

            st.subheader("2. Gerando Consulta SQL com LLM")
            prompt_final = f"Contexto do Banco de Dados:\n{contexto_rag}\n\nCom base SOMENTE no contexto acima, traduza a seguinte pergunta para SQL.\n\nPergunta do Usuário:\n{pergunta_usuario}"
            
            response = llm.invoke([SystemMessage(content=PROMPT_SISTEMA), HumanMessage(content=prompt_final)])
            resposta_json_str = response.content
            
            if not resposta_json_str:
                st.error("O modelo não retornou uma resposta.")
                return

            try:
                resposta_obj = json.loads(resposta_json_str)
                sql_gerado = resposta_obj.get("query")
                descricao = resposta_obj.get("descricao")
                
                st.code(sql_gerado, language='sql')
                st.info(f"**Descrição:** {descricao}")

                if not sql_gerado or sql_gerado == "ERRO":
                    st.error("O modelo não conseguiu gerar uma consulta SQL válida para a sua pergunta.")
                    return

            except json.JSONDecodeError:
                st.error(f"O LLM retornou uma resposta em formato inválido. Resposta recebida:")
                st.code(resposta_json_str)
                return

            st.subheader("3. Executando Consulta no Trino")
            if sql_gerado.strip().endswith(";"):
                sql_gerado = sql_gerado.strip()[:-1]

            df_resultado, mensagem = trino_service.execute_query(sql_gerado)
            
            if not df_resultado.empty:
                st.success(f"{mensagem}")
                st.dataframe(df_resultado)

                st.subheader("4. Interpretando os Resultados")
                with st.spinner("Gerando resposta final em linguagem natural..."):
                    dados_markdown = df_resultado.to_markdown()
                    interpretador_prompt = f"**PERGUNTA ORIGINAL DO USUÁRIO:**\n{pergunta_usuario}\n\n**DADOS DA CONSULTA:**\n{dados_markdown}\n\nCom base apenas nos dados acima, responda à pergunta original do usuário."
                    
                    resposta_final_obj = llm.invoke([SystemMessage(content=PROMPT_SISTEMA_INTERPRETADOR), HumanMessage(content=interpretador_prompt)])
                    st.markdown(resposta_final_obj.content)
            else:
                st.warning(f"A consulta foi executada, mas não retornou resultados. {mensagem}")


if __name__ == "__main__":
    main()
