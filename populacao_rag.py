from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

def criar_documentos_de_conhecimento():
    """
    Aqui definimos todo o conhecimento que o RAG terá.
    Os placeholders das descrições das tabelas foram preenchidos com
    as informações dos arquivos PDF.
    """
    print("Criando os documentos de conhecimento com o dicionário de dados...")

    # -- Documento 1: Schema da tabela ot_consolidada --
    schema_ot_consolidada = """
    Caminho: iceberg.landing_trusted
    Nome da Tabela: ot_consolidada
    Descrição: Esta tabela possui todas as perguntas e respostas detalhadas das atividades de tratamento de dados.

    Colunas:
    - assessment_id (ID): Chave primária que pode ser usada para apontar para a atividade no OneTrust. 
    - forms_number (Número da Atividade de Tratamento de dados): Número de série da atividade.
    - forms_org_grupo_name (Departamentos): Sigla do departamento que realiza a coleta. 
    - forms_uf (Unidades Sebrae): Unidade da Federação (UF) ou 'NA' para Unidade Nacional.
    - tema_name (tema): Tema relacionado a princípios da LGPD. 
    - subtema_name (subtema): Subtema relacionado a princípios da LGPD. 
    - topico_name (topico): Tópico relacionado a princípios da LGPD. 
    - forms_template_name (Tipo de Versão da Atividade de Tratamento de Dados): Versão do formulário (ex: ROPA, RAT).
    - forms_name (Nome da Atividade de Tratamento de Dados): Nome da atividade.
    - forms_template_version (Número da Versão): Versão numérica do formulário.
    - data_de_criacao (data de criação): Data de criação da atividade.
    - forms_status (Status): Status em que a atividade se encontra (ex: 'COMPLETED').
    - submitted_on (Data Submissão): Data de submissão do formulário.
    - completed_on (Data Conclusão): Data em que o formulário foi completado (status 'COMPLETED').
    - section_questions_risk_level (nivel de risco): Perguntas relacionadas a riscos.
    - section_questions_risk_score (pontuacao_de_risco): Score de risco da resposta (>= 1 indica risco).
    - section_questions_risk_probability (probabilidade risco): Probabilidade do risco ocorrer.
    - section_questions_risk_impact_level: Impacto caso o risco ocorra.
    """

    # -- Documento 2: Schema da tabela nx_org_group_classified_v2 --
    schema_nx_org_group_classified = """
    Caminho: iceberg.landing_trusted
    Nome da Tabela: nx_org_group_classified_v2
    Descrição: Esta tabela consolida informações macro de todos os formulários e suas classificações organizacionais.

    Colunas:
    - forms_assessment_id (id): Chave primária.
    - forms_status (status): Status da avaliação da atividade.
    - forms_template_name (tipo formulario): Tipo de avaliação, como ROPA ou RAT.
    - forms_uf (uf): Unidade do Sebrae ('NA' para Nacional).
    - forms_org_grupo_name (sg_departamento): Sigla do Departamento.
    - hybrid_category (cadeia valor): Classificação de Cadeia de Valor.
    - assessment_risk_level_name (risco_residual): Risco Residual após tratamento dos dados.
    - inherent_risk_level_name (risco inerente): Risco inerente atribuído após preenchimento.
    - forms_create_dt (data criacao): Data de Criação do formulário.
    - forms_updated_dt (data atualizacao): Data de atualização do formulário.
    - last_updated (data_ultima_atualizacao): Data da última atualização.
    - submitted_on (data submissao): Data de submissão.
    - end_date (data fim): Data fim da atividade.
    - worked_days (dias trabalho): Diferença em dias entre end_date e submitted_on.
    - risco_residual_numerico: Risco residual em escala numérica.
    - risco_inerente_numerico: Risco inerente em escala numérica.
    - primary_record_number (nm_inventario): Número do inventário de dados.
    - primary_record_name (nome inventario): Nome do inventário de dados.
    """

    documentos = [
        Document(
            page_content=schema_ot_consolidada,
            metadata={"fonte": "schema_ot_consolidada"}
        ),
        Document(
            page_content=schema_nx_org_group_classified,
            metadata={"fonte": "schema_nx_org_group_classified_v2"}
        ),
        # Document(
        #    page_content="PLACEHOLDER: Pergunta e SQL exemplo para contagem e agregação (GROUP BY, COUNT, ORDER BY).",
        #    metadata={"fonte": "exemplo_sql_agregacao"}
        #),
        #Document(
        #    page_content="PLACEHOLDER: Pergunta e SQL exemplo para a lógica de 'testes de balanceamento' (COUNT DISTINCT, OR).",
        #    metadata={"fonte": "exemplo_sql_balanceamento"}
        #),
        #Document(
        #    page_content="PLACEHOLDER: Pergunta e SQL exemplo para buscas em texto com 'LIKE'.",
        #    metadata={"fonte": "exemplo_sql_like"}
        #),
        #Document(
        #    page_content="PLACEHOLDER: Exemplo de como fazer um JOIN entre as duas tabelas.",
        #    metadata={"fonte": "exemplo_sql_join"}
        #
    ]
    print(f"{len(documentos)} documentos criados.")
    return documentos

def criar_base_de_conhecimento_rag(documentos, nome_diretorio_db="base_chroma_db"):
    """
    Função atualizada para usar ChromaDB.
    """
    try:
        print("Inicializando o modelo de embedding...")
        modelo_embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

        print(f"Criando e persistindo o banco de dados Chroma na pasta '{nome_diretorio_db}'...")
        # O ChromaDB cria e persiste os dados no diretório especificado de uma só vez.
        db_vetorial = Chroma.from_documents(
            documents=documentos,
            embedding=modelo_embedding,
            persist_directory=nome_diretorio_db  # <--- MUDANÇA AQUI
        )
        
        print("\nBase de conhecimento criada com ChromaDB e salva com sucesso!")
        return db_vetorial

    except Exception as e:
        print(f"Ocorreu um erro ao criar a base de conhecimento com ChromaDB: {e}")
        return None

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    documentos_base = criar_documentos_de_conhecimento()
    if documentos_base:
        criar_base_de_conhecimento_rag(documentos_base)