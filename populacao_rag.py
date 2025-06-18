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

    forms_assessment_id (id): chave primária e compoem concatenado com informações uma url de acesso ao One Trust 
    forms_name (nome avaliação): Nome da Avaliação de Tratamento de Dados. 
    forms_status (status): Status daa Avaliação de Atividade de Dados 
    forms_template_name (tipo avaliacao): Tipo de Avaliação se referea versão macro que pode ser classificada também como ROPA ou RAT 
    forms_uf (uf): Unidade do Sebrae referente, NA se refere a Unidade Nacional do Sebrae 
    forms_org_grupo_name (sg_unidade): Sigla do Unidade Organizacional 
    hybrid_category (cadeia valor): Classificação de Cadeia de Valor 
    assessment_risk_level_name (risco_residual): Risco Residual pode ser classificados depois do tratamento de dados realizado ou não 
    inherent_risk_level_name (risco_inerente): Risco Inerente, risco atribuido depois da realização do preenchimento do usuário. 
    forms_create_dt (data criacao): Data de Criação do Avaliação de Atividade de Tratamento de Dados 
    forms_updated_dt (data atualizacao): Data de atualização do Avaliação de Atividade de Tratamento de Dados 
    last_updated (data ultima atualizacao): Data da Última atualização do avaliação de Atividade de Tratamento de Dados 
    submitted_on (data submissao): Data de Submissão da atividade de Tratamento de dados 
    completed_on (data_conclusao): Data de conclusão da Atividade de Tratamento de Dados 
    deadline (data_prazo): Data de prazo de conclusão do Avaliaçãode Atividades de Tratamento de Dados,campo não utilizado pelo Sebrae 
    end_date_year (ano fim): Métrica gerada conforme o Status, se a data de referência for COMPLETED ele tem como data referência completed_on se o status for UNDER_REVIEW a data de refência é last_updated trazendo assim o ano de referencia de data selecionada 
    end_date_month (mes fim): Métrica gerada conforme o Status, se a data de referência for COMPLETED ele tem como data referência completed_on se o status for UNDER_REVIEW a data de refência é last_updated trazendo assim o ano de referencia de data selecionada 
    end_date_day (dia fim): Métrica gerada conforme o Status, se a data de referência for COMPLETED ele tem como data referência completed_on se o status for UNDER_REVIEW a data de refência é last_updated trazendo assim o ano de referencia de data selecionada 
    end_date (data fim): Métrica gerada conforme o Status, se a data de referência for COMPLETED ele tem como data referência completed_on se o status for UNDER_REVIEW a data de refência é last_updated trazendo assim o ano de referencia de data selecionada 
    worked_days (dias trabalho): calculo em dias da diferença entre(end_date - submitted_on) 
    risco_residual_numerico (risco residual numerico): tranformação da assessment risk level name em dado numérico diante da variável nominal ordinário é transformado em escala de 1 de nulas e escala de grandeza. Com objetivo de entender concentração de risco diante de medidas de disperção, posição e assimetria e curtose. 
    risco_inerente_numerico (risco inerente numerico): tranformação da risco_inerente_numerico em dado numérico diante da variável nominal ordinário é transformado em escala de 1 de nulas e escala de grandeza. Com objetivo de entender concentração de risco diante de medidas de disperção, posição e assimetria e curtose. 
    primary_record_number (nm inventario): número do inventário de dados relacionado à atividade de tratamento de dados 
    primary_record_name (nome_inventario): nome do invetário de dados relacionado à atividade de tratamento de dados 
    forms_number (numero atividade): número da atividade de tratamento de dados 
    inventory_processing_activities_id (): id do inventário é se relacionada de maneira aglutinas todas as avaliação de tratamento de dados, testes de balanceamento, relatórios de impacto a proteção de dados, análises do DPO, sobre um respectiva atividade, que pode ser recorrente ou não. Exemplo: Folha de Pagamento do Sebrae/PR, formulário para capção de dados de ujm evento específico. 
    inventory_processing_activities_name (): Nome do inventário é se relacionada de maneira aglutinas todas as avaliação de tratamento de dados, testes de balanceamento, relatórios de impacto a proteção de dados, análises do DPO, sobre um respectiva atividade, que pode ser recorrente ou não. Exemplo: Folha de Pagamento do Sebrae/PR, formulário para capção de dados de ujm evento específico. 
    flag_ropa_rat (): Classifica os forms_template_name por versões, ROPA é uma versão antiga já descontinuada e RAT éa versão mais atual das Avaliações de Tratamento de Dados. 
    mitigacao_risco (mitigacao_risco): Descreve o comportamento do risco, comparando o risco inerente (inherent_risk_level_name) e o risco residual (assessment_risk_level_name) com isso é possível identificar a mitigação ou o aumento do risco diante da variável stage_name="monitoramento", somente neste estágio é feito o tratamento do risco. O risco inerente é identificado no início do processo e o risco residual é definido na fase final. Só podendo afirmar sobre mitigação ou aumento do risco diante do estágio de monitoramento. 
    tratamento_risco (comportamento_risco): comparação simples entre o risco inerente () e o risco residual () 
    stage_name (estagio_nome): Estágio do Tratamento do Risco, somente no estágio de Monitoramento que é realizado o tratamento do risco 
    description_org (descricao_unidade): Descrição da Unidade Organizacional
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
        modelo_embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

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