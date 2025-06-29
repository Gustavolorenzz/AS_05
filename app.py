import streamlit as st
import os
import tempfile
from typing import List
from tools import PDFProcessor, VectorStore, SimpleVectorStore, GeminiChat, format_context_from_results
from dotenv import load_dotenv

API_KEY = os.environ.get('GOOGLE_API_KEY')

# Configuração da página
st.set_page_config(
    page_title="Assistente Conversacional PDF",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state
if 'vector_store' not in st.session_state:
    try:
        st.session_state.vector_store = VectorStore()
        st.sidebar.success("🔬 Sistema de embeddings avançado carregado")
    except Exception as e:
        st.sidebar.warning("⚠️ Usando sistema de busca simplificado devido a limitações do ambiente")
        try:
            st.session_state.vector_store = SimpleVectorStore()
        except Exception as e2:
            st.error(f"Erro crítico ao inicializar o sistema: {str(e2)}")
            st.stop()
        
if 'pdf_processor' not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'gemini_chat' not in st.session_state:
    st.session_state.gemini_chat = None

# Título principal
st.title("AS05")
st.markdown("### Faça perguntas sobre seus documentos PDF usando IA")

# Sidebar
st.sidebar.markdown("### [🔗 Link para o repositório](https://drive.google.com/drive/folders/10tXY6ERto4np60fJPjb30YfEuTxqq8WW?usp=sharing)")
st.sidebar.title("⚙️ Configurações")
st.sidebar.markdown("---")

# Configuração da API do Gemini
st.sidebar.subheader("API do Google Gemini")
st.sidebar.info("💡 Usando o modelo gratuito Gemini 1.5 Flash")
# Load API key from .env.example file

# Try to load API key from .env.example file
api_key = os.environ.get('API_KEY_GOOGLE')
env_api_key = None
if not api_key:
    if os.path.exists('.env.example'):
        load_dotenv('.env.example')
        env_api_key = os.getenv('GOOGLE_API_KEY')

    # Use the API key from the file or fall back to text input
    if env_api_key:
        st.sidebar.success("✅ API Key carregada do arquivo .env.example")
        api_key = env_api_key
    else:
        api_key = st.sidebar.text_input(
            "Digite sua API Key do Google Gemini:",
            type="password",
            help="Obtenha sua API key GRATUITA em: https://makersuite.google.com/app/apikey"
        )

if api_key:
    try:
        st.session_state.gemini_chat = GeminiChat(api_key)
        st.sidebar.success("✅ API key configurada!")
    except Exception as e:
        st.sidebar.error(f"❌ Erro na API key: {str(e)}")

st.sidebar.markdown("---")

# Upload de arquivos PDF
st.sidebar.subheader("📁 Upload de Documentos")
uploaded_files = st.sidebar.file_uploader(
    "Escolha arquivos PDF:",
    type=['pdf'],
    accept_multiple_files=True,
    help="Carregue um ou mais arquivos PDF para indexação"
)

# Botão para processar documentos
if st.sidebar.button("🚀 Processar Documentos", disabled=not uploaded_files):
    if not uploaded_files:
        st.sidebar.warning("⚠️ Selecione pelo menos um arquivo PDF")
    else:
        with st.spinner("📖 Processando documentos..."):
            all_documents = []
            
            # Processar cada arquivo PDF
            for uploaded_file in uploaded_files:
                st.sidebar.info(f"Processando: {uploaded_file.name}")
                documents = st.session_state.pdf_processor.process_pdf_file(uploaded_file)
                all_documents.extend(documents)
            
            if all_documents:
                # Construir índice de vetores
                st.sidebar.info("🔍 Criando índice de embeddings...")
                st.session_state.vector_store.build_index(all_documents)
                st.session_state.documents_loaded = True
                
                st.sidebar.success(f"✅ {len(all_documents)} chunks processados com sucesso!")
                
                # Salvar índice para uso futuro
                try:
                    st.session_state.vector_store.save_index("vector_index.pkl")
                    st.sidebar.info("💾 Índice salvo localmente")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Erro ao salvar índice: {str(e)}")
            else:
                st.sidebar.error("❌ Erro ao processar documentos")


# Interface principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")
    
    # Verificar se tudo está configurado
    if not api_key:
        st.warning("⚠️ Configure sua API key do Google Gemini na barra lateral")
    elif not st.session_state.documents_loaded:
        st.info("📄 Carregue e processe documentos PDF para começar a fazer perguntas")
    else:
        # Interface de chat
        chat_container = st.container()
        
        # Mostrar histórico de chat
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"**🙋 Você:** {question}")
                st.markdown(f"**🤖 Assistente:** {answer}")
                st.markdown("---")
        
        # Input para nova pergunta
        question = st.text_input(
            "Faça uma pergunta sobre seus documentos:",
            key="question_input",
            placeholder="Ex: Qual é o tema principal dos documentos?"
        )
        
        col_ask, col_clear = st.columns([1, 1])
        
        with col_ask:
            if st.button("❓ Perguntar", disabled=not question):
                if question:
                    with st.spinner("🤔 Pensando..."):
                        # Buscar documentos relevantes
                        search_results = st.session_state.vector_store.search(question, k=5)
                        
                        # Formatar contexto
                        context = format_context_from_results(search_results)
                        
                        # Gerar resposta
                        response = st.session_state.gemini_chat.generate_response(context, question)
                        
                        # Adicionar ao histórico
                        st.session_state.chat_history.append((question, response))
                        
                        # Recarregar a página para mostrar a nova resposta
                        st.rerun()
        
        with col_clear:
            if st.button("🗑️ Limpar Chat"):
                st.session_state.chat_history = []
                st.rerun()

with col2:
    st.subheader("📊 Informações")
    
    # Status do sistema
    st.markdown("**Status do Sistema:**")
    if api_key:
        st.success("✅ API Gemini configurada")
    else:
        st.error("❌ API Gemini não configurada")
    
    if st.session_state.documents_loaded:
        st.success("✅ Documentos carregados")
        num_docs = len(st.session_state.vector_store.documents)
        st.info(f"📄 {num_docs} chunks indexados")
    else:
        st.error("❌ Nenhum documento carregado")
    
    st.markdown("---")
    
    # Estatísticas
    st.markdown("**📈 Estatísticas:**")
    st.metric("Perguntas feitas", len(st.session_state.chat_history))
    
    if st.session_state.documents_loaded:
        st.metric("Documentos indexados", len(st.session_state.vector_store.documents))
    
    st.markdown("---")
    
    # Ajuda
    st.markdown("**❓ Como usar:**")
    st.markdown("""
    1. Configure sua API key do Google Gemini
    2. Faça upload de arquivos PDF
    3. Clique em "Processar Documentos"
    4. Faça perguntas sobre o conteúdo
    """)
    
    st.markdown("**💡 Dicas:**")
    st.markdown("""
    - Use perguntas específicas para melhores respostas
    - O sistema busca nos documentos mais relevantes
    - Você pode carregar múltiplos PDFs
    - Modelo gratuito tem limite de uso diário
    """)
    
    st.markdown("**⚠️ Plano Gratuito:**")
    st.markdown("""
    - Limite de requisições por minuto
    - Se atingir o limite, aguarde alguns minutos
    - API key gratuita disponível no Google AI Studio
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Desenvolvido com ❤️ usando Streamlit e Google Gemini"
    "</div>",
    unsafe_allow_html=True
)