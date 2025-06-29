import os
import tempfile
import pickle
from typing import List, Dict, Any
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st

class PDFProcessor:
    """Classe para processar documentos PDF"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extrai texto de um arquivo PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Erro ao extrair texto do PDF: {str(e)}")
            return ""
    
    def process_pdf_file(self, pdf_file) -> List[Document]:
        """Processa um arquivo PDF e retorna chunks de documentos"""
        text = self.extract_text_from_pdf(pdf_file)
        if not text:
            return []
        
        # Criar documento
        doc = Document(
            page_content=text,
            metadata={"source": pdf_file.name}
        )
        
        # Dividir em chunks
        chunks = self.text_splitter.split_documents([doc])
        return chunks

class VectorStore:
    """Classe para gerenciar embeddings e busca vetorial"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        self.embeddings = []
    
    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Cria embeddings para uma lista de documentos"""
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts)
        return embeddings
    
    def build_index(self, documents: List[Document]):
        """Constrói o índice FAISS com os documentos"""
        if not documents:
            return
        
        self.documents = documents
        embeddings = self.create_embeddings(documents)
        self.embeddings = embeddings
        
        # Criar índice FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product para similaridade de cosseno
        
        # Normalizar embeddings para usar produto interno como similaridade de cosseno
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Busca documentos similares à query"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Criar embedding da query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Buscar documentos similares
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def save_index(self, filepath: str):
        """Salva o índice e documentos em arquivo"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Salvar índice FAISS separadamente
        if self.index is not None:
            faiss.write_index(self.index, filepath.replace('.pkl', '.index'))
    
    def load_index(self, filepath: str):
        """Carrega o índice e documentos de arquivo"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            
            # Carregar índice FAISS
            index_path = filepath.replace('.pkl', '.index')
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                return True
        except Exception as e:
            st.error(f"Erro ao carregar índice: {str(e)}")
        
        return False

class GeminiChat:
    """Classe para interação com o modelo Gemini"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Usando o modelo gratuito gemini-1.5-flash
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_response(self, context: str, question: str) -> str:
        """Gera resposta baseada no contexto e pergunta"""
        # Prompt mais conciso para o modelo gratuito
        prompt = f"""Contexto: {context[:2000]}

Pergunta: {question}

Responda com base apenas no contexto fornecido. Se não houver informação suficiente, diga que não foi possível encontrar a resposta nos documentos."""
        
        try:
            # Configuração para o modelo gratuito
            generation_config = {
                "temperature": 0.3,
                "max_output_tokens": 1000,
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            # Tratamento de erros específicos do modelo gratuito
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                return "⚠️ Limite de uso da API atingido. Tente novamente em alguns minutos."
            elif "safety" in str(e).lower():
                return "⚠️ Resposta bloqueada por questões de segurança. Tente reformular sua pergunta."
            else:
                return f"Erro ao gerar resposta: {str(e)}"

def format_context_from_results(results: List[Dict[str, Any]]) -> str:
    """Formata os resultados da busca em contexto para o LLM"""
    if not results:
        return "Nenhum documento relevante encontrado."
    
    context_parts = []
    for i, result in enumerate(results[:3], 1):  # Usar apenas os 3 mais relevantes
        doc = result['document']
        source = doc.metadata.get('source', 'Desconhecido')
        content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        
        context_parts.append(f"Documento {i} (Fonte: {source}):\n{content}\n")
    
    return "\n".join(context_parts)

def save_uploaded_file(uploaded_file) -> str:
    """Salva arquivo carregado temporariamente e retorna o caminho"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Erro ao salvar arquivo: {str(e)}")
        return None