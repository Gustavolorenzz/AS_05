import os
import tempfile
import pickle
from typing import List, Dict, Any
import PyPDF2
import google.generativeai as genai
import faiss
import numpy as np
import streamlit as st

# Classe simples para documentos (substituindo langchain)
class Document:
    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class TextSplitter:
    """Classe simples para dividir texto em chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Divide texto em chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Tentar quebrar em uma palavra
            chunk = text[start:end]
            last_space = chunk.rfind(' ')
            
            if last_space > 0:
                chunk = text[start:start + last_space]
                chunks.append(chunk)
                start = start + last_space + 1 - self.chunk_overlap
            else:
                chunks.append(chunk)
                start = end - self.chunk_overlap
            
            # Evitar overlap negativo
            if start < 0:
                start = 0
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Divide uma lista de documentos em chunks"""
        result = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={**doc.metadata, 'chunk_id': i}
                )
                result.append(new_doc)
        
        return result

class PDFProcessor:
    """Classe para processar documentos PDF"""
    
    def __init__(self):
        self.text_splitter = TextSplitter(
            chunk_size=1000,
            chunk_overlap=200
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
        try:
            # Tentar importar SentenceTransformer
            from sentence_transformers import SentenceTransformer
            import torch
            device = 'cpu'
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            self.use_sentence_transformers = True
        except Exception as e:
            st.warning(f"SentenceTransformer não disponível: {str(e)}. Usando embedding básico.")
            self.model = None
            self.use_sentence_transformers = False
        
        self.index = None
        self.documents = []
        self.embeddings = []
    
    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Cria embeddings para uma lista de documentos"""
        texts = [doc.page_content for doc in documents]
        
        if self.use_sentence_transformers and self.model is not None:
            try:
                embeddings = self.model.encode(texts)
                return embeddings
            except Exception as e:
                st.warning(f"Erro com SentenceTransformer: {str(e)}. Usando embedding básico.")
        
        # Usar embedding básico com TF-IDF
        return self._create_basic_embeddings(texts)
    
    def _create_basic_embeddings(self, texts: List[str]) -> np.ndarray:
        """Cria embeddings básicos usando TF-IDF como fallback"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        try:
            vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            embeddings = vectorizer.fit_transform(texts).toarray()
            self.vectorizer = vectorizer  # Salvar para uso na busca
            return embeddings.astype('float32')
        except Exception as e:
            st.error(f"Erro ao criar embeddings básicos: {str(e)}")
            # Último fallback: embeddings aleatórios normalizados
            num_docs = len(texts)
            embeddings = np.random.random((num_docs, 384)).astype('float32')
            # Normalizar
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
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
        
        if self.use_sentence_transformers:
            # Usar Inner Product para SentenceTransformer
            self.index = faiss.IndexFlatIP(dimension)
            # Normalizar embeddings para usar produto interno como similaridade de cosseno
            faiss.normalize_L2(embeddings)
        else:
            # Usar L2 para TF-IDF
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Busca documentos similares à query"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        try:
            # Criar embedding da query
            if self.use_sentence_transformers and self.model is not None:
                query_embedding = self.model.encode([query])
            else:
                # Usar TF-IDF para a query se disponível
                if hasattr(self, 'vectorizer'):
                    query_embedding = self.vectorizer.transform([query]).toarray()
                else:
                    # Fallback: usar embedding básico
                    query_embedding = self._create_basic_embeddings([query])
            
            # Normalizar embedding da query se necessário
            if self.use_sentence_transformers:
                faiss.normalize_L2(query_embedding.astype('float32'))
            
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
            
        except Exception as e:
            st.error(f"Erro na busca: {str(e)}")
            # Fallback: retornar primeiros documentos
            results = []
            for i, doc in enumerate(self.documents[:k]):
                results.append({
                    'document': doc,
                    'score': 0.5,  # Score neutro
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

# Imports adicionais para compatibilidade
import hashlib
import json

class SimpleVectorStore:
    """Versão simplificada do VectorStore para ambientes com limitações"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.index = None
        
    def create_simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Cria embeddings simples baseados em hash e contagem de palavras"""
        embeddings = []
        
        for text in texts:
            # Criar um vetor baseado em características do texto
            words = text.lower().split()
            
            # Características básicas
            features = [
                len(text),  # Tamanho do texto
                len(words),  # Número de palavras
                len(set(words)),  # Palavras únicas
                text.count('.'),  # Número de pontos
                text.count('?'),  # Número de perguntas
                text.count('!'),  # Número de exclamações
            ]
            
            # Adicionar hash das palavras mais comuns
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Apenas palavras com mais de 3 letras
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Top 10 palavras mais comuns
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for i in range(10):
                if i < len(top_words):
                    features.append(hash(top_words[i][0]) % 1000)
                else:
                    features.append(0)
            
            # Expandir para 100 features
            while len(features) < 100:
                features.append(0)
            
            embeddings.append(features[:100])
        
        return np.array(embeddings, dtype='float32')
    
    def build_index(self, documents: List[Document]):
        """Constrói índice simples"""
        if not documents:
            return
            
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.embeddings = self.create_simple_embeddings(texts)
        
        # Criar índice FAISS simples
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # Usar distância L2
        self.index.add(self.embeddings)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Busca simples"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Criar embedding da query
        query_embedding = self.create_simple_embeddings([query])
        
        # Buscar
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(1.0 / (1.0 + score)),  # Converter distância em similaridade
                    'rank': i + 1
                })
        
        return results