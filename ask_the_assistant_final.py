# -*- coding: utf-8 -*-

import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

# Pastas onde estão os seus ficheiros de texto.
THEMATIC_DATA_FOLDER = './processed_data' # Para os ficheiros de temas
SUMMARIES_FOLDER = './summaries'         # Para os resumos dos programas completos

# Modelo de embeddings.
EMBEDDING_MODEL_NAME = 'distiluse-base-multilingual-cased-v1'

# ==============================================================================
# CLASSE PARA GERIR A BASE DE CONHECIMENTO
# ==============================================================================

class KnowledgeBase:
    """
    Esta classe gere a base de conhecimento. Agora, ela carrega dados de duas fontes:
    1. Ficheiros temáticos detalhados.
    2. Resumos gerais dos programas eleitorais.
    Isto torna o assistente mais completo.
    """
    def __init__(self, thematic_folder, summaries_folder, model_name):
        print("-> A inicializar a Base de Conhecimento...")
        self.thematic_folder = thematic_folder
        self.summaries_folder = summaries_folder
        self.embedding_model = SentenceTransformer(model_name)
        
        self.chunks = []
        self.metadata = []
        self.index = None
        
        self.build()

    def _process_thematic_documents(self):
        """
        Lê os ficheiros temáticos, divide-os por partido e adiciona à base de conhecimento.
        """
        print("-> A processar documentos temáticos...")
        party_pattern = re.compile(r"^\s*([A-ZÇÃ][a-zA-ZçÇãéáíóúâêô\s–-]+)\s*$", re.MULTILINE)
        
        if not os.path.exists(self.thematic_folder):
            print(f"AVISO: A pasta de temas '{self.thematic_folder}' não foi encontrada. A ignorar.")
            return

        for filename in os.listdir(self.thematic_folder):
            if 'tema' in filename:
                file_path = os.path.join(self.thematic_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                parts = party_pattern.split(content)
                
                for i in range(1, len(parts), 2):
                    party_name = parts[i].strip()
                    party_proposals_text = parts[i+1].strip()
                    
                    if not party_proposals_text:
                        continue

                    # Adiciona cada proposta/bloco do partido como um chunk separado.
                    chunk_text = f"Sobre o tema '{filename.split('tema_')[1].split('.')[0]}', a proposta de {party_name} é: {party_proposals_text}"
                    self.chunks.append(chunk_text)
                    self.metadata.append({
                        'source_file': filename,
                        'source_type': 'documento_tematico',
                        'party': party_name
                    })

    def _process_summary_documents(self):
        """
        Lê os ficheiros de resumos dos programas e adiciona-os à base de conhecimento.
        """
        print("-> A processar resumos dos programas completos...")
        if not os.path.exists(self.summaries_folder):
            print(f"AVISO: A pasta de resumos '{self.summaries_folder}' não foi encontrada. A ignorar.")
            return
            
        for filename in os.listdir(self.summaries_folder):
            if filename.startswith('resumo_'):
                file_path = os.path.join(self.summaries_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    summary_content = f.read()

                # Tenta extrair o nome do partido do nome do ficheiro.
                # Ex: "resumo_programa_eleitoral_PS_legislativas_2025.txt" -> "PS"
                try:
                    party_name = filename.split('_')[3]
                except IndexError:
                    party_name = "Desconhecido" # fallback

                chunk_text = f"Resumo geral do programa de {party_name}: {summary_content}"
                self.chunks.append(chunk_text)
                self.metadata.append({
                    'source_file': filename,
                    'source_type': 'resumo_geral',
                    'party': party_name
                })

    def build(self):
        """
        Constrói o índice vetorial FAISS a partir de TODAS as fontes de dados.
        """
        # Carrega os chunks de ambas as fontes.
        self._process_thematic_documents()
        self._process_summary_documents()
        
        if not self.chunks:
            print("ERRO: Nenhum documento foi carregado. A base de conhecimento está vazia. Verifique as pastas.")
            return

        print(f"\n-> Total de {len(self.chunks)} chunks de informação de todas as fontes.")
        print("-> A gerar embeddings (vetores) para cada chunk de texto...")
        
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(np.array(embeddings, dtype=np.float32))
        
        print(f"-> Base de conhecimento construída com sucesso. Índice tem {self.index.ntotal} vetores.")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Pesquisa os 'k' chunks mais relevantes para uma dada query.
        """
        if not self.index:
            print("ERRO: O índice não foi construído.")
            return []
            
        print(f"\n-> A pesquisar por: '{query}'")
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        results = []
        for i in indices[0]:
            if i != -1:
                results.append({
                    'text': self.chunks[i],
                    'metadata': self.metadata[i]
                })
                
        print(f"-> Encontrados {len(results)} documentos relevantes das seguintes fontes:")
        for doc in results:
            print(f"   - {doc['metadata']['source_file']} (Tipo: {doc['metadata']['source_type']})")
            
        return results

# ==============================================================================
# FUNÇÃO PRINCIPAL DO ASSISTENTE
# ==============================================================================

def main():
    """
    Função principal que inicia a base de conhecimento e entra num loop
    para responder às perguntas do utilizador.
    """
    kb = KnowledgeBase(THEMATIC_DATA_FOLDER, SUMMARIES_FOLDER, EMBEDDING_MODEL_NAME)
    
    if not kb.index:
        print("\nO assistente não pode iniciar porque a base de conhecimento está vazia. Termine o programa e verifique os seus ficheiros.")
        return

    print("\n-> A carregar modelo de Geração de Respostas...")
    generator = pipeline('text2text-generation', model='google/flan-t5-base')
    print("\n=========================================================")
    print("      Assistente LLM de Política Portuguesa ATIVO")
    print("=========================================================")
    print("Pode fazer perguntas sobre os temas ou pedir resumos.")
    print("Escreva 'sair' para terminar.")
    
    while True:
        user_query = input("\n[Você]: ")
        if user_query.lower() == 'sair':
            print("Até à próxima!")
            break
            
        relevant_docs = kb.search(user_query, k=4)
        
        if not relevant_docs:
            print("[Assistente]: Desculpe, não encontrei informação relevante sobre isso nos documentos.")
            continue
            
        context = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        prompt = f"""
        Baseado estritamente no seguinte contexto retirado de documentos eleitorais de 2025, responde à pergunta.
        A tua resposta deve ser clara, concisa e focar-se apenas na informação fornecida no contexto.
        Se a informação não estiver no contexto, indica isso mesmo.

        Contexto Fornecido:
        ---
        {context}
        ---

        Pergunta do Utilizador: {user_query}

        Resposta Concisa:
        """
        
        generated_answer = generator(prompt, max_length=512, num_beams=5, early_stopping=True)
        
        print(f"\n[Assistente]: {generated_answer[0]['generated_text']}")

# Ponto de entrada do script.
if __name__ == "__main__":
    main()