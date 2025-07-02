'''
LLMs apenas para os documentos tematicos
'''

import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DATA_FOLDER = './processed_data'
EMBEDDING_MODEL_NAME = 'distiluse-base-multilingual-cased-v1'


class KnowledgeBase:
    """
    Esta classe encapsula toda a lógica para carregar os documentos,
    processá-los, criar a base de dados vetorial e pesquisar nela.
    """

    def __init__(self, folder_path, model_name):
        print("-> A inicializar a Base de Conhecimento")
        self.folder_path = folder_path
        self.embedding_model = SentenceTransformer(model_name)
        
        # estas listas vão guardar os nossos dados processados
        self.chunks = []  # com os pedaços de texto (ex: proposta da AD para a Defesa)
        self.metadata = []  # com a info sobre a origem de cada chunk (ficheiro, partido)

        self.index = None  # BD vetorial
        self.build()

    def _load_and_process_documents(self):
        #divide os ficheiros temáticos em chunks lógicos e extrai metadados.

        print("-> A ler e processar os documentos temáticos...")
        
        # o nome do partido está no início de uma linha e seguido por uma nova linha
        party_pattern = re.compile(r"^\s*([A-ZÇÃ][a-zA-ZçÇãéáíóúâêô\s–-]+)\s*$", re.MULTILINE)
        
        for filename in os.listdir(self.folder_path):
            if 'tema' in filename:
                file_path = os.path.join(self.folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # divide o conteúdo por partido. 're.split' usa o padrão para dividir,
                # mantendo os delimitadores (nomes dos partidos)
                parts = party_pattern.split(content)
                
                # a 1a parte antes do primeiro partido é geralmente lixo (cabeçalho, etc.).
                # (nome_do_partido, bloco_de_texto_do_partido).
                for i in range(1, len(parts), 2):
                    party_name = parts[i].strip()
                    party_proposals_text = parts[i+1].strip()
                    
                    if not party_proposals_text:
                        continue

                    # Adiciona o bloco inteiro como um chunk
                    # Poderíamos dividir ainda mais (por bullet point), mas isto já é um bom começo.
                    self.chunks.append(f"Propostas de {party_name}: {party_proposals_text}")
                    self.metadata.append({
                        'source_file': filename,
                        'party': party_name
                    })
        
        print(f"-> Foram encontrados e processados {len(self.chunks)} chunks de informação.")

    def build(self):
        """
        Constrói o índice vetorial FAISS a partir dos chunks de texto.
        """
        self._load_and_process_documents()
        
        if not self.chunks:
            print("AVISO: Nenhum chunk foi carregado. A base de conhecimento está vazia.")
            return

        print("-> A gerar embeddings (vetores) para cada chunk de texto...")
        # Converte todos os chunks de texto em vetores numéricos.
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        
        # A dimensão do vetor (ex: 768) é determinada pelo modelo de embedding.
        d = embeddings.shape[1]
        
        # Cria o índice FAISS. 'IndexFlatL2' é um índice simples e eficaz para começar.
        self.index = faiss.IndexFlatL2(d)
        
        # Adiciona os vetores ao índice.
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
        # Converte a query do utilizador num vetor.
        query_embedding = self.embedding_model.encode([query])
        
        # Procura no índice FAISS pelos 'k' vetores mais próximos.
        # D: distâncias, I: índices (posições) dos vetores encontrados.
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        # Constrói a lista de resultados com o texto e os metadados.
        results = []
        for i in indices[0]:
            if i != -1: # FAISS usa -1 se encontrar menos de 'k' resultados.
                results.append({
                    'text': self.chunks[i],
                    'metadata': self.metadata[i]
                })
                
        print(f"-> Encontrados {len(results)} documentos relevantes.")
        return results

# ==============================================================================
# FUNÇÃO PRINCIPAL DO ASSISTENTE
# ==============================================================================

def main():
    """
    Função principal que inicia a base de conhecimento e entra num loop
    para responder às perguntas do utilizador.
    """
    # Carrega e constrói a base de conhecimento. Isto pode demorar um pouco na primeira vez.
    kb = KnowledgeBase(DATA_FOLDER, EMBEDDING_MODEL_NAME)

    # Carrega um modelo para gerar as respostas (pode ser o mesmo da sumarização ou outro).
    # Usar um modelo mais leve como o flan-t5-base é bom para velocidade.
    print("\n-> A carregar modelo de Geração de Respostas (pode demorar um pouco)...")
    generator = pipeline('text2text-generation', model='google/flan-t5-base')
    print("-> Assistente pronto para responder. Escreva 'sair' para terminar.")
    
    while True:
        # Pede input ao utilizador.
        user_query = input("\n[Você]: ")
        if user_query.lower() == 'sair':
            break
            
        relevant_docs = kb.search(user_query, k=3) # pega nos 3 documentos mais relevantes
        
        if not relevant_docs:
            print("[Assistente]: Desculpe, não encontrei informação relevante sobre isso nos documentos.")
            continue
            
        context = ""
        for doc in relevant_docs:
            context += "\n\n" + doc['text']
        
        prompt = f"""
        Baseado estritamente no seguinte contexto retirado dos programas eleitorais de 2025, responde à pergunta do utilizador de forma clara e concisa.
        Não inventes informação. Se a resposta não estiver no contexto, diz que não encontraste essa informação específica.

        Contexto:
        ---
        {context}
        ---

        Pergunta: {user_query}

        Resposta:
        """
        generated_answer = generator(prompt, max_length=512, num_beams=4, early_stopping=True)
        print(f"\n[Assistente]: {generated_answer[0]['generated_text']}")


if __name__ == "__main__":
    main()