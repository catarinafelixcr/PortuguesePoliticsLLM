'''
APENAS para programas eleitorais
'''

import os
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import math

def load_summarization_model(model_name = "csebuetnlp/mT5_multilingual_XLSum"):
    # carrega o modelo de sumarização e o seu tokenizer da Hugging Face.

    print(f"-> A carregar o tokenizer e o modelo: {model_name}...")
    
    try:
        # o tokenizer é responsável por converter o texto em números (tokens) que o modelo entende
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        # o modelo é a "inteligência" que realiza a tarefa de resumo
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        print("-> Modelo e tokenizer carregados com sucesso.")
        return tokenizer, model
    
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo '{model_name}'.")
        print(f"Detalhe do erro: {e}")
        exit()

def read_text_file(file):

    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"-> Ficheiro '{os.path.basename(file)}' lido com sucesso com ({len(content)} caracteres).")
        return content
    except Exception as e:
        print(f"ERRO ao ler o ficheiro '{file}': {e}")
        return ""

def chunk_text(full_text, tokenizer, max_chunk_tokens = 400, overlap_tokens = 50):

    # aqui dividimos um texto longo em pedaços (chunks) mais pequenos
    # porquê? os modelos LLM têm um limite de contexto (context window), que é o número máximo de tokens
    # que conseguem processar de uma só vez. Para o mT5, este limite é 512 tokens!!!

    # 1. tokenização do texto completo para trabalhar com tokens em vez de caracteres
    tokens = tokenizer.encode(full_text)
    
    # 2. inicializacao da lista de chunks de tokens
    token_chunks = []

    # 3. percorremos a lista de tokens e criamos os chunks
    start = 0
    while start < len(tokens):
        # O fim do chunk é o início mais o tamanho máximo
        end = start + max_chunk_tokens
        # pegamos no pedaço de tokens e adiciona à lista
        chunk = tokens[start:end]
        token_chunks.append(chunk)
        
        # o próximo chunk começa um pouco antes do fim do atual ==> temos de ter sobreposição
        start += max_chunk_tokens - overlap_tokens

    print(f"-> Texto dividido em {len(token_chunks)} chunks de tokens.")

    # 4. convertemos os chunks de tokens de volta para texto.
    text_chunks = []
    for chunk in token_chunks:
        text_chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    
    return text_chunks


def summarize_document(file_path, tokenizer, model, output_folder = './summaries'):
    # aqui implementamos MapReduce.

    full_text = read_text_file(file_path)
    if not full_text:
        return None

    text_chunks = chunk_text(full_text, tokenizer)

    # PASSO 1: SUMARIZAÇÃO DE CADA CHUNK (MAP)
    print("\n(Etapa 'Map')")
    chunk_summaries = []
    for i, chunk in enumerate(text_chunks):
        print(f"   chunk {i + 1}/{len(text_chunks)}")
        
        # preparamos o input para o modelo
        inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        
        # geramos o resumo do chunk
        summary_ids = model.generate(inputs, 
                                     max_length=150, min_length=40, 
                                     length_penalty=2.0, 
                                     num_beams=4, early_stopping=True)
        
        # convertemos os IDs do resumo de volta para texto
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        chunk_summaries.append(summary_text)

    # PASSO 2: SUMARIZAÇÃO FINAL (REDUCE)
    print("\n(Etapa 'Reduce')")

    # juntamos todos os resumos num único texto
    combined_summaries = " ".join(chunk_summaries)
    
    inputs = tokenizer.encode("summarize: " + combined_summaries, 
                              return_tensors="pt", max_length=1024, truncation=True) 
    
    # por fim: geramos o resumo FINAL.
    # decidimos aumentar o max_length para permitir um resumo final mais completo
    final_summary_ids = model.generate(inputs, 
                                       max_length=512, min_length=150, 
                                       length_penalty=2.0, num_beams=4, 
                                       early_stopping=True)
    final_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)

    #print("\nRESUMO FINAL GERADO")
    #print(final_summary)
    
    # finalmente, guardamos o resumo
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    base_name = os.path.basename(file_path)
    output_filename = os.path.join(output_folder, f"resumo_{base_name}")
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_summary)
        print(f"\n-> guardado com sucesso: '{output_filename}'")
    except Exception as e:
        print(f"ERRO: ao guardar o resumo no ficheiro '{output_filename}': {e}")

    return final_summary


if __name__ == "__main__":

    processed_data_folder = './processed_data'
    
    file_to_process = 'programa_eleitoral_PS_legislativas_2025.txt'
    #file_to_process = 'programa_eleitoral_IL_legislativas_2025.txt'
    #file_to_process = 'programa_eleitoral_PCP_legislativas_2025.txt'

    full_file_path = os.path.join(processed_data_folder, file_to_process)
    tokenizer, model = load_summarization_model() ## basta carregar uma vez
    
    #summarize_document(full_file_path, tokenizer, model)

    print("\n\n--- A PROCESSAR TODOS OS PROGRAMAS ELEITORAIS ---")
    for filename in os.listdir(processed_data_folder):
        if filename.startswith('programa_eleitoral_') and 'tema' not in filename:
            print(f"* " * 30, f" A PROCESSAR {filename} ", "* " * 30)
            path = os.path.join(processed_data_folder, filename)
            summarize_document(path, tokenizer, model)