import os
from transformers import pipeline

processed_data_folder = './processed_data'

file_to_load = 'programa_eleitoral_PS_legislativas_2025.txt' # ir trocando os ficheiros !! 
file_path = os.path.join(processed_data_folder, file_to_load)

# um modelo para sumarizar o conteúdo que seja bom em português
summarizer_model = "csebuetnlp/mT5_multilingual_XLSum" # modelo MT5 treinado para sumários multilingues
print(f"A carregar o modelo de sumarização: {summarizer_model}")

# inicializamos o pipeline de sumarização com o modelo escolhido
# este aqui tende a demorar algum tempo a descarregar o modelo na primeira vez
summarizer = pipeline("summarization", model=summarizer_model)
print("Modelo carregado.")

if os.path.exists(file_path):
    try:
        # aqui vemos o conteúdo
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        print(f"Conteúdo carregado com sucesso do ficheiro: {file_to_load}")

        # só para verificar o conteudo
        #print("\n", "- " * 30, " Início do Conteúdo ", " -" * 30, "\n")
        #print(file_content[:500])
        #print("\n", "- " * 30, " Fim do Conteúdo ", " -" * 30, "\n")

        # sumarizacao
        # os modelos de sumarização têm limites de tokens de entrada.
        # e um programa eleitoral inteiro é demasiado longo.
        # Vamos pegar apenas numa parte do texto para testar.
        # Podes ajustar este valor. Tenta encontrar uma secção que faça sentido.
        # Cuidado com os limites do modelo (geralmente 512 ou 1024 tokens).
        # Pegar nos primeiros N caracteres não garante que seja uma secção coerente,
        # mas serve para um teste inicial.
        text_to_summarize = file_content[:2000] # Pegar nos primeiros 2000 caracteres como exemplo

        print(f"\n--- A Sumarizar (primeiros {len(text_to_summarize)} caracteres) ---")

        # Realiza a sumarização
        # max_length e min_length controlam o tamanho do resumo gerado
        summary = summarizer(text_to_summarize, max_length=150, min_length=30, do_sample=False)

        print("\n--- Resumo Gerado ---")
        print(summary[0]['summary_text'])
        print("---------------------")

    except Exception as e:
        print(f"Erro ao ler o ficheiro {file_to_load}: {e}")
else:
    print(f"Erro: O ficheiro {file_to_load} não existe vê lá melhor isso!")