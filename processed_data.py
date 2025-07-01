import fitz # PyMuPDF
import os

# os ficheiros PDF são ótimos para leitura humana, mas para um programa de computador, 
# extrair o texto de forma limpa pode exigir algumas ferramentas.

input_folder = './data' 
output_folder = './processed_data' 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# todos os ficheiros na pasta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(input_folder, filename)
        txt_filename = filename.replace('.pdf', '.txt')
        txt_path = os.path.join(output_folder, txt_filename)

        try:
            # abrimos o PDF
            doc = fitz.open(pdf_path)
            text = ""
            # vamos pagina a pagina para extrair todo o texto
            # e guardamos num ficheiro .txt
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"Texto extraído de {filename} para {txt_filename}")

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")

print("Extração concluída.")
