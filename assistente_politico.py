import os
import google.generativeai as genai

# os partidos podem ser mencionados de várias formas, por isso vamos criar um dicionário
# com as siglas oficiais e os nomes completos, além de aliases comuns.
PARTIDOS_INFO = {
    "AD": {
        "nome_completo": "Aliança Democrática",
        "aliases": ["ad", "alianca democratica", "aliança democrática", "psd", "cds-pp", "cds", "ppm"]
    },
    "BE": {
        "nome_completo": "Bloco de Esquerda",
        "aliases": ["be", "bloco de esquerda", "bloco"]
    },
    "CHEGA": {
        "nome_completo": "Chega",
        "aliases": ["chega", "ch"]
    },
    "IL": {
        "nome_completo": "Iniciativa Liberal",
        "aliases": ["il", "iniciativa liberal"]
    },
    "LIVRE": {
        "nome_completo": "Livre",
        "aliases": ["livre", "l"]
    },
    "PAN": {
        "nome_completo": "Pessoas-Animais-Natureza",
        "aliases": ["pan", "pessoas-animais-natureza", "pessoas animais natureza"]
    },
    "PS": {
        "nome_completo": "Partido Socialista",
        "aliases": ["ps", "partido socialista"]
    },
    "PCP": { 
        "nome_completo": "CDU - Coligação Democrática Unitária (PCP-PEV)",
        "aliases": ["cdu", "pcp", "pev", "comunista", "partido comunista portugues", "os verdes", "psp"]
    }
}

def carregar_api_key(nome_ficheiro="API.md"):
    # por seguranca, criei um ficheiro onde coloco a chave de API. pedi a do google!!
    try:
        with open(nome_ficheiro, 'r') as f:
            chave = f.read().strip()

        if not chave:
            print(f"erro: ficheiro vazio.")
            return None
        return chave
    
    except FileNotFoundError:
        print(f"erro: ficheiro não encontrado.")
        return None

api_key = carregar_api_key()

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
    except Exception as e:
        print(f"erro: chave inválida. erro: {e}")
        exit()
else:
    print("não podemos continuar sem uma chave de API válida.")
    exit()

# --------------------------------------------------------------------------

def carregar_dados_processados(pasta_dados):
    print("A carregar os dados dos programas eleitorais!!")
    dados = {"partidos": {}, "temas": {}}
    
    if not os.path.isdir(pasta_dados):
        print(f"erro: pasta '{pasta_dados}' não encontrada.")
        exit()
    for nome_ficheiro in os.listdir(pasta_dados):
        caminho_completo = os.path.join(pasta_dados, nome_ficheiro)
        try:
            with open(caminho_completo, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            if "_tema_" in nome_ficheiro:
                tema = nome_ficheiro.split('_tema_')[1].replace('.txt', '')
                dados["temas"][tema] = conteudo
            else:
                partido = nome_ficheiro.split('_')[2]
                dados["partidos"][partido] = conteudo
        except Exception as e:
            print(f"Não foi possível ler o ficheiro {nome_ficheiro}. erro: {e}")

    print("Dados carregados com sucesso!\n")
    return dados


def encontrar_partido(input_utilizador, info_partidos):

    input_normalizado = input_utilizador.lower().strip()

    for sigla, info in info_partidos.items():
        if input_normalizado in info["aliases"]:
            return sigla
        
    return None # não encontrou aiaiaiiai

def resumir_programa_partido(sigla, dados_eleitorais):
    nome_completo = PARTIDOS_INFO[sigla]["nome_completo"]
    print(f"\nA gerar resumo para o partido: {nome_completo}")
    
    programa_completo = dados_eleitorais["partidos"][sigla]
    prompt = f"Age como um analista político português, neutro e informativo. Com base no seguinte programa eleitoral do partido {nome_completo} para as legislativas de 2025, gera um resumo conciso com no máximo 5 parágrafos. Não uses vocabulário demasiado formal, mas formal o suficiente. Foca-te nos objetivos principais, nas propostas mais importantes e na visão geral do partido.\n\n--- INÍCIO DO PROGRAMA ---\n{programa_completo}\n--- FIM DO PROGRAMA ---"
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao gerar o resumo: {e}"

def apresentar_info_por_tema(tema, dados_eleitorais):
    print(f"\nA apresentar propostas para o tema: {tema}...")
    
    if tema not in dados_eleitorais["temas"]:
        return "erro: tema não encontrado. Tente um dos seguintes temas: " + ", ".join(dados_eleitorais["temas"].keys())
    
    return dados_eleitorais["temas"][tema]

def criar_perfil_partido(partido, dados_eleitorais):
    print(f"\nA criar perfil básico para o partido: {partido}")
    
    if partido not in dados_eleitorais["partidos"]:
        return "erro: partido não encontrado. Tente uma das seguintes siglas: " + ", ".join(dados_eleitorais["partidos"].keys())
    programa_completo = dados_eleitorais["partidos"][partido]
    prompt = f"Cria uma descrição sucinta do partido {partido}, baseando-te estritamente na informação do seu programa eleitoral de 2025 que te é fornecido. O perfil deve ter um tom neutro, informativo e ser apresentado num único parágrafo com cerca de 150 palavras.\n\n--- INÍCIO DO PROGRAMA ---\n{programa_completo}\n--- FIM DO PROGRAMA ---"
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao criar o perfil: {e}"

# <<< SUBSTITUA A FUNÇÃO ANTIGA POR ESTA NOVA VERSÃO >>>

def formatar_e_apresentar_tema(texto_tema):
    """
    Recebe um bloco de texto com propostas e formata-o
    de forma clara e legível. Assume que cada proposta está numa só linha.
    """
    linhas = texto_tema.split('\n')
    output_formatado = []
    
    for linha in linhas:
        linha_limpa = linha.strip()
        
        if not linha_limpa:
            continue # Ignora linhas em branco
        
        # Se a linha começa com '•', é uma proposta.
        if linha_limpa.startswith('•'):
            proposta = linha_limpa[1:].strip()
            output_formatado.append(f"  - {proposta}")
        # Se não, é um título de partido ou o título geral.
        else:
            # Adiciona uma linha em branco antes de um novo título para espaçamento
            if output_formatado and not output_formatado[-1].strip() == "":
                 output_formatado.append("") 
            
            output_formatado.append(f"--- {linha_limpa} ---")

    # Junta todas as linhas formatadas numa única string de texto
    return "\n".join(output_formatado)

def formatar_tema_com_llm(tema, texto_bruto_tema):

    prompt = f"""
    Age como um editor de texto extremamente organizado. A tua tarefa é pegar no seguinte texto bruto sobre o tema '{tema}'
    e formatá-lo numa lista clara, limpa e fácil de ler.

    Siga estas regras estritamente:
    1.  O nome de cada partido deve ser um título principal. Formata-o assim: 'NOME DO PARTIDO - - - - - - -- - - - - - - - --\n'.
    2.  Cada proposta individual de um partido deve começar com um traço e um espaço, como numa lista. Exemplo: '  - Aumentar o investimento.'
    3.  **Importante:** Se uma proposta estiver dividida em várias linhas no texto original, junta-as numa única linha.
    4.  Remove todo o "lixo visual", como linhas em branco desnecessárias ou símbolos de '•'. Não adiciones comentários teus.
    5.  Mantém a linguagem original das propostas.

    Aqui está o texto bruto:
    --- INÍCIO DO TEXTO BRUTO ---
    {texto_bruto_tema}
    --- FIM DO TEXTO BRUTO ---

    Apresenta apenas o resultado final formatado.
    """

    try:
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao formatar com a IA: {e}"
    
def main():
    print("Bem-vindo ao Assistente LLM de Política Portuguesa!")
    print("Este assistente permite explorar os programas eleitorais de 2025 dos principais partidos portugueses.")
    print("Pode pedir resumos, explorar propostas por tema e obter perfis básicos dos partidos.\n")
    dados = carregar_dados_processados('processed_data')
    
    while True:
        print("\n", "* " * 30, "MENU PRINCIPAL", "* " * 30)
        print("1. Resumo do Programa de um Partido")
        print("2. Ver Propostas por Tema")
        print("3. Perfil Básico de um Partido")
        print("0. Sair do Programa")
        
        escolha = input("\nEscolha uma opção: ")
        
        if escolha in ['1', '3']:
            partido_input = input("Insira a sigla ou nome do partido (ex: PS, Bloco de Esquerda, AD): ")
            sigla_oficial = encontrar_partido(partido_input, PARTIDOS_INFO)

            if sigla_oficial:
                if escolha == '1':
                    resultado = resumir_programa_partido(sigla_oficial, dados)
                    nome_completo = PARTIDOS_INFO[sigla_oficial]["nome_completo"]
                    print()
                    print("*" * 30, f" Resumo do Programa: {nome_completo}", "*" * 30)
                    print(resultado)
                    print("- " * 30)
                elif escolha == '3':
                    resultado = criar_perfil_partido(sigla_oficial, dados)
                    nome_completo = PARTIDOS_INFO[sigla_oficial]["nome_completo"]
                    print(f"\n* " * 30, f" Perfil Básico: {nome_completo}", "* " * 30)
                    print(resultado)
                    print("- " * 30)
            else:
                print()
                print("erro: partido não reconhecido, tente novamente.")

        elif escolha == '2':
            temas_disponiveis = ", ".join(dados["temas"].keys())
            tema_escolhido = input(f"Insira o tema ({temas_disponiveis}): ").lower()
            
            texto_bruto = apresentar_info_por_tema(tema_escolhido, dados)
            resultado_formatado = formatar_tema_com_llm(tema_escolhido, texto_bruto)
            print()
            print("*" * 30, f"Propostas sobre {tema_escolhido.capitalize()}", "*" * 30)
            print(resultado_formatado)
            print("-" * 30)

        elif escolha == '0':
            print("Obrigado por usar o assistente. Até à próxima!")
            break
            
        else:
            print("Opção inválida. Por favor, tente novamente.")

        input("\nPressione Enter para voltar ao menu...")

if __name__ == "__main__":
    main()