# Portuguese Politics LLM Assistant

## Overview

The **Portuguese Politics LLM Assistant** is a tool designed to assist users in exploring the electoral programs of major Portuguese political parties for the 2025 legislative elections. Using large language models (LLMs) and retrieval-augmented generation (RAG), it provides summaries, thematic insights, party profiles, and personalized responses based on official party documents. This project is a work in progress and currently focuses solely on the 2025 electoral programs.

## Features

- **Party Program Summaries**: Concise summaries of each party's 2025 electoral program.
- **Thematic Proposals**: View party proposals by key themes (e.g., work, health).
- **Party Profiles**: Basic profiles derived from electoral programs.
- **Personalized Queries**: Ask questions or state positions to see party alignments.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
   ```

2. **Set Up a Virtual Environment** :
    - Ensure you have Python 3.11 installed.
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **API Key Setup**:
   - Create a file named `API.md` in the root directory.
   - Add your Google Generative AI API key (e.g., `YOUR_API_KEY_HERE`).

5. **Process Data**:
   - Place PDF electoral programs in the `data/` folder.
   - Run `processed_data.py` to convert PDFs to text files in `processed_data/`.

## Usage

1. **Run the Assistant**:
   ```bash
   python assistente_politico.py
   ```

2. **Menu Options**:
   - **1. Resumo do Programa de um Partido**: Summarize a party’s program (e.g., "PS").
   - **2. Ver Propostas por Tema**: View proposals by theme (e.g., "trabalho").
   - **3. Perfil Básico de um Partido**: Get a party profile (e.g., "BE").
   - **4. Pergunta Aberta / Análise de Posições**: Ask a question or state a position (e.g., "I support renewable energy").
   - **0. Sair do Programa**: Exit.

3. **Examples**:
   - Summary: Option 1 → "PS" → View Partido Socialista’s program summary.
   - Theme: Option 2 → "saúde" → See health-related proposals.
   - Query: Option 4 → "Defendo o aumento do salário mínimo" → Get party alignments.

## Data Sources

- **Primary**: 2025 electoral programs from party websites (PDFs in `data/`).
- **Processed**: Text files in `processed_data/`.

## Limitations

- Limited to 2025 electoral programs; no historical data or real-time updates.
- Dependent on the quality and availability of processed data.
- Work in progress; some features (e.g., thematic summaries) may be incomplete.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch for your changes.
3. Submit a pull request with a clear description.


## Contact

For questions or suggestions, feel free to contacte me !!

