# Data App: Análise de Acidentes de Trânsito

Este projeto consiste em um aplicativo web interativo desenvolvido em Python utilizando o framework **Streamlit**. O aplicativo analisa a relação entre o número de veículos envolvidos em acidentes de trânsito e o número de mortes, usando diversas técnicas de análise estatística e modelos de aprendizado de máquina.

## Requisitos

Antes de executar o aplicativo, certifique-se de que você tem o seguinte software instalado:

- Python 3.6 ou superior
- pip (gerenciador de pacotes do Python)

## Instalação

1. **Clone o repositório** ou **baixe** os arquivos do projeto.

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_DIRETORIO>
   ```
2. **Instale as dependências** necessárias para o projeto:
    ```bash
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn statsmodels category_encoders imblearn
    ```
## Execução

3. **Prepare seu dataset:** Certifique-se de que você possui um arquivo CSV:
- **nome_arquivo.csv**: contendo as informações necessárias para a análise.
4. **Execute o aplicativo** usando o comando abaixo:
    ```python
        streamlit run data_app.py
    ```