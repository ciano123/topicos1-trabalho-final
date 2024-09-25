import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import statsmodels.api as sm
from scipy import stats
import category_encoders as ce
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score


# Carregar o dataset
# df = pd.read_csv('./data/datatran_tratado.csv', encoding='ISO-8859-1', delimiter=';')  # Altere para o caminho do seu dataset
df = pd.read_csv('./data/datatran_tratado.csv')  # Altere para o caminho do seu dataset

## ==================================
df_new = df.copy() # cria uma cópia do df original
df_new['data_inversa'] = pd.to_datetime(df_new['data_inversa']) # Formata a data para evitar inconsistências
df_new['horario'] = pd.to_datetime(df_new['horario'], format='%H:%M:%S').dt.time # Formata o horário para evitar inconsistências

df_new['month'] = df_new['data_inversa'].dt.month # Gera uma nova coluna com o dia em que o acidente aconteceu
df_new['day'] = df_new['data_inversa'].dt.day # Gera uma nova coluna com o dia em que o acidente aconteceu
df_new['hour'] = pd.to_datetime(df_new['horario'], format='%H:%M:%S').dt.hour # Gera uma nova coluna com a hora em que o acidente aconteceu

df_new['km'] = df_new['km'].str.replace(',', '.').astype(float) # Formata a coluna 'km' convertendo todos os seus valores para float

le = LabelEncoder()

colunas_para_label_encode = ['uf', 'causa_acidente', 'tipo_acidente',
    'sentido_via','classificacao_acidente', 'tracado_via', 'condicao_metereologica',
    'tipo_pista', 'regional', "dia_semana", "fase_dia", "br", "uso_solo"]

for coluna in colunas_para_label_encode:
  df_new[coluna] = le.fit_transform(df_new[coluna])

df = df_new

# Aplicação do Frequency Encoding nas colunas de cardinalidade
for col in ['municipio', 'delegacia', 'uop']:
    freq = df[col].value_counts(normalize=True)  # Frequência de cada categoria
    df_new[col + '_freq'] = df_new[col].map(freq)  # Substitui pela frequência

# Aplicação do Binary Encoding nas colunas de cardinalidade
binary_encoder = ce.BinaryEncoder(cols=['classificacao_acidente','municipio', 'delegacia', 'uop', 'condicao_metereologica', 'tipo_pista'])
df_new = binary_encoder.fit_transform(df_new)

df_numeric = df.select_dtypes(include=['int64', 'float64'])

# Gerar a matriz de correlação
correlation_matrix = df_numeric.corr()


## ======== INICIO PAG STREAM LIT

# Título do aplicativo
st.title("Análise de Acidentes de Trânsito")

# Dashboard Inicial
st.header("Visão Geral")
st.write("Este data app é uma ferramenta interativa para analisar dados de acidentes de trânsito, oferecendo visualizações e insights baseados em análises estatísticas e modelagem de dados. Ele permite visualizar uma interface simples e acessível, onde os usuários poderão explorar os dados e ver como variáveis como o número de veículos envolvidos afetam o número de mortes em acidentes.")


X = df[['causa_acidente', 'tipo_acidente', 'condicao_metereologica', 'veiculos', 'tracado_via', 'dia_semana', 'fase_dia', 'tipo_pista', 'sentido_via', 'uso_solo',
            'pessoas', 'br', 'km']]
y = df['classificacao_acidente']

# Separar em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestClassifier()

model.fit(X_train, y_train)

y_test_class = y_test.astype(int)  # Converter para inteiro, se necessário
y_pred_class = model.predict(X_test)

# Garantir que y_pred_class são previsões discretas
if y_pred_class.dtype != int:
    y_pred_class = np.round(y_pred_class).astype(int)

# Sidebar para seleção de visualização
st.sidebar.title("Analise exploratoria")
visualizacao = st.sidebar.selectbox("Escolha a visualização de dados:", 
                                    ['Matriz de Correlação', 
                                     'Matriz de Confusão',
                                     'Matriz de correlação em Heatmap', 
                                     'Importância das Variáveis',
                                     'Relação entre Número de Veículos e Mortes',
                                     'Análise ANOVA',
                                     'Modelo de Regressão Linear'])

if visualizacao == 'Matriz de Correlação':

    # Visualizar a matriz de correlação com um heatmap
    st.subheader("Matriz de Correlação")

    plt.figure(figsize=(15, 11))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.1f', linewidths=0.5)
    plt.title('Matriz de Correlação')
    st.pyplot(plt)

elif visualizacao == 'Matriz de Confusão':
    
    st.subheader("Matriz de Confusão")
    # Gerando a matriz de confusão

    plt.figure(figsize=(6, 4))  # Ajuste o tamanho (6 polegadas de largura por 4 polegadas de altura)
    cm = confusion_matrix(y_test_class, y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    st.pyplot(plt)

elif visualizacao == 'Matriz de correlação em Heatmap':
    # Matriz de correlação em um heatmap
    st.subheader("Matriz de correlação em Heatmap")
    # Correlação entre regiões e a gravidade dos acidentes
    correlation_matrix = df.groupby('regional')['classificacao_acidente'].value_counts(normalize=True).unstack()
    correlation_matrix = correlation_matrix.fillna(0)  # Preencher valores NaN com 0
    correlation_matrix = correlation_matrix.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlação entre Regiões e Gravidade dos Acidentes')
    plt.xlabel('Classificação do Acidente')
    plt.ylabel('Classificação do Acidente')
    st.pyplot(plt)

elif visualizacao == 'Importância das Variáveis':
    
    # Importância das Características
    st.subheader("Importância das Variáveis")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns  # Substitua se necessário

    plt.figure()
    plt.title('Importância das Características')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    st.pyplot(plt)

elif visualizacao == 'Relação entre Número de Veículos e Mortes':
    # Gráfico de dispersão do número de veículos e mortos
    st.subheader("Relação entre Número de Veículos e Mortes")
    plt.figure(figsize=(10, 6))
    plt.scatter(df['veiculos'], df['mortos'], color='blue', alpha=0.5)
    plt.title("Dispersão entre Veículos e Mortes")
    plt.xlabel("Número de Veículos")
    plt.ylabel("Número de Mortes")
    st.pyplot(plt)

elif visualizacao == 'Análise ANOVA':
    # Análise Estatística: ANOVA
    st.subheader("Análise ANOVA")
    F, p = stats.f_oneway(df['veiculos'], df['mortos'])
    st.write(f"F-statistic: {F:.2f}")
    st.write(f"p-value: {p:.2f}")

elif visualizacao == 'Modelo de Regressão Linear':
    # Regressão Linear
    st.subheader("Modelo de Regressão Linear")
    X = df[['veiculos']]
    y = df['mortos']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Avaliação do Modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'R² Score: {r2:.2f}')

    # Gráfico de Dispersão com Linha de Regressão
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Valores Reais', s=100)
    plt.plot(X_test, y_pred, color='red', label='Linha de Regressão', linewidth=2)
    plt.title("Regressão Linear: Valores Reais vs Linha de Regressão")
    plt.xlabel("Número de Veículos")
    plt.ylabel("Número de Mortes")
    plt.legend()
    st.pyplot(plt)


# Conclusão
st.header("Conclusão")
st.write("Esse aplicativo interativo permite explorar a relação entre o número de veículos e mortes em acidentes de trânsito, utilizando análises estatísticas e modelos de machine learning.")