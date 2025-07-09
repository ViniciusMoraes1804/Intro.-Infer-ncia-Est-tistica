# -*- coding: utf-8 -*-


#Download Dataset
!pip install kaggle -q
import os

os.environ['KAGGLE_USERNAME'] = "drysaliva"
os.environ['KAGGLE_KEY'] = "c5bf9cb0ce67bb03eed03787d9e96cf7"

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

!kaggle datasets download -d mahdimashayekhi/social-media-vs-productivity --unzip

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# leitura dos dados
df = pd.read_csv('social_media_vs_productivity.csv')
print("Informações gerais do dataframe:")
df.info()
print("\n")
df

# quantidade de dados faltantes em cada coluna
print('Dados Faltantes por Coluna:')
print(df.isnull().sum())
print('\n')

# Box-plots das variáveis quantitativas do dataset
plt.figure(figsize=(9,6))
df_num = df.select_dtypes(include='number')
ax = sns.boxplot(data=df_num)
plt.xticks(rotation=45, ha='right')
plt.savefig('grafico_pizza_genero.png', dpi=300, bbox_inches='tight')

plt.show()

# dados estatísticos de cada item do dataset
df_num.describe()

tab1 = pd.crosstab(index=df['gender'], columns='count')
tab1.plot.pie(y='count')

tab2 = pd.crosstab(index=df['social_platform_preference'], columns='count')
tab2.plot.pie(y='count')

#Tratamento dos dados

df_limpo = df.dropna()
df_limpo

print('Estatísticas descritivas:')
df_limpo.describe()

# tiro uma amostra de tamanho 500 com uma semente fixa para replicação do experimento

amostra = df_limpo.sample(n=500, random_state=42)
amostra

# dados estatísticos da amostra
amostra.describe()

"""# **Teste para investigar a relação entre o uso de plataformas de mídia social e a produtividade dos indivíduos, utilizando o conjunto de dados fornecido.**

**H0**: Não há diferença significativa na
produtividade entre indivíduos que utilizam
diferentes plataformas de mídia social

**H1**: Existe uma diferença significativa na
produtividade  entre pelo menos dois grupos
de indivíduos que utilizam diferentes plataformas de mídia social

Visualização:
"""

#Distribuição da pontuação de produtividade real

plt.figure(figsize=(10,6))
sns.histplot(df_limpo['actual_productivity_score'], kde=True)

plt.title('Distribuição da Pontuação de Produtividade Real')
plt.xlabel('Pontuação de produtividade real')

plt.ylabel('Frequência')
plt.show()
plt.close()

""":**Visualmente, a distribuição da produtividade real se
assemelha a uma curva Normal**
"""

#Distribuição da preferência por plataforma social
plt.figure(figsize=(10, 6))

sns.countplot(data=df_limpo,y='social_platform_preference',order=df_limpo['social_platform_preference'].value_counts().index)
plt.title('Distribuição da Preferência por Rede Social')
plt.xlabel('Contagem')===
plt.ylabel('Plataforma Social Preferida')
plt.show()
plt.close()

"""**As plataformas
(TikTok, Telegram, Instagram, Twitter, Facebook) têm uma distribuição
relativamente equilibrada de usuários.**
"""

#Box plot da produtividade real por plataforma social preferida
plt.figure(figsize=(12,7))
sns.boxplot(data=df_limpo,x='social_platform_preference',y='actual_productivity_score')
plt.title('Pontuação de Produtividade Real por Rede Social Preferida')
plt.xlabel('Plataforma Social Preferida')
plt.ylabel('Pontuação de Produtividade Real')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()
plt.close()

"""**Visualmente, as medianas entre os grupos parecem
ser bastante próximas.**
"""

# Violin plot da produtividade real por rede social preferida
plt.figure(figsize=(12,7))

sns.violinplot(data=df_limpo,x='social_platform_preference',y='actual_productivity_score')
plt.title('Distribuição da Pontuação de Produtividade Real por Rede Social Preferida')
plt.xlabel('Plataforma Social Preferida')
plt.ylabel('Pontuação de Produtividade Real')
plt.xticks(rotation=45,ha='right')

plt.tight_layout()
plt.show()
plt.close()

"""***a densidade da distribuição, confirma que as
distribuições de produtividade entre as plataformas são semelhantes.***

### **Teste de Normalidade**
Antes de escolher o teste inferencial adequado, verificaremos a normalidade da variável
actual_productivity_score para cada grupo de social_platform_preference. Faremos uma amostragem de 500 dados por grupo para aplicar o teste de Shapiro-Wilk,
que é mais adequado para amostras de tamanho moderado e menos sensível a pequenos desvios em grandes populações.

**H0**: Os dados seguem uma distribuição normal.

**H1**: Os dados não seguem uma distribuição normal.
"""

print('---Teste de Normalidade por Rede Social(com amostragem) ---\n')
shapiro_resultado ={}
plataformas = amostra['social_platform_preference'].unique()

for plataforma in plataformas:
    print(f'\nRede Social: {plataforma}')
    #seleciona os dados da rede social
    dados = amostra[amostra['social_platform_preference']== plataforma]['actual_productivity_score']

    #Teste de Shapiro-Wilk
    if len(dados)>=3:
        estat_shapiro, p_shapiro = stats.shapiro(dados)
        shapiro_resultado[plataforma] = {'statistic':estat_shapiro,'p_value':p_shapiro}
        print(f'Shapiro-Wilk: Estatística = {estat_shapiro:.4f}, p-valor = {p_shapiro:.4f}')
        if p_shapiro > 0.05:
            print('Não rejeito H0 (dados da amostra parecem normais)')

        else:
            print('Rejeito H0 (dados da amostra não parecem normais)')
    else:
        print(f'Shapiro-Wilk: Não aplicável para {len(amostra)} amostras (deve ter no mínimo 3)')
        shapiro_resultado[plataforma] = {'statistic': np.nan, 'p_value': np.nan}

    #visualização da normalidade da amostra(QQ-PLOT)
    plt.figure(figsize=(8,6))

    stats.probplot(dados, dist="norm",plot=plt)

    plt.title(f'QQ-Plot para Produtividade Real(Amostra) - {plataforma}')
    plt.xlabel('Quantis Teóricos')
    plt.ylabel('Quantis da Amostra')

    plt.show()
    plt.close()

"""Com base nos resultados do teste de Shapiro-Wilk aplicado às amostras de 500 dados por plataforma:
*   Se o p-valor for > 0.05 para a maioria dos grupos, podemos considerar que os dados da amostra são aproximadamente normais e optar por um teste paramétrico (ANOVA).
*   Se o p-valor for <= 0.05 para a maioria dos grupos, os dados da amostra não são normais, e um teste não paramétrico (Kruskal-Wallis) é mais apropriado.

Vamos verificar os resultados para decidir

"""

#Vamos contar quantos grupos rejeitaram a normalidade
num_rejeicoes_normalidade = sum(1 for p in shapiro_resultado.values() if p['p_value']<= 0.05)


#Se mais da metade dos grupos rejeitar a normalidade:
if num_rejeicoes_normalidade/len(plataformas) > 0.5:
    print('Como mais da metade dos grupos rejeitou a normalidade dos dados,usar teste não paramétrico (Kruskal-Wallis)\n')
    dados_kruskal = []

    for plataforma in plataformas:
        dados_kruskal.append(df_limpo[df_limpo['social_platform_preference']==plataforma]['actual_productivity_score'])

    estat_teste,p_valor_teste = stats.kruskal(*dados_kruskal)
    nome_teste = "Kruskal-Wallis"
    h0_teste = "Não há diferença significativa na produtividade entre os grupos de plataformas de rede social"
    h1_teste = "Existe uma diferença significativa na produtividade entre pelo menos dois grupos de plataformas de rede social"
#Se mais da metade dos grupos não rejeitar a normalidade:
else:
    print('Como mais da metade dos grupos não rejeitou a normalidade dos dados, usar Teste Paramétrico (ANOVA)\n')
    print('Verificando homocedasticidade para ANOVA.')

    # Verificar homocedasticidade para ANOVA
    # H0:As variâncias dos grupos são iguais
    # H1:Pelo menos uma variância é diferente

    grupos_anova = [df_limpo[df_limpo['social_platform_preference'] == p]['actual_productivity_score'] for p in plataformas]

    estat_levene, p_levene = stats.levene(*grupos_anova)

    print(f'Teste de Levene (Homocedasticidade): Estatística = {estat_levene:.4f}, p-valor = {p_levene:.4f}')
    if p_levene >0.05:
        print('Não rejeito H0 (variâncias parecem homogêneas).Vamos usar ANOVA.')
        estat_teste,p_valor_teste = stats.f_oneway(*grupos_anova)
        nome_teste = "ANOVA"
        h0_teste = "Não há diferença significativa na produtividade entre os grupos de plataformas de rede social"
        h1_teste = "Existe uma diferença significativa na produtividade entre pelo menos dois grupos  de rede social"
    else:
        print('Rejeito H0 (variâncias não são homogêneas).')
        #Se as variâncias não são homogêneas,mesmo com normalidade, Kruskal-Wallis ainda é uma opção
        dados_kruskal =[]
        for plataforma in plataformas:

            dados_kruskal.append(df_limpo[df_limpo['social_platform_preference'] == plataforma]['actual_productivity_score'])

        estat_teste,p_valor_teste =stats.kruskal(*dados_kruskal)

        nome_teste ="Kruskal-Wallis"
        h0_teste ="Não há diferença significativa na produtividade entre os grupos "
        h1_teste = "Existe uma diferença significativa na produtividade entre pelo menos dois grupos"

print(f' Resultados do Teste do {nome_teste}\n')
print(f'Objetivo: Verificar se existe diferença significativa na pontuação de produtividade real entre indivíduos que utilizam diferentes plataformas de mídia social.\n')

print('Hipóteses:')
print(f'H0: {h0_teste}')
print(f'H1: {h1_teste}\n')


print("coclusao do teste:\n")
print(f'Resultados do Teste de {nome_teste}\nEstatística: {estat_teste:.4f}\nValor-p: {p_valor_teste:.4f}\n')


if p_valor_teste < 0.05:
    print(f'Rejeitamos H0.')
else:
    print('não rejeitamos H0.')

"""### **Conclusão final**
A análise exploratória dos dados nos revelou que a pontuação de produtividade real
supostamente apresentava uma tendência para uma distribuição normal.No entanto, o
teste de normalidade de Shapiro-Wilk, aplicado a amostras de cada grupo de rede social,
indicou que a suposição de normalidade não era concreta para a maioria dos grupos.
Isso nos levou a optar por um teste não paramétrico(Kruskal-Wallis) para comparar os grupos.
O resultado do teste mostrou que não devemos rejeitar  a hipótese nula.
Isso significa que, com os dados disponíveis,nao tivemos evidências estatísticas suficientes para afirmar que a preferência por uma rede social
teve um impacto significativo na produtividade real dos indivíduos.Em
outras palavras, a plataforma que uma pessoa prefere usar não aparentou estar associada
a diferentes níveis de produtividade,pelo menos não de forma estatisticamente
significativa nesse conjunto de dados.

# **Teste para verificar a relação entre a produtividade e a quantidade de horas trabalhadas**

### **Teste de Normalidade**

Para descobrir qual teste estatístico é mais adequado, realizaremos o teste de Shapiro-Wilk para verificar a normalidade das variáveis (se aproximam-se de uma curva Normal).

H0: As amostras seguem uma distribuição normal.

H1: As amostras não seguem uma distribuição normal.
"""

# teste de Shapiro Wilk para normalidade

from scipy import stats

for coluna in ['work_hours_per_day', 'actual_productivity_score']:
    x = amostra[coluna]
    stat, p = stats.shapiro(x)
    if(p > 0.05):
      print(f'\n{coluna}: Não rejeito a normalidade (p-valor = {p})')
    else:
      print(f'\n{coluna}: Rejeito a normalidade (p-valor = {p})')

    #visualização da normalidade
    plt.figure(figsize=(8, 5))
    stats.probplot(amostra[coluna], dist="norm", plot=plt)
    plt.title(f'Aproximação da normalidade - {coluna}')
    plt.xlabel('Quantis Teóricos')
    plt.ylabel('Quantis da Amostra')
    plt.show()
    plt.close()

"""**Conclusão do teste de normalidade:** Como um dos p-valores para o teste de normalidade foi menor do que 0.05, rejeitamos a hipótese nula para 'actual_productivity_score'. Ou seja, temos evidências para afirmar que os dados dessa coluna não seguem uma distribuição Normal. Isso também é visível no gráfico, em que os pontos da amostra divergem da linha da Normal.

### **Teste para coeficiente de correlação**

Como um dos testes para normalidade foram rejeitados, utilizaremos o teste de correlação de Spearman para investigar a relação entre produtividade e a quantidade de horas trabalhadas.

H0: Não há correlação entre as variáveis (ρ = 0)

H1: Há correlação entre as variáveis (𝜌 ≠ 0)
"""

from scipy.stats import spearmanr

rho, p = spearmanr(amostra['actual_productivity_score'], amostra['work_hours_per_day'])

print(f"Coeficiente de Spearman = {rho}")
print(f"p-valor = {p}")

if p < 0.05:
    print("Rejeito H0 - há evidência significativa de correlação linear.")
else:
    print("Não rejeito H0 - Não há evidência significativa de correlação linear.")

sns.regplot(x=amostra['actual_productivity_score'], y=amostra['work_hours_per_day'], scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
plt.title(f"Correlação de Spearman")
plt.xlabel("actual_productivity_score")
plt.ylabel("work_hours_per_day")
plt.show()

"""### **Conclusão final**

Para averiguar se havia correlação entre a produtividade e a quantidade de horas trabalhadas, primeiro testamos a normalidade utilizando o teste de Shapiro-Wilk para as variáveis 'actual_productivity_score' e 'work_hours_per_day'. Como uma delas rejeitou a normalidade, utilizamos o teste não-paramétrico para coeficiente de correlação de Spearman. Nele, o p-valor foi maior do que 0.05, o que nos leva a não rejeitar H0. Isso significa que não temos evidências para rejeitar a ideia de que as duas variáveis não estão correlacionadas.

# **Teste para investigar a relação entre a quantidade de horas de sono e a de horas de trabalho**

### **Teste de normalidade**

Para descobrir qual teste estatístico é mais adequado, realizaremos o teste de Shapiro-Wilk para verificar a normalidade das variáveis (se aproximam-se de uma curva Normal).

H0: As amostras seguem uma distribuição normal.

H1: As amostras não seguem uma distribuição normal.
"""

# teste de Shapiro Wilk para normalidade

from scipy import stats

for coluna in ['work_hours_per_day', 'sleep_hours']:
    x = amostra[coluna]
    stat, p = stats.shapiro(x)
    if(p > 0.05):
      print(f'\n{coluna}: Não rejeito a normalidade (p-valor = {p})')
    else:
      print(f'\n{coluna}: Rejeito a normalidade (p-valor = {p})')

    #visualização da normalidade
    plt.figure(figsize=(8, 5))
    stats.probplot(amostra[coluna], dist="norm", plot=plt)
    plt.title(f'Aproximação da normalidade - {coluna}')
    plt.xlabel('Quantis Teóricos')
    plt.ylabel('Quantis da Amostra')
    plt.show()
    plt.close()

"""**Conclusão do teste de normalidade:** como o p-valor de ambos é maior do que 0.05 (p-valor > 0.05), não rejeitamos a hipótese nula H0. Ou seja, não há evidências para rejeitar a normalidade das variáveis 'work_hours_per_day' e 'sleep_hours' na amostra. Isso também é visível nos gráficos, na qual os pontos da amostras se aproximam da reta linear da Normal.

### **Teste para coeficiente de correlação**

Como ambas as variáveis seguem uma distribuição aproximadamente Normal, utilizaremos o teste para coeficiente de correlação de Pearson para verificar se as variáveis estão associadas entre si ou não.

H0: Não há correlação entre as variáveis (ρ = 0)

H1: Há correlação entre as variáveis (𝜌 ≠ 0)
"""

from scipy.stats import pearsonr

# Teste de correlação de Pearson
r, p_valor = pearsonr(amostra['work_hours_per_day'], amostra['sleep_hours'])

print(f"Coeficiente de correlação r = {r}")
print(f"p-valor = {p_valor}")

if p_valor < 0.05:
    print("Rejeito H0 - há evidência significativa de correlação linear.")
else:
    print("Não rejeito H0 - não há evidências de correlação linear.")

# Visualização do resultado do teste - gráfico de dispersão com linha de regressão
plt.figure(figsize=(8, 5))
sns.regplot(x=amostra['work_hours_per_day'], y=amostra['sleep_hours'], line_kws={'color': 'red'})
plt.title(f"Correlação de Pearson")
plt.xlabel("Quantidade de horas trabalhadas")
plt.ylabel("Quantidade de horas dormidas")
plt.tight_layout()
plt.show()

"""### **Conclusão Final**

Para verificar se há correlação entre a quantidade de horas trabalhadas e a de horas de sono, primerio tivemos que aplicar um teste de normalidade, que no caso, foi escolhido o Shapiro-Wilk, para descobrir se seria aplicado um teste paramétrico ou não-paramétrico. Como as variáveis não rejeitaram a normalidade, aplicamos o teste paramétrico para o coeficiente de correlação de Pearson. Esse teste resultou num p-valor maior do que 0.05, portanto não rejeitamos a hipótese nula. Ou seja, não temos evidências para rejeitar a hipótese de que a quantidade de horas trabalhadas e de horas dormidas não estão correlacionadas.

# **Teste para verificar a relação entre o gênero e a quantidade de horas trabalhadas**
"""

amostra['gender'].value_counts()

"""Como a categoria 'others' ficou com poucos observações, ela será retirada da análise e focaremos na relação de horas trabalhadas entre homens e mulheres."""

#excluindo linhas da amostra em que o gênero
amostra_teste = amostra[amostra['gender'].isin(['Female', 'Male'])]
amostra_teste

"""### **Teste de normalidade e de homocedasticidade**

Para descobrir qual teste estatístico é mais adequado, realizaremos o teste de Shapiro-Wilk para verificar a normalidade das variáveis (se aproximam-se de uma curva Normal) e o teste de Levene para verificar a homogeneidade das variâncias (homocedasticidade).

*Teste de Shapiro-Wilk*

H0: As amostras seguem uma distribuição normal.

H1: As amostras não seguem uma distribuição normal.
"""

# Teste de Shapiro-Wilk para normalidade

for genero in amostra_teste['gender'].unique():
    grupo = amostra_teste[amostra_teste['gender'] == genero]['work_hours_per_day']
    stat, p = stats.shapiro(grupo)
    if(p > 0.05):
      print(f'\n{genero}: Não rejeito a normalidade (p-valor = {p})')
    else:
      print(f'\n{genero}: Rejeito a normalidade (p-valor = {p})')

    #visualização da normalidade
    plt.figure(figsize=(8, 5))
    stats.probplot(grupo, dist="norm", plot=plt)
    plt.title(f'Aproximação da normalidade - {genero}')
    plt.xlabel('Quantis Teóricos')
    plt.ylabel('Quantis da Amostra')
    plt.show()
    plt.close()

"""**Conclusão do teste de normalidade:** como o p-valor de todas as categorias é maior do que 0.05 (p-valor > 0.05), não rejeitamos a hipótese nula H0. Ou seja, não há evidências para rejeitar a normalidade de 'work_hours_per_day' das categorias em 'gender'. Isso também é visível nos gráficos, na qual os pontos da amostras se aproximam da reta linear da Normal.

*Teste de Levene*

H0: as variâncias de todos os grupos são iguais.

H1: pelo menos uma das variâncias é diferente.
"""

# Teste de Levene para homogeneidade de variâncias
grupos = [amostra_teste[amostra_teste['gender'] == genero]['work_hours_per_day'] for genero in amostra_teste['gender'].unique()]
stat, p = stats.levene(*grupos)
if(p > 0.05):
  print(f'Não rejeito a homocedasticidade (p-valor = {p})')
else:
  print(f'Rejeito a homocedasticidade (p-valor = {p})')

# Visualizar os desvios-padrões
desvios = amostra_teste.groupby('gender')['work_hours_per_day'].std()
desvios.plot(kind='bar')
plt.title("Desvio padrão por gênero")
plt.ylabel("Desvio padrão")
plt.xlabel("Gênero")
plt.show()

"""**Conclusão do teste de homocedasticidade:** Como o p-valor é maior do que 0.05, não rejeitamos a hipótese nula H0. Ou seja, não temos evidências na amostra para rejeitar a ideia de que as variâncias dos grupos são iguais. Isso também é visível no gráfico de barras que mostra as variâncias de 'work_hours_per_day' por grupo('gender').

### **Teste t de student**

Como ambos os grupos ('Female' e 'Male') não rejeitaram a normalidade e a homecedasticidade de 'work_hours_per_day', utilizaremos o teste t de student para comparar as duas médias.

H0: as médias de horas trabalhadas de homens e mulheres são iguais. (μ1 = μ2)

H1: as médias são diferentes. (μ1 ≠ μ2)
"""

from scipy.stats import ttest_ind

genders = amostra_teste.groupby("gender")

t_stat, p = ttest_ind(genders.get_group("Female")['work_hours_per_day'], genders.get_group("Male")['work_hours_per_day'], equal_var=True)

print(f'Estatística t = {t_stat}')
print(f'p-valor = {p}')

if p < 0.05:
    print("Rejeito H0 - há diferença entre as médias.")
else:
    print("Não rejeito H0 - não há diferença significativa entre as médias.")

plt.figure(figsize=(8, 5))
sns.boxplot(x=amostra_teste['gender'], y=amostra_teste['work_hours_per_day'], data=amostra_teste)
plt.title("Horas trabalhadas por gênero")
plt.xlabel("Gênero")
plt.ylabel("Horas de trabalho")
plt.show()

amostra_teste.groupby("gender").describe()

"""### **Conclusão final**

A fim de verificar a relação entre o gênero e a quantidade de horas trabalhadas, primeiro tivemos que verificar a normalidade (teste de Shapiro_Wilk) e a homocedasticidade (teste de Levene) das variáveis, para descobrir qual teste de comparação de médias seria mais adequado. Como nenhum dos dois foi rejeitado, aplicamos o teste t de Student. Nele, foi encontrado um p-valor maior do que 0.05, de forma a não rejeitarmos H0. Isso significa que não há diferenças significativas na quantidade de horas trabalhadas entre homens e mulheres (ambos trabalham, aproximadamente, por um mesmo período de tempo).
"""
