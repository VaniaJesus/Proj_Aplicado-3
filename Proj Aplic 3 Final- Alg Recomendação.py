#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


# Dataset de receitas 

receitas_1 = pd.read_excel("C:\\Users\\kika_\\OneDrive\\Área de Trabalho\\dados\\Receitas.TudoGostoso.xlsx",engine='openpyxl')


# In[9]:


receitas_1.columns


# In[10]:


# Dataset de Compras
compras_1 = pd.read_csv("C:\\Users\\kika_\\OneDrive\\Documentos\\Meus Documentos\\Mackeinzie\\compras_supermercado_correto2.csv")


# In[11]:


compras_1.columns


# In[12]:


# Consolidando itens comprados em uma lista e padronizando

compras_1['Itens Comprados'] = compras_1.apply(
    lambda row: [row[col].lower().strip() for col in compras_1.columns if 'item' in col and row[col].strip() != ''], axis=1)

# Remover as colunas individuais de itens

colunas_para_remover = [col for col in compras_1.columns if 'item' in col]
compras_1.drop(columns=colunas_para_remover, inplace=True)


# In[13]:


compras_1.columns


# In[14]:


# Certificando que todos os itens na coluna 'Ingredientes' são tratados como strings

compras_1['Itens Comprados'] = compras_1['Itens Comprados'].astype(str)


# In[15]:


# Transformar os itens novamente em uma lista

compras_1['Itens Comprados'] = compras_1['Itens Comprados'].apply(lambda x: [ingrediente.lower().strip() for ingrediente in x.split(',')])


# In[16]:


import numpy as np

compras_1['Itens Comprados']


# In[17]:


# Certificando que todos os itens na coluna 'Ingredientes' são tratados como strings

receitas_1['Ingredientes'] = receitas_1['Ingredientes'].astype(str)


# In[18]:


# Transformar os itens novamente em uma lista

receitas_1['Ingredientes'] = receitas_1['Ingredientes'].apply(lambda x: [ingrediente.lower().strip() for ingrediente in x.split(',')])


# In[19]:


receitas_1['Ingredientes']


# In[20]:


# Normalizar os dados

import re

def normalizar_ingredientes(texto):
    # Remove tudo entre parênteses, incluindo os próprios parênteses
    texto = re.sub(r'\(.*?\)', '', texto)
    # Substitui " e " por ", "
    texto = texto.replace(" e ", ", ")
    # Remove caracteres especiais exceto acentos e vírgulas
    texto = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛãõÃÕñÑçÇ,\s]', '', texto)
    # Remove espaços extras
    texto = re.sub(r'\s+', ' ', texto).strip()
    # Remove pontos
    texto = texto.replace('.', '')
    return texto

def normalizar_lista_ingredientes(lista_ingredientes):
    ingredientes_normalizados = [normalizar_ingredientes(ingrediente) for ingrediente in lista_ingredientes]
    return ingredientes_normalizados


# In[21]:


# Aplicando a base de compras

compras_1['Itens Comprados N'] = compras_1['Itens Comprados'].apply(normalizar_lista_ingredientes)


# In[22]:


# Exibindo os resultados normalizados

print(compras_1[['Itens Comprados', 'Itens Comprados N']])


# In[23]:


# Aplicando a base de receitas

receitas_1['Ingredientes N'] = receitas_1['Ingredientes'].apply(normalizar_lista_ingredientes)

# Exibindo os resultados normalizados

print(receitas_1[['Ingredientes', 'Ingredientes N']])


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Criando a coluna 'Ingredientes_Str' a partir dos ingredientes normalizados

receitas_1['Ingredientes_Str'] = receitas_1['Ingredientes N'].apply(lambda x: ' '.join(x))

print(receitas_1.head())


# In[25]:


vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split())

# Assegure-se de que a coluna 'Ingredientes_Str' está presente e corretamente preenchida

vectorizer.fit(receitas_1['Ingredientes_Str'])


# In[26]:


# Definindo vetor de receitas

receitas_vec = vectorizer.transform(receitas_1['Ingredientes_Str'])


# In[27]:


# Definindo vetor de compras

compra_1_str = ' '.join(compras_1['Itens Comprados N'].iloc[0]) 

compras_vec = vectorizer.transform([compra_1_str])  


# In[28]:


# Função para recomendar receitas baseada em uma lista de compras

def recomendar_receitas_para_compra(lista_compras_str):
    
    compra_vec = vectorizer.transform([lista_compras_str])
    
    similaridades = cosine_similarity(compra_vec, receitas_vec)
    
    # Obtendo os índices das receitas com as maiores similaridades
    
    indices_similares = similaridades.argsort()[0][::-1][:3]
     
    # Extrai as recomendações baseadas nos índices obtidos
    
    recomendacoes = [(receitas_1.iloc[indice]['Nome da Receita'], similaridades[0][indice]) for indice in indices_similares]
    
    return recomendacoes


# In[29]:


# Função para imprimir recomendações formatadas para cada lista de compras

def imprimir_recomendacoes(recomendacoes, num_lista):
    print(f"Recomendações para a lista de compras {num_lista}:")
    for nome_receita, _ in recomendacoes:
        print(f"- {nome_receita}")

# Aplicando a recomendação para cada compra e imprimindo os resultados

for indice, compra in compras_1.iterrows():
    lista_compras_str = ' '.join(compra['Itens Comprados N'])
    recomendacoes = recomendar_receitas_para_compra(lista_compras_str)
    imprimir_recomendacoes(recomendacoes, indice + 1) 
    print("\n") 


# In[30]:


# Análise dos resultados. Gráfico de distribuição dos valores de Similaridade
# Convertendo em uma lista plana de valores de similaridade 

similaridades_todas_compras = cosine_similarity(compras_vec, receitas_vec)
valores_similaridade = similaridades_todas_compras.flatten()


# In[31]:


# Gráfico de distribuição

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(valores_similaridade, bins=50, color='skyblue', edgecolor='black')
plt.title('Valores de Similaridade de Cosseno')
plt.xlabel('Similaridade de Cosseno')
plt.ylabel('Frequência')
plt.grid(axis='y', alpha=0.75)

plt.show()


# In[32]:


# Calcular o total de valores de similaridade em cada compra

quantidade_abaixo_0_5 = np.sum(valores_similaridade < 0.5)
quantidade_acima_ou_igual_0_5 = np.sum(valores_similaridade >= 0.5)

# Calcular o total de valores de similaridade 

total_valores = len(valores_similaridade)

# Calcular os %

percentual_abaixo_0_5 = (quantidade_abaixo_0_5 / total_valores) * 100
percentual_acima_ou_igual_0_5 = (quantidade_acima_ou_igual_0_5 / total_valores) * 100

# Resultados

print(f"Percentual de recomendações com similaridade abaixo de 0.5: {percentual_abaixo_0_5:.2f}%")
print(f"Percentual de recomendações com similaridade igual ou acima de 0.5: {percentual_acima_ou_igual_0_5:.2f}%")


# In[33]:


# Definindo um limiar de aceitação para melhoria da relevância das receitas recomendadas

threshold_de_aceitacao = 0.4 


# In[34]:


def recomendar_receitas_para_compra(lista_compras_str, threshold):
    compra_vec = vectorizer.transform([lista_compras_str])
    similaridades = cosine_similarity(compra_vec, receitas_vec)
    
    # Filtro baseado no threshold
    
    indices_similares = [indice for indice in similaridades.argsort()[0][::-1] if similaridades[0][indice] >= threshold]
    
    # As 3 top recomendações que atendem ao threshold
    
    recomendacoes = [(receitas_1.iloc[indice]['Nome da Receita'], similaridades[0][indice]) for indice in indices_similares][:3]
    return recomendacoes


# In[35]:


# Aplicando a função com o threshold para cada compra

compras_1['Recomendações'] = compras_1['Itens Comprados N'].apply(lambda x: ' '.join(x)).apply(lambda x: recomendar_receitas_para_compra(x, threshold_de_aceitacao))

# Exibindo as recomendações filtradas

for indice, recomendacoes in enumerate(compras_1['Recomendações']):
    print(f"Recomendações para a lista de compras {indice+1}:")
    for nome_receita, similaridade in recomendacoes:
        print(f"- {nome_receita} (Similaridade: {similaridade})")
    print("\n")


# In[36]:


# Nova avaliação dos resultados após a aplicação do threshold

def recomendar_receitas_e_coletar_similaridades(lista_compras_str, threshold):
    compra_vec = vectorizer.transform([lista_compras_str])
    similaridades = cosine_similarity(compra_vec, receitas_vec)[0]
    
    # Filtrando os valores de similaridade que atendem ao threshold
    similaridades_filtradas = similaridades[similaridades >= threshold]
    
    # Ordenando as similaridades filtradas e pegando os top N valores
    
    top_similaridades = np.sort(similaridades_filtradas)[::-1][:3]
    
    return top_similaridades

# Coletando os valores de similaridade para todas as listas de compras

valores_similaridade_filtrados = np.concatenate(
    compras_1['Itens Comprados N'].apply(lambda x: ' '.join(x)).apply(lambda x: recomendar_receitas_e_coletar_similaridades(x, threshold_de_aceitacao))
).ravel()


# In[37]:


# Gráfico

plt.figure(figsize=(10, 6))
plt.hist(valores_similaridade_filtrados, bins=30, color='skyblue', edgecolor='black')
plt.title('Valores de Similaridade Após Threshold')
plt.xlabel('Similaridade de Cosseno')
plt.ylabel('Frequência')
plt.grid(axis='y', alpha=0.75)

plt.show()


# In[38]:


# Verificação dos resultados em %

dentro_threshold = np.sum(valores_similaridade_filtrados >= threshold_de_aceitacao) / len(valores_similaridade_filtrados)
abaixo_threshold_acima_0 = np.sum((valores_similaridade_filtrados > 0) & (valores_similaridade_filtrados < threshold_de_aceitacao)) / len(valores_similaridade_filtrados)
abaixo_0 = np.sum(valores_similaridade_filtrados < 0) / len(valores_similaridade_filtrados)

# Convertendo para %

percentual_dentro_threshold = dentro_threshold * 100
percentual_abaixo_threshold_acima_0 = abaixo_threshold_acima_0 * 100
percentual_abaixo_0 = abaixo_0 * 100

# Resultados

print(f"Percentual de receitas dentro do threshold ({threshold_de_aceitacao}): {percentual_dentro_threshold:.2f}%")
print(f"Percentual de receitas com similaridade entre 0 e {threshold_de_aceitacao - 0.01}: {percentual_abaixo_threshold_acima_0:.2f}%")
print(f"Percentual de receitas com similaridade abaixo de 0: {percentual_abaixo_0:.2f}%")


# In[40]:


def recall_at_k(recommended_items, relevant_items, k):
    recommended_items_set = set(recommended_items[:k])
    relevant_items_set = set(relevant_items)
    relevant_and_recommended = recommended_items_set.intersection(relevant_items_set)
    
    if len(relevant_items_set) == 0:
        return 0
    return len(relevant_and_recommended) / len(relevant_items_set)


# In[41]:


# Análise de desempenho

# Atualizar a função recall@k 

def recall_at_k(recommended_items, relevant_items, k):
    recommended_items_set = set(recommended_items[:k])
    relevant_items_set = set(relevant_items)
    relevant_and_recommended = recommended_items_set.intersection(relevant_items_set)
    if len(relevant_items_set) == 0:
        return 0
    return len(relevant_and_recommended) / len(relevant_items_set)

# Métricas

k = 3
precisions = []
recalls = []

for recomendacoes, relevantes in zip(compras_1['Recomendações'], itens_relevantes):
    nomes_recomendados = [rec[0] for rec in recomendacoes]
    precision = precision_at_k(nomes_recomendados, relevantes, k)
    recall = recall_at_k(nomes_recomendados, relevantes, k)
    precisions.append(precision)
    recalls.append(recall)


# In[42]:


# Calcular a média de precisão e recall

average_precision = np.mean(precisions)
average_recall = np.mean(recalls)

print("Average Precision@k:", average_precision)
print("Average Recall@k:", average_recall)


# In[51]:


# A cobertura do catálogo

todos_itens_recomendados = [rec[0] for sublist in compras_1['Recomendações'] for rec in sublist]

itens_unicos_recomendados = set(todos_itens_recomendados)

catalog_size = len(receitas_1) 
catalog_coverage = len(itens_unicos_recomendados) / catalog_size

# Gráfico 
plt.figure(figsize=(6, 4))
plt.bar(['Cobertura do Catálogo'], [catalog_coverage], color='blue', edgecolor='black')
plt.title('Métrica Cobertura do Catálogo')
plt.ylabel('Proporção')
plt.ylim(0, 1)  
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.show()


# In[50]:


print("Cobertura do Catálogo:", catalog_coverage)


# In[52]:


# Visualizando as métricas

import matplotlib.pyplot as plt

# Histograma de Precision@k

plt.figure(figsize=(10, 5))
plt.hist(precisions, bins=20, color='skyblue', edgecolor='black')
plt.title('Métrica Precision@k')
plt.xlabel('Precision@k')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()


# In[53]:


# Histograma de Recall@k

plt.figure(figsize=(10, 5))
plt.hist(recalls, bins=20, color='green', edgecolor='black')
plt.title('Métrica Recall@k')
plt.xlabel('Recall@k')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()


# In[54]:


# Comparando as métricas

plt.figure(figsize=(10, 6))
metrics = ['Precision@k', 'Recall@k', 'Cobertura de Catálogo']
values = [average_precision, average_recall, catalog_coverage]

plt.bar(metrics, values, color=['skyblue', 'green', 'orange'], edgecolor='black')
plt.title('Métricas de Avaliação do Sistema de Recomendação')
plt.ylabel('Valor')
plt.ylim(0, 1)  
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

