# 🧠 Fluxo de treino de uma LLM/SML

## 🗄️ Preparação do Dataset

* Coletar um grande conjunto de dados de texto (e.g. livros, artigos, etc.)
* Preparar o dataset para treinamento (e.g. tokenização, remoção de stop words, normalização de frequência, etc.)
* Dividir o dataset em conjuntos de treinamento, validação e teste

Aqui quero deixar um exemplo de um ficheiro formato json que pedi ao ChatGPT para fazer com dados sobre futebol Português.
Como podemos verificar é possivel usar as já existentes AIs para 'preparar' dados para fazermos o nosso proprio modelo.

```batch
da-me um exemplo de um ficheiro json com dados sobre futebol portugues, não precisa ser muito grande ,
so com 20 linhas , para usar como dataset para um ML
```

```batch
Claro! Aqui está um exemplo de um ficheiro JSON com dados sobre futebol português, adequado para uso como
dataset para Machine Learning:
[
    {
        "nome": "Benfica",
        "cidade": "Lisboa",
        "fundacao": 1904,
        "titulos_liga": 37,
        "estadio": "Estádio da Luz",
        "capacidade": 64642
    },
    {
        "nome": "Porto",
        "cidade": "Porto",
        "fundacao": 1893,
        "titulos_liga": 30,
        "estadio": "Estádio do Dragão",
        "capacidade": 50033
    },
    {
        "nome": "Sporting",
        "cidade": "Lisboa",
        "fundacao": 1906,
        "titulos_liga": 19,
        "estadio": "Estádio José Alvalade",
        "capacidade": 50095
    },
    {
        "nome": "Braga",
        "cidade": "Braga",
        "fundacao": 1921,
        "titulos_liga": 0,
        "estadio": "Estádio Municipal de Braga",
        "capacidade": 30286
    },
    {
        "nome": "Vitória de Guimarães",
        "cidade": "Guimarães",
        "fundacao": 1922,
        "titulos_liga": 0,
        "estadio": "Estádio D. Afonso Henriques",
        "capacidade": 30008
    },
    {
        "nome": "Boavista",
        "cidade": "Porto",
        "fundacao": 1903,
        "titulos_liga": 1,
        "estadio": "Estádio do Bessa",
        "capacidade": 28263
    },
    {
        "nome": "Marítimo",
        "cidade": "Funchal",
        "fundacao": 1910,
        "titulos_liga": 0,
        "estadio": "Estádio dos Barreiros",
        "capacidade": 10932
    },
    {
        "nome": "Belenenses",
        "cidade": "Lisboa",
        "fundacao": 1919,
        "titulos_liga": 1,
        "estadio": "Estádio do Restelo",
        "capacidade": 19500
    },
    {
        "nome": "Paços de Ferreira",
        "cidade": "Paços de Ferreira",
        "fundacao": 1950,
        "titulos_liga": 0,
        "estadio": "Estádio da Mata Real",
        "capacidade": 9125
    },
    {
        "nome": "Rio Ave",
        "cidade": "Vila do Conde",
        "fundacao": 1939,
        "titulos_liga": 0,
        "estadio": "Estádio dos Arcos",
        "capacidade": 12815
    }
]
Este exemplo contém informações básicas sobre 10 clubes de futebol portugueses, incluindo nome,
cidade, ano de fundação, títulos da liga, estádio e capacidade do estádio. É um dataset simples
mas útil para começar a explorar modelos de Machine Learning.
```


## 🗃️ Pré-Processamento de Dados

* Tokenização: dividir o texto em palavras ou tokens (e.g. utilizando o algoritmo WordPiece)
* Remoção de stop words: remover palavras comuns como "the", "and", etc.
* Normalização: normalizar a frequência de palavras e tokens (e.g. utilizando o algoritmo TF-IDF)
* Conversão de texto para formato de entrada para o modelo (e.g. sequência de tokens)

## 🧮 Escolha do Algoritmo ML

* Selecionar um algoritmo de aprendizado de máquina adequado para o problema (e.g. BERT, RoBERTa, etc.)
* Configurar os parâmetros do algoritmo (e.g. número de camadas, número de neurônios, etc.)

## 🏋️ Etapas de Treinamento

* Treinamento: repetir as etapas de treino várias vezes, ajustando os parâmetros do modelo para minimizar a perda
* Validação: avaliar o desempenho do modelo no conjunto de validação para evitar overfitting
* Teste: avaliar o desempenho do modelo no conjunto de teste para avaliar sua generalização

## 🏋️ Etapas no Treino

1. **Forward Pass**: o modelo processa o input (e.g. uma frase) e produz uma saída (e.g. uma sequência de tokens)
2. **Cálculo da Perda**: o modelo calcula a diferença entre a saída prevista e a saída real (e.g. a perda de cross-entropy)
3. **Backward Pass**: o modelo ajusta seus parâmetros internos para reduzir a perda (e.g. ajustando os pesos e bias da rede neural)
4. **Atualização dos Parâmetros**: o modelo atualiza seus parâmetros internos com base na diferença entre a saída prevista e a saída real

## 💾 Ficheiros finais treinados

* O modelo treinado produz ficheiros de saída que podem ser usados para:
	+ Gerar texto com base em um input (e.g. um prompt)
	+ Realizar tarefas de processamento de linguagem natural (e.g. classificação de texto, geração de resumo, etc.)
	+ Treinar modelos adicionais para tarefas específicas

## ❓ Como usar esses 'ficheiros'

Existem várias opções para alojar os ficheiros treinados de saída para uso como API. Aqui estão algumas opções:

1. **Servidor próprio**: você pode hospedar os ficheiros em um servidor próprio, utilizando tecnologias como Apache, Nginx ou IIS. Isso permite que você tenha controle total sobre a infraestrutura e a segurança.
2. **Servidores de nuvem**: você pode utilizar serviços de nuvem como AWS S3, Google Cloud Storage ou Microsoft Azure Blob Storage para armazenar e servir os ficheiros. Esses serviços oferecem escalabilidade, segurança e facilidade de uso.
3. **Plataformas de API**: você pode utilizar plataformas de API como AWS API Gateway, Google Cloud Endpoints ou Microsoft Azure API Management para criar e gerenciar APIs. Essas plataformas oferecem recursos como autenticação, autorização e monitoramento.
4. **Serviços de armazenamento de dados**: você pode utilizar serviços de armazenamento de dados como MongoDB, Cassandra ou PostgreSQL para armazenar e servir os ficheiros. Esses serviços oferecem escalabilidade, segurança e facilidade de uso.
5. **Serviços de computação em nuvem**: você pode utilizar serviços de computação em nuvem como AWS Lambda, Google Cloud Functions ou Microsoft Azure Functions para executar o modelo de linguagem e servir os resultados como API. Esses serviços oferecem escalabilidade, segurança e facilidade de uso.

### Quando escolher uma opção, considere fatores como:

* Necessidade de recursos (harware) proderosos para ter uma boa resposta! Modelos que usam 4 biliões de parametros* (LLM) em treino precisam de um CPU Intel Core i9-11900K ou um AMD Ryzen 9 5900X, 32Gb Ram e  NVIDIA GeForce RTX 3080 ou um AMD Radeon RX 6800 XT
* Escalabilidade: como o seu modelo de linguagem irá crescer, você precisará de uma solução que possa lidar com o aumento do tráfego e do armazenamento de dados.
* Segurança: você precisará garantir que os ficheiros treinados de saída sejam protegidos contra acesso não autorizado e que a API seja segura para uso.
* Custos: você precisará considerar os custos associados à escolha da opção, incluindo custos de armazenamento, processamento e transferência de dados.
* Facilidade de uso: você precisará considerar a facilidade de uso e a documentação da opção escolhida.

> Os parâmetros de um modelo de linguagem são as variáveis que são ajustadas durante o treinamento do modelo para melhorar sua capacidade de prever a saída correta para uma entrada. Mais parâmetros significam que o modelo tem mais capacidade para aprender e generalizar a partir dos dados de treinamento.
> 100M a 1B (100 milhões a 1 bilhão) de parâmetros: Esses são os parâmetros típicos de um SLM que é treinado para tarefas de processamento de linguagem natural, como classificação de texto, sentiment analysis e geração de texto.
10M a 100M (10 milhões a 100 milhões) de parâmetros: Esses são os parâmetros típicos de um SLM que é treinado para tarefas mais específicas, como reconhecimento de entidade ou geração de resumo.
1M a 10M (1 milhão a 10 milhões) de parâmetros: Esses são os parâmetros típicos de um SLM que é treinado para tarefas mais simples, como classificação de texto ou sentiment analysis.

###  Hugging Face 

é um serviço que fornece uma plataforma para desenvolvedores e cientistas de dados trabalhar com modelos de linguagem e outros tipos de modelos de aprendizado de máquina. Eles oferecem uma variedade de recursos, incluindo:

* Um repositório de modelos de linguagem pré-treinados, incluindo modelos como BERT, RoBERTa e XLNet.
* Uma plataforma para treinar e testar modelos de linguagem.
* Uma API para integrar modelos de linguagem em aplicativos e serviços.
* Ferramentas para processamento de linguagem natural, incluindo tokenização, paralelismo e geração de texto.

Hugging Face também fornece uma variedade de serviços, incluindo:

* Hugging Face Hub: um repositório de modelos de linguagem pré-treinados e treinados.
* Hugging Face Transformers: uma biblioteca de transformadores de linguagem natural.
* Hugging Face Datasets: uma plataforma para armazenar e compartilhar conjuntos de dados de linguagem natural.

Em resumo, Hugging Face é um serviço que fornece uma plataforma para desenvolvedores e cientistas de dados trabalhar com modelos de linguagem e outros tipos de modelos de aprendizado de máquina. Eles oferecem uma variedade de recursos e serviços para ajudar a desenvolver e implementar aplicativos e serviços que utilizam modelos de linguagem.
