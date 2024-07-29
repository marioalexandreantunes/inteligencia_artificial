# 🖥 Processo de criação de uma Large/Small Language Model (LLM/SLM) do zero.

Este artigo mergulha no mundo da Inteligência Artificial (IA) e modelos de linguagem de grande escala (LLMs/SLMs), como o famoso GPT da OpenAI. 

LLMs/SLMs aprendem a imitar funções cognitivas humanas, identificando e replicando padrões em dados de texto.  Eles utilizam uma estrutura inovadora chamada "transformers" para treinar com alta eficiência, prevendo a próxima palavra em uma sequência com precisão impressionante.

O artigo explora três tipos principais de arquiteturas Transformer:

* **Encoder:** Para tarefas de compreensão de texto (ex: BERT, RoBERTa).
* **Decoder:** Para geração de texto (ex: GPT).
* **Encoder-Decoder:** Combinando ambas as funcionalidades (ex: T5).

Através de exemplo prático, aprenderá a construir o seu próprio modelo de linguagem com Transformers. 

---

# 📄 Pré-Processamento de Dados

O pré-processamento de dados para modelos de linguagem de grande escala (LLMs/SLMs) envolve várias etapas detalhadas

1. Os dados são divididos em pequenas unidades chamadas tokens, que podem ser palavras, subpalavras, números ou símbolos.
2. Os dados são limpos para remover erros, conteúdo ofensivo ou spam. O texto é então normalizado, o que pode incluir a conversão de todas as letras para minúsculas, remoção de stopwords, e aplicação de técnicas de stemming ou lematização.
3. Esses 'tokens' são transformados em números através de técnicas como embedding, para que o modelo possa processá-los eficientemente.

> Um **token** é uma unidade básica de linguagem em um modelo de linguagem geradora, como GPT-2 ou BERT. É uma palavra ou um símbolo que representa uma ideia ou uma informação específica. Por exemplo, "O gato" é um token, e "gato" é um subtoken.

> Uma técnica de **embedding** é uma abordagem utilizada em machine learning para mapear objetos de diferentes domínios para um espaço vetorial comum. Em outras palavras, ela permite que os modelos de linguagem geradora representem informações de diferentes fontes de dados em um formato comum e fácil de processar.

Lembre-se de que a criação de um **dataset** é uma tarefa desafiadora e requer tempo e esforço. No entanto, com certeza e dedicação, você pode criar um dataset útil para o treinamento do modelo de linguagem ou para outras tarefas específicas.
Um **dataset** pode ser uma forma de tabela de perguntas e respostas e em forma JSON e deverá ser em inglês para obter melhores resultados.

Verifica se já não há um dataset que possas melhorar e usar, https://huggingface.co/datasets

Exemplo ficheiro json com tema de futebol:
```
[
{
    "question": "What are the dimensions of a regulation soccer field?",
    "answer": "A regulation soccer field is 100-130 yards long and 50-100 yards wide."
  },
  {
    "question": "How many players are on the field for each team in a soccer match?",
    "answer": "Each team has 11 players on the field at a time, including the goalkeeper."
  },
  {
    "question": "What is an offside rule in soccer?",
    "answer": "A player is offside if they are closer to the opponent's goal line than both the ball and the second-to-last defender when the ball is played to them."
  },
  {
    "question": "What is a penalty kick in soccer?",
    "answer": "A penalty kick is awarded to the attacking team when a foul is committed by a defender inside the penalty area."
  },
  {
    "question": "What are the different positions in a soccer team?",
    "answer": "Common positions include goalkeeper, defenders (center-backs, full-backs), midfielders (central midfielders, wingers), and forwards (strikers)."
  }
]
```
---
# 🧠 Machine Learning

Para treinar um LLM/SLM, é necessário equipamento especializado, como GPUs ou TPUs de alto desempenho, para processar grandes volumes de dados e realizar cálculos complexos. Neste exemplo, utilizarei o **Google Colab**, que oferece acesso gratuito a GPUs e TPUs , possibilitando um treinamento sem a necessidade de investir em hardware caro.

> Um arquivo `.ipynb` no Google Colab é um arquivo de notebook Jupyter. O Google Colab é uma plataforma gratuita e em nuvem que permite a execução de código Python, R e Julia em um ambiente de notebook interativo. Os arquivos `.ipynb` são usados para armazenar e compartilhar códigos Jupyter, que são documentos que contêm código, texto e visualizações.

## 🔨 Crie um Google Colab e habilite a T4 GPU

Aqui estão os passos para criar um Google Colab e habilitar a GPU T4:

### 1. Acesse o Google Colab:**

* Vá para o site do Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)

### 2. Crie um novo Notebook:**

* Clique no botão "Novo" para criar um novo notebook.

### 3. Habilite a GPU T4:**

* **Clique no menu "Runtime" no topo da tela.**
* **Selecione "Change runtime type".**
* **Na janela que aparece, escolha "GPU" como o "Hardware accelerator".**
* **Selecione "Tesla T4" como o tipo de GPU.**
* **Clique em "Save".**

### 4. Verifique se a GPU está habilitada:**

* Execute o seguinte código no seu notebook:

```python
  !nvidia-smi
```

* Isso exibirá informações sobre a GPU T4 habilitada, incluindo o nome do modelo, memória disponível e uso da GPU.

**Observações:**

* A disponibilidade de GPUs T4 pode variar dependendo da demanda. Se você não conseguir encontrar a opção T4, tente escolher outra GPU disponível, como a Tesla K80.
* O uso de GPUs pode resultar em custos adicionais, dependendo do tempo de execução e da quantidade de recursos utilizados.


## 🔨 Instalação das bibliotecas necessárias

```batch
  pip install transformers[torch] datasets torch
```

A biblioteca transformers da Hugging Face oferece modelos de linguagem pré-treinados como GPT-2, BERT e T5 para tarefas como geração de texto, resposta a perguntas e tradução. É fácil de usar e permite treinar ou ajustar modelos para necessidades específicas. É uma ferramenta essencial para aplicar inteligência artificial em linguagem natural. A biblioteca torch é usada para computação e treinamento eficiente, e a biblioteca datasets é usada para manipular nossos dados de treino.

## 🔨 Importação das bibliotecas

Importação das bibliotecas
```python
  from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
  from datasets import Dataset
  import json
  import pandas as pd
```

Neste passo, importamos as bibliotecas essenciais para manipulação de dados e deep learning. Utilizamos o torch para operações de tensor e computação em GPU.
A biblioteca transformers para carregar e treinar o modelo GPT-2, incluindo GPT2Tokenizer, GPT2LMHeadModel, Trainer e TrainingArguments.
O datasets do Hugging Face para preparar e manipular conjuntos de dados.
A biblioteca padrão json para leitura de arquivos JSON que é o nosso ficheiro de dados, e o pandas para transformar dados JSON em DataFrames para análise e manipulação eficientes. Essas bibliotecas são fundamentais para realizar tarefas de processamento de linguagem natural (NLP) de maneira eficaz.

## 🔨 Carregar dados do arquivo JSON
```python
  def load_data(file_path):
    with open(file_path, 'r') as file:
    data = json.load(file)
    return pd.DataFrame(data)
  
  file_path = '/content/dataset.json'
  df = load_data(file_path)
```

No colab ao lado esquerdo folder icon, '/content' é o root, carrega o ficheiro para lá!

O objetivo desta etapa é carregar os dados contidos em um arquivo JSON e convertê-los em um DataFrame do Pandas. 
Um DataFrame é uma estrutura de dados bidimensional que facilita a manipulação e análise dos dados.
A Criação do JSON que contem perguntas e respostas podes criar usando uma AI. Utiliza uma série de prompts para gerar JSONs contendo perguntas e respostas sobre uma ampla gama de tópicos que desejas treinar o teu modelo. 
Para garantir a diversidade das informações, usa [crewAI](https://www.crewai.com/) onde terás ligação a modelos OpenGPT, Gemini, Claude etc
Verifica depois se tens ou não perguntas repetidas! É muito importante não ter perguntas repetidas. Possivelmente terás de criar um código/script que faça isso automaticamente.
Nesta fase é onde terás de ler e aprender a ter os dados necessários e correctos. O Fine Tuning usará também datasets!

## 🔨 Inicializar tokenizador e preparar dados  

```python
	tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
	tokenizer.pad_token = tokenizer.eos_token

	df['text'] = df['question'] + " " + df['answer']

	def tokenize_function(examples):
    		return tokenizer(examples, padding="max_length", truncation=True, max_length=512)

	tokenized_datasets = df['text'].apply(lambda x: tokenize_function(x))
	dataset = Dataset.from_pandas(df)

	def tokenize_dataset(dataset):
    		return dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=512), batched=True, remove_columns=["text", "question", "answer"])

	tokenized_dataset = tokenize_dataset(dataset)
```

**[distilgpt2](https://huggingface.co/distilbert/distilgpt2)** é um modelo de língua **inglesa pré-treinado** gerador de texto baseado no modelo GPT-2 (Generative Pre-trained Transformer 2), que é uma das melhores opções para a geração de texto natural. Ele foi projetado para ser mais eficiente e fácil de usar do que o modelo original, tornando-o uma opção popular entre os desenvolvedores que buscam treinar modelos de linguagem Decoders (GPT). Exitem outros modelos como BART, ELECTRA ou T5.

Neste passo, inicializamos o tokenizador GPT2Tokenizer do modelo **distilgpt2** e configuramos o token de padding para ser o mesmo que o token de fim de sequência (EOS). Concatenamos perguntas e respostas em uma nova coluna text no DataFrame e definimos uma função de tokenização que aplica padding e truncamento até um comprimento máximo de 512 tokens. Aplicamos esta função de tokenização a cada texto no DataFrame, convertendo-o em tokens. Em seguida, transformamos o DataFrame em um Dataset do Hugging Face, facilitando o processamento eficiente e removendo colunas originais após a tokenização para otimização.
O token de padding é utilizado para garantir que todas as sequências de entrada em um lote (batch) de dados tenham o mesmo comprimento. Isso é necessário porque os modelos de deep learning, como os Transformers, requerem que as entradas tenham dimensões consistentes para processamento eficiente em paralelo.

## 🔨 Adicionar labels e dividir dataset

codigo completo :
```python
	tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)

	train_test_split = tokenized_dataset.train_test_split(test_size=0.15)
	train_dataset = train_test_split['train']
	test_dataset = train_test_split['test']
```

## 🔨 Explicação detalhada da etapa de adição de labels e divisão do dataset:

Este passo é fundamental para preparar os dados para o treinamento do modelo de linguagem. 

### 1. Adicionando Labels:

```python
tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
```

* **Objetivo:** Atribuir labels aos dados tokenizados. 
* **Como funciona:**
    * `tokenized_dataset.map`: Aplica uma função a cada exemplo do dataset.
    * `lambda examples: {'labels': examples['input_ids']}`:  Função que cria um novo dicionário para cada exemplo, adicionando uma chave 'labels' com o valor de 'input_ids'.  Os 'input_ids' representam as representações numéricas das palavras no texto, que serão usadas como labels para o modelo aprender a prever as próximas palavras.
    * `batched=True`: Processa os exemplos em batches para otimizar o desempenho.

### 2. Dividindo o Dataset:

```python
train_test_split = tokenized_dataset.train_test_split(test_size=0.15)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
```

* **Objetivo:** Separar o dataset em conjuntos de treino e teste.
* **Como funciona:**
    * `tokenized_dataset.train_test_split(test_size=0.15)`: Divide o dataset em 85% para treino e 15% para teste.
    * `train_dataset = train_test_split['train']`: Atribui o conjunto de treino à variável `train_dataset`.
    * `test_dataset = train_test_split['test']`: Atribui o conjunto de teste à variável `test_dataset`.

**Importância da Divisão:**

* **Prevenção de Overfitting:** O modelo aprende melhor com dados que ele não viu durante o treinamento. O conjunto de teste garante que o modelo seja avaliado com dados não vistos, evitando que ele memorize o conjunto de treino e se torne incapaz de generalizar para novos dados.
* **Avaliação do Desempenho:** O conjunto de teste permite avaliar o desempenho do modelo em dados reais e comparar diferentes modelos.


## 🔨 Função de agrupamento de dados

```python
	def data_collator(features):
    		batch = {}
    		batch['input_ids'] = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    		batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    		batch['labels'] = torch.tensor([f['labels'] for f in features], dtype=torch.long)
    		return batch
```

## 🔨 A função `data_collator`:  Organizando os dados para o treinamento eficiente

A função `data_collator` organiza dados tokenizados em batches para treinamento eficiente de modelos de linguagem. Ela converte listas de tokens, máscaras de atenção e rótulos em tensores PyTorch, garantindo que todos os batches tenham a mesma estrutura através de técnicas de padding. Essa organização facilita o processamento computacional, garante consistência no treinamento e permite uma integração simplificada com o Hugging Face Trainer. 

> Tensores PyTorch são estruturas de dados fundamentais similares aos arrays multidimensionais mas com a vantagem adicional de serem otimizados para operações de alto desempenho em GPUs (unidades de processamento gráfico).

## 🔨 Inicializar modelo e argumentos de treinamento

```python
	model = GPT2LMHeadModel.from_pretrained('distilgpt2')
	
	training_args = TrainingArguments(
	    output_dir='./results',
	    eval_strategy="epoch",
	    learning_rate=3e-5,
	    per_device_train_batch_size=8,
	    per_device_eval_batch_size=8,
	    num_train_epochs=5,
	    weight_decay=0.01,
	    logging_dir='./logs',
	    logging_steps=10,
	)
```

Neste passo, preparamos o ambiente para o treinamento do modelo GPT-2. 

* **Carregamos o modelo:** Utilizamos `GPT2LMHeadModel.from_pretrained('distilgpt2')` para carregar uma versão otimizada e menor do GPT-2, chamada distilgpt2.

* **Configuramos o treinamento:** Definimos os parâmetros de treinamento usando `TrainingArguments`, controlando aspectos como:
    * **Saída:** Diretório para salvar os resultados do treinamento.
    * **Avaliação:** Estratégia para avaliar o desempenho do modelo.
    * **Taxa de aprendizado:**  Velocidade de ajuste dos parâmetros do modelo.
    * **Tamanho do batch:** Quantidade de dados processados por vez.
    * **Número de épocas:**  Número de vezes que todo o dataset é usado para treinamento.
    * **Decaimento de peso:**  Redução gradual da taxa de aprendizado durante o treinamento.
    * **Logs:** Diretório para salvar informações sobre o progresso do treinamento.
    * **Frequência de registro:**  Intervalo de tempo para registrar informações sobre o treinamento.

* **Iniciando o treinamento:**  O `Trainer` utiliza esses argumentos, juntamente com os datasets e a função de agrupamento de dados (`data_collator`), para gerenciar o processo de treinamento e avaliação do modelo de forma eficiente e controlada.









