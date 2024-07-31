# 🖥 Processo de criação de uma Large/Small Language Model (LLM/SLM) do zero.

Este artigo mergulha no mundo da Inteligência Artificial (AI) e modelos de linguagem de grande escala (LLMs/SLMs), como o famoso GPT da OpenAI. 

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

> [!NOTE]
> Um **token** é uma unidade básica de linguagem em um modelo de linguagem geradora, como GPT-2 ou BERT. É uma palavra ou um símbolo que representa uma ideia ou uma informação específica. Por exemplo, "O gato" é um token, e "gato" é um subtoken.

> [!NOTE]
> Uma técnica de **embedding** é uma abordagem utilizada em machine learning para mapear objetos de diferentes domínios para um espaço vetorial comum. Em outras palavras, ela permite que os modelos de linguagem geradora representem informações de diferentes fontes de dados em um formato comum e fácil de processar.

Lembre-se de que a criação de um **dataset** é uma tarefa desafiadora e requer tempo e esforço. No entanto, com certeza e dedicação, você pode criar um dataset útil para o treinamento do modelo de linguagem ou para outras tarefas específicas.
Um **dataset** pode ser uma forma de tabela de perguntas e respostas e em forma JSON e deverá ser em inglês para obter melhores resultados.

Verifica se já não há um dataset que possas melhorar e usar, https://huggingface.co/datasets

👉 [YouTube](https://www.youtube.com/results?search_query=como+fazer+datasets+para+llm) mais videos sobre datasets

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

### 1. Acesse o Google Colab:

* Vá para o site do Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)

### 2. Crie um novo Notebook:

* Clique no botão "Novo" para criar um novo notebook.

### 3. Habilite a GPU T4:

* **Clique no menu "Runtime" no topo da tela.**
* **Selecione "Change runtime type".**
* **Na janela que aparece, escolha "GPU" como o "Hardware accelerator".**
* **Selecione "Tesla T4" como o tipo de GPU.**
* **Clique em "Save".**

### 4. Verifique se a GPU está habilitada:

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

Aqui está um resumo dos modelos mais populares da biblioteca Transformers da Hugging Face:

1. BERT (Bidirectional Encoder Representations from Transformers)
	- **Descrição**: Um modelo pré-treinado bidirecional para NLP, especializado em compreensão contextual.
	- **Usos**: Classificação de texto, resposta a perguntas, reconhecimento de entidades nomeadas.
	- **Exemplo**: `bert-base-uncased`

2. GPT (Generative Pre-trained Transformer)
	- **Descrição**: Um modelo de geração de texto autoregressivo.
	- **Usos**: Geração de texto, completamento de texto, chatbots.
	- **Exemplo**: `gpt2`, `gpt-3.5-turbo`

3. RoBERTa (A Robustly Optimized BERT Pretraining Approach)
	- **Descrição**: Uma variante de BERT otimizada com mais dados e ajustes de hiperparâmetros.
	- **Usos**: Tarefas semelhantes ao BERT, com melhor desempenho.
	- **Exemplo**: `roberta-base`

4. DistilBERT (Distilled version of BERT)
	- **Descrição**: Uma versão compacta de BERT que é mais rápida e eficiente.
	- **Usos**: Tarefas de NLP com menor necessidade computacional.
	- **Exemplo**: `distilbert-base-uncased`

5. T5 (Text-To-Text Transfer Transformer)
	- **Descrição**: Um modelo que trata todas as tarefas de NLP como problemas de tradução de texto-para-texto.
	- **Usos**: Tradução, sumarização, geração de texto.
	- **Exemplo**: `t5-small`, `t5-base`

6. XLNet
	- **Descrição**: Um modelo autoregressivo que também captura dependências bidirecionais.
	- **Usos**: Modelos de linguagem com predição bidirecional.
	- **Exemplo**: `xlnet-base-cased`

7. ALBERT (A Lite BERT)
	- **Descrição**: Uma versão leve de BERT com menos parâmetros e arquitetura eficiente.
	- **Usos**: Tarefas de NLP com menor consumo de memória e mais rápidas.
	- **Exemplo**: `albert-base-v2`

8. Bart (Bidirectional and Auto-Regressive Transformers)
	- **Descrição**: Um modelo que combina as capacidades de modelos bidirecionais e autoregressivos.
	- **Usos**: Tradução, sumarização, geração de texto.
	- **Exemplo**: `facebook/bart-large`

9. Electra (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)
	- **Descrição**: Um modelo eficiente que aprende a discriminar entre tokens reais e substituídos.
	- **Usos**: Pré-treino eficiente para tarefas de NLP.
	- **Exemplo**: `google/electra-small-discriminator`

10. BERTweet
	- **Descrição**: Um modelo baseado em BERT treinado especificamente em dados do Twitter.
	- **Usos**: Análise de sentimentos, classificação de texto em tweets.
	- **Exemplo**: `vinai/bertweet-base`

## 🔨 Importação das bibliotecas

Importação das bibliotecas
```python
	  from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
	  from datasets import Dataset
	  import json
	  import pandas as pd
```

[LINK](https://huggingface.co/docs/transformers/v4.43.3/en/model_doc/gpt2)

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

A biblioteca Transformers da Hugging Face oferece uma ampla variedade de modelos pré-treinados que você pode utilizar com o `GPT2Tokenizer.from_pretrained`. Aqui estão **algumas** opções:

1. **`gpt2`** : O modelo original GPT-2, treinado em um conjunto de dados de 45 GB de texto.
2. **`gpt2-medium`** : Uma versão média do modelo GPT-2, treinada em um conjunto de dados de 10 GB de texto.
3. **`gpt2-large`** : Uma versão grande do modelo GPT-2, treinada em um conjunto de dados de 300 GB de texto.
4. **`gpt2-xl`** : Uma versão extra grande do modelo GPT-2, treinada em um conjunto de dados de 1,2 TB de texto.
5. **`gpt2-125M`** : Uma versão do modelo GPT-2 com 125 milhões de parâmetros, treinada em um conjunto de dados de 10 GB de texto.
6. **`gpt2-355M`** : Uma versão do modelo GPT-2 com 355 milhões de parâmetros, treinada em um conjunto de dados de 10 GB de texto.
7. **`gpt2-774M`** : Uma versão do modelo GPT-2 com 774 milhões de parâmetros, treinada em um conjunto de dados de 10 GB de texto.
8. **`gpt2-125M-uncased`** : Uma versão do modelo GPT-2 com 125 milhões de parâmetros, treinada em um conjunto de dados de 10 GB de texto e sem case sensitivity.
9. **`gpt2-355M-uncased`** : Uma versão do modelo GPT-2 com 355 milhões de parâmetros, treinada em um conjunto de dados de 10 GB de texto e sem case sensitivity.
10. **`gpt2-774M-uncased`** : Uma versão do modelo GPT-2 com 774 milhões de parâmetros, treinada em um conjunto de dados de 10 GB de texto e sem case sensitivity.

Lembre-se de que cada modelo pré-treinado tem suas próprias características e habilidades, e você deve escolher o modelo que melhor se adequare às suas necessidades específicas.

**[distilgpt2](https://huggingface.co/distilbert/distilgpt2)** é um modelo de língua **inglesa pré-treinado** gerador de texto baseado no modelo GPT-2 (Generative Pre-trained Transformer 2), que é uma das melhores opções para a geração de texto natural. Ele foi projetado para ser mais eficiente e fácil de usar do que o modelo original, tornando-o uma opção popular entre os desenvolvedores que buscam treinar modelos de linguagem Decoders (GPT).

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

> [!NOTE]
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

Quando você executa o comando `model = GPT2LMHeadModel.from_pretrained('distilgpt2')`, você está carregando o modelo de linguagem GPT-2, que é uma rede neural treinada para prever as próximas palavras em uma sequência de texto.

Aqui estão alguns dos parâmetros do modelo GPT-2:

* **Número de camadas**: 12
* **Número de neurônios por camada**: 768
* **Tamanho da janela de atenção**: 512
* **Número de heads de atenção**: 12
* **Taxa de aprendizado**: 1e-4
* **Número de épocas de treinamento**: 30
* **Tamanho do batch**: 32
* **Regularização**: L2 regularization com um peso de 0.1
* **Dropout**: 0.1

É importante notar que esses parâmetros são os padrões utilizados durante o treinamento do modelo GPT-2. Você pode ajustar esses parâmetros para melhorar o desempenho do modelo em seu problema específico.

Se você quiser saber mais sobre os parâmetros do modelo GPT-2, recomendo que você leia o artigo original sobre o modelo GPT-2, publicado na arXiv em 2019.

* **Configuramos o treinamento:** Definimos os parâmetros de treinamento usando `TrainingArguments`, controlando aspectos como:
    * **Saída:** Diretório para salvar os resultados do treinamento.
    * **Avaliação:** Estratégia para avaliar o desempenho do modelo.
    * **Taxa de aprendizado:**  Velocidade de ajuste dos parâmetros do modelo.
    * **Tamanho do batch:** Quantidade de dados processados por vez.
    * **Número de passagens:**  Número de vezes que todo o dataset é usado para treinamento.
    * **Decaimento de peso:**  Redução gradual da taxa de aprendisagem durante o treinamento.
    * **Logs:** Diretório para salvar informações sobre o progresso do treinamento.
    * **Frequência de registro:**  Intervalo de tempo para registrar informações sobre o treinamento.

Alguns frameworks de machine learning, como TensorFlow e PyTorch, oferecem suporte à **quantização** de pesos e ativamentos em modelos de Transformer.
A quantização é uma técnica usada para reduzir a precisão dos números que representam os pesos e ativamentos de uma rede neural. Em vez de usar números de 32 bits (como é comum em muitas implementações), os valores são representados com menos bits, como 8 bits ou até 4 bits. Isso tem várias vantagens:

1. Redução do tamanho do modelo: Um modelo quantizado ocupa menos espaço na memória, o que é útil para implementações em dispositivos com recursos limitados, como smartphones ou dispositivos IoT.
2. Melhoria na velocidade de inferência: Operações com números de menor precisão podem ser computadas mais rapidamente, o que acelera o tempo de inferência.
3. Menor consumo de energia: Modelos quantizados consomem menos energia, o que é importante para dispositivos móveis e outros sistemas embarcados.

No Hugging Face e outras plataformas de modelagem, você pode ver modelos quantizados em 8 bits ou 4 bits usados para tarefas como processamento de linguagem natural, visão computacional, entre outros. Estes modelos são frequentemente utilizados em cenários onde a eficiência e a velocidade são cruciais, e onde os dispositivos de execução têm limitações de memória e poder de processamento.

* **Iniciando o treinamento:**  O `Trainer` utiliza esses argumentos, juntamente com os datasets e a função de agrupamento de dados (`data_collator`), para gerenciar o processo de treinamento e avaliação do modelo de forma eficiente e controlada.

## 🔨 Configurar o trainer e iniciar o treinamento

```python
	trainer = Trainer(
	    model=model,
	    args=training_args,
	    train_dataset=train_dataset,
	    eval_dataset=test_dataset,
	    data_collator=data_collator
	)
	
	trainer.train()
```

Este passo envolve a configuração do objeto Trainer da biblioteca transformers.

   - model=model: Especifica o modelo de aprendizado de máquina que será treinado. Esse modelo já deve estar previamente carregado e configurado que será **distilgpt2**
   - args=training_args: Configurações e parâmetros de treinamento, geralmente um objeto da classe TrainingArguments. Isso pode incluir informações como número de épocas, tamanhos de lote (batch size), taxas de aprendizado e dispositivos a serem usados (CPU/GPU).
   - train_dataset=train_dataset: O conjunto de dados que será usado para treinar o modelo.
   - eval_dataset=test_dataset: O conjunto de dados que será usado para avaliar o desempenho do modelo durante o treinamento.
   - data_collator=data_collator: Um objeto que identifica como os dados devem ser agrupados em lotes (batches) durante o treinamento e a avaliação.

Após a configuração, o treinamento é iniciado com `trainer.train()`.

Etapas de treino são forward pass >> cálculo da perda >> backward pass >> atualização dos parâmetros

> [!NOTE]
> O forward pass é a etapa em que os dados de entrada são passados pela rede neural, camada por camada, até que uma previsão (ou saída) seja gerada. Ele transforma inputs em outputs. Imagine que você está fornecendo ao modelo uma frase, como "O gato está dormindo". O modelo lê a frase e tenta prever a próxima palavra na frase, com base nas palavras que viu antes. Isso é chamado de **forward pass** porque o modelo está se movendo para frente, processando a frase de entrada e fazendo previsões.

> [!NOTE]
> O cálculo da perda quantifica o erro das predições da rede comparado aos valores reais, utilizando funções de perda específicas. Este valor é crucial para ajustar os pesos da rede e melhorar a precisão do modelo, mede o quão distante as predições da rede estão dos valores reais. Se o modelo prevê a palavra correta, a perda é baixa. Se ele prevê uma palavra errada, a perda é alta. O objetivo é minimizar a perda, o que significa que o modelo está melhorando para prever a próxima palavra.

> [!NOTE]
> O backward pass é um passo importante no treinamento de modelos de inteligência artificial. Nesse passo, **backward pass** é o oposto do **forward pass**. Em vez de se mover para frente, o modelo se move para trás, ajustando seus parâmetros internos para reduzir a perda. Isso é como o modelo dizendo: "Ah, eu errei! Vou tentar novamente e farei melhor!"

> [!NOTE]
> No Atualização dos Parâmetros durante o backward pass, o modelo atualiza seus parâmetros internos, como pesos e bias de sua rede neural. Esses parâmetros são ajustados com base na diferença entre a saída prevista e a saída real. O objetivo é encontrar o conjunto ótimo de parâmetros que minimize a perda.

Após o terminus do treinamento irá aparecer na consola:
```
	TrainOutput(global_step=800, training_loss=0.2628884120285511, 
	metrics={'train_runtime': 692.0873, 'train_samples_per_second': 9.211, 
	'train_steps_per_second': 1.156, 'total_flos': 832883392512000.0, 
	'train_loss': 0.2628884120285511, 'epoch': 5.0})
```
Ele fornece informações sobre o estado do treinamento, incluindo:

* O passo global do treinamento (global_step): 800
* A perda de treinamento (training_loss): 0.2628884120285511
* Métricas de desempenho do treinamento, incluindo:
	+ Tempo de execução do treinamento (train_runtime): 692.0873 segundos
	+ Número de amostras por segundo (train_samples_per_second): 9.211
	+ Número de passos por segundo (train_steps_per_second): 1.156
	+ Número total de operações floating-point (total_flos): 832883392512000.0
	+ Perda de treinamento (train_loss): 0.2628884120285511
	+ Época atual (epoch): 5.0

Essas informações podem ser úteis para monitorar o progresso do treinamento e ajustar os parâmetros do modelo para melhorar o desempenho.

## 🔨 Salvar o modelo e tokenizador treinados

```python
	model.save_pretrained("./gpt2-chatbot")
	tokenizer.save_pretrained("./gpt2-chatbot")
```

Após o treinamento do modelo, é crucial salvar tanto o modelo quanto o tokenizador para reutilização futura. Isso é feito utilizando os métodos save_pretrained do GPT2LMHeadModel e GPT2Tokenizer, que armazenam os pesos treinados, configurações, e vocabulário em um diretório especificado, como ./gpt2-chatbot. Salvar esses componentes permite carregá-los posteriormente com from_pretrained, evitando a necessidade de retraining, facilitando o compartilhamento, backup e versionamento do modelo, garantindo eficiência e consistência nas inferências futuras.

## 🔨 Carregar modelo e tokenizador treinados

```python
	model = GPT2LMHeadModel.from_pretrained("./gpt2-chatbot")
	tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-chatbot")
```

Este passo envolve carregar o modelo e o tokenizador treinados usando a função from_pretrained(), que permite reutilizar os pesos ajustados do modelo GPT-2 e as configurações do tokenizador sem precisar retrainar. Isso é essencial para aplicações práticas, como responder perguntas dos usuários em um aplicativo web. O código carrega o modelo e o tokenizador do diretório ./gpt2-chatbot, e inclui uma função gerar_resposta (passo 12) que tokeniza a entrada do usuário, gera uma resposta com o modelo, e decodifica a saída para texto. Este processo garante inferências rápidas e consistentes com o treinamento.

## 🔨 Função para gerar resposta

```python
	def gerar_resposta(model, tokenizer, input_text, max_length=50, num_return_sequences=1):
	    inputs = tokenizer.encode(input_text, return_tensors='pt')
	    attention_mask = [1] * len(inputs[0])
	    outputs = model.generate(inputs, attention_mask=torch.tensor([attention_mask]), max_length=max_length, num_return_sequences=num_return_sequences, pad_token_id=tokenizer.eos_token_id)
	    generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
	    return generated_text
```

Esta é uma função Python que gera uma resposta baseada em um modelo de linguagem treinado. A função é chamada `gerar_resposta` e tem quatro parâmetros:

* `model`: um modelo de linguagem treinado
* `tokenizer`: um objeto que é usado para codificar e decodificar texto
* `input_text`: o texto de entrada que será usado para gerar a resposta
* `max_length` (opcional): o tamanho máximo da resposta gerada (padrão é 50)
* `num_return_sequences` (opcional): o número de respostas geradas (padrão é 1)

Aqui está o que a função faz:

1. Codifica o texto de entrada usando o objeto `tokenizer` e armazena o resultado em uma variável chamada `inputs`.
2. Cria uma máscara de atenção* que é usada para indicar quais tokens do texto de entrada devem ser considerados quando o modelo gera a resposta.
3. Chama o método `generate` do modelo para gerar a resposta. O método `generate` é usado para gerar texto baseado em um texto de entrada e um modelo de linguagem.
4. O método `generate` retorna uma lista de saídas, que são as respostas geradas. A função itera sobre essa lista e decodifica cada saída usando o objeto `tokenizer`.
5. A função remove os tokens especiais (como tokens de início e fim de texto) da resposta gerada usando o método `decode` do objeto `tokenizer`.
6. A função retorna a lista de respostas geradas.

Em resumo, esta função é usada para gerar respostas baseadas em um modelo de linguagem treinado, com base em um texto de entrada.

> [!NOTE]
> Máscara de atenção* é uma ferramenta usada em modelos de linguagem, especialmente em arquiteturas de transformadores, como GPT-2, para controlar quais tokens (palavras ou sub-palavras) em uma sequência de entrada devem ser considerados (ou “atendidos”) pelo modelo em diferentes etapas de processamento.


## 🔨 Exemplo de uso da função de geração de resposta

```python
	input_text = "What is the term for a foul committed by a player that prevents an opponent from scoring a goal?"
	resposta = gerar_resposta(model, tokenizer, input_text)
	print(f"{resposta}")
```

Essa função é útil para criar chatbots e sistemas de resposta automática que necessitam de geração de texto baseado em modelos de linguagem.







