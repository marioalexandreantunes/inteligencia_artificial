# 🖥 Inteligência Artificial (IA)

A IA é um ramo da ciência da computação que desenvolve sistemas capazes de executar tarefas que normalmente requerem inteligência humana, como aprendizado, reconhecimento de padrões, tomada de decisões e resolução de problemas. Está presente em diversas aplicações, desde assistentes virtuais até sistemas de recomendação e carros autônomos.

### Categorias de IA
- **Inteligência Artificial Restrita (ANI)**: Focada em tarefas específicas, como vencer um jogo de xadrez ou identificar rostos em fotos.
- **Inteligência Artificial Geral (AGI)**: Equivalente à inteligência humana em todas as tarefas.
- **Superinteligência Artificial (ASI)**: Superaria a inteligência humana.

### Componentes para Criar IA
1. **Dados**: Essenciais para o treinamento.
2. **Algoritmos**: Processam e aprendem com os dados.
3. **Computação**: Necessária para processar grandes volumes de dados.
4. **Especialistas**: Profissionais em ciência de dados e aprendizado de máquina.
5. **Infraestrutura**: Servidores e armazenamento.

### Representação grafica :

<p align="center">
  <img src="https://i.imgur.com/2QMJU20.png" alt="My cool logo" width="350px"/>
</p>

---

## ➡️ Machine Learning (ML)

Machine Learning é uma área da inteligência artificial que permite que sistemas aprendam e façam previsões ou decisões sem serem explicitamente programados. Utiliza algoritmos para identificar padrões em dados e melhorar o desempenho ao longo do tempo com experiência. 
Inclui:
- Coleta e pré-processamento de dados.
- Engenharia de recursos e seleção de modelos.
- Treinamento e avaliação do modelo.
- Implantação e manutenção contínua.

### Exemplo de Ferramentas

1. **TensorFlow:** Biblioteca de código aberto do Google.
2. **PyTorch:** Popular por sua flexibilidade e facilidade de uso.
3. **Scikit-learn:** Focado em tarefas de aprendizado supervisionado e não supervisionado.
4. **Keras:** Interface de alto nível para redes neurais.
5. **Apache Spark MLlib:** Biblioteca de machine learning para processamento em larga escala.

### Exemplo de Algoritmos

1. **Regressão Linear:** Modelo simples para prever valores contínuos.
2. **Árvores de Decisão:** Usado para classificação e regressão.
3. **Redes Neurais:** Inspiradas no cérebro humano, excelentes para padrões complexos.
4. **Máquinas de Vetores de Suporte (SVM):** Classificação e regressão com margens máximas.
5. **K-Means:** Algoritmo de clusterização não supervisionado.
6. **Random Forest:** Conjunto de árvores de decisão para melhorar a precisão.

---

## ➡️ Redes Neurais Artificiais (ANNs)

As redes neurais artificiais são inspiradas no funcionamento do cérebro humano, compostas por neurônios artificiais interconectados. São usadas para tarefas como classificação, reconhecimento de padrões e processamento de imagens.

### Principais Arquiteturas:
- **Perceptron Simples**: Modelo básico com uma única camada de neurônios.
- **Multilayer Perceptron (MLP)**: Várias camadas, capaz de resolver problemas mais complexos.
- **Redes Neurais Convolucionais (CNNs)**: Especializadas em reconhecimento de imagem e processamento de vídeo.
- **Redes Neurais Recorrentes (RNNs)**: Usadas para processamento de linguagem natural e reconhecimento de voz.
- **Long Short-Term Memory (LSTM)**: Variante das RNNs para dados sequenciais.
- **Transformers**: Para tarefas de linguagem e multimodais.
- **Redes Generativas Adversariais (GANs)**: Geram dados realistas, como imagens e música.

---

## ➡️ Deep Learning

O deep learning utiliza múltiplas camadas de processamento para aprender representações complexas de dados, sendo uma aplicação avançada de machine learning.
Redes neurais são a base do deep learning. 
Assim, o deep learning é uma técnica avançada que utiliza redes neurais profundas para resolver problemas complexos, como reconhecimento de imagem e processamento de linguagem natural.

### Transformers

Transformers são uma arquitetura de deep learning projetada para lidar com dados sequenciais, como texto. Introduzidos no artigo "Attention is All You Need" em 2017, eles revolucionaram o processamento de linguagem natural (NLP). São amplamente utilizados em várias aplicações de inteligência artificial devido à sua eficácia e flexibilidade.
Os Transformers têm se tornado padrão em muitas soluções de IA modernas e AI generativas.

### IA Generativa?

Subárea da IA que se concentra em criar modelos capazes de gerar novos conteúdos, ideias ou dados que são coerentes e plausíveis, muitas vezes se assemelhando às saídas geradas por humanos. Isso muitas vezes envolve o uso de modelos de Deep Learning, como redes generativas adversariais (GANs) ou modelos de linguagem.
Em uma IA generativa, surge o conceito de 💬 **prompt**, que é basicamente um texto de entrada que o humano fornece à IA para iniciar uma conversa ou gerar conteúdo. Assim, o 💬 **prompt** é essencialmente a maneira de comunicar à IA o que o humano quer que ela faça.

## ➡️ Large Language Models (LLMs)

Um Large Language Model é um modelo de inteligência artificial treinado com grandes quantidades de dados textuais para realizar tarefas de processamento de linguagem natural (PLN). Esses modelos conseguem gerar, compreender e traduzir texto, além de responder perguntas e resumir informações.
Os dados em uma LLM (Language Model) são armazenados de forma indireta, como parte dos pesos do modelo. Aqui está como funciona:

### Estrutura de Armazenamento

1. **Pesos do Modelo:** Os dados de treinamento influenciam os pesos das redes neurais, que representam o conhecimento aprendido.
2. **Vetores Embeddings:** As palavras e frases são convertidas em vetores numéricos que capturam significado e contexto.
3. **Arquitetura Neural:** As camadas do modelo utilizam esses pesos e vetores para gerar respostas baseadas no input recebido.

### Detalhes

- **Não-Armazenamento Direto:** Os modelos não armazenam dados em formato de texto literal, mas em padrões numéricos aprendidos.
- **Generalização:** O modelo aprende a generalizar a partir dos dados, permitindo gerar respostas novas e contextuais.

Esse método de armazenamento permite que as LLMs respondam de maneira inteligente a uma variedade de 💬 **prompts**.

### Exemplos de LLM em 2024
- Claude 3 (Anthropic)
- LLaMA (Meta)
- Mistral (Mistral AI)
- GPT (OpenAI)
- Gemma e Gemini (Google)
- Grok (xAI)

## ➡️ Small Language Models (SLMs)

Os SLMs são versões mais compactas de modelos de linguagem treinados para executar tarefas específicas de processamento de linguagem natural (PLN), com menos recursos computacionais.

### Características:
- **Eficiência**: Consomem menos memória e poder de processamento.
- **Rapidez**: Oferecem respostas mais rápidas devido ao menor tamanho.
- **Customização**: Podem ser facilmente adaptados para tarefas específicas.

### Vantagens:
- **Implementação em Dispositivos Móveis**: Ideais para aplicações em smartphones e dispositivos IoT.
- **Custos Reduzidos**: Menos exigentes em termos de infraestrutura.
- **Treinamento e Atualização Mais Rápidos**: Facilitam o processo de ajuste e melhorias contínuas.

### Aplicações Comuns:
- Chatbots simples.
- Assistentes virtuais em dispositivos limitados.
- Sistemas de resposta automática de emails.

### Exemplos de SLMs:
- **DistilBERT**: Versão menor do BERT, mantendo boa parte do desempenho.
- **TinyBERT**: Outra variante compacta do BERT, otimizada para velocidade e eficiência.

SLMs são essenciais para levar o poder do processamento de linguagem a aplicações com restrições de recursos, permitindo que a inteligência artificial seja amplamente acessível.


## ➡️ Large Language Models (LLMs) vs. Small Language Models (SLMs)

**Large Language Models (LLMs):**

- **Características:**
  - Grande número de parâmetros.
  - Capazes de lidar com tarefas complexas de linguagem.
  - Necessitam de mais recursos computacionais.

- **Vantagens:**
  - Excelente desempenho em compreensão de texto.
  - Suportam múltiplas tarefas simultaneamente.
  - Melhor capacidade de generalização.

- **Desvantagens:**
  - Alto custo de treino e implementação.
  - Lentos em dispositivos com recursos limitados.

**Small Language Models (SLMs):**

- **Características:**
  - Menor número de parâmetros.
  - Otimizados para tarefas específicas.
  - Mais leves e rápidos.

- **Vantagens:**
  - Rápidos e eficientes em recursos.
  - Mais fáceis de personalizar.
  - Ideais para dispositivos móveis e IoT.

- **Desvantagens:**
  - Menor capacidade de generalização.
  - Limitados em tarefas complexas.

**Comparação:**

- **Desempenho vs. Eficiência:** LLMs oferecem desempenho superior, enquanto SLMs são mais eficientes e econômicos.
- **Escalabilidade:** LLMs são melhores para grandes aplicações, SLMs são ideais para soluções específicas.
- **Implementação:** SLMs são mais fáceis de implementar em ambientes com restrições de recursos.

---

## 📌 Curiosidades

🖐🏼 Para rodar uma LLM de **32 bilhões** de parâmetros localmente, você precisaria de um computador com as seguintes especificações:

1. **GPU Potente:**
   - Placas como NVIDIA RTX 3090 ou superiores, preferencialmente com mais de 24 GB de VRAM.
   - Suporte a CUDA para aceleração de processamento.

2. **Memória RAM:**
   - Pelo menos 64 GB de RAM para lidar com a carga de dados e operações simultâneas.

3. **Processador (CPU):**
   - Processador moderno com múltiplos núcleos, como Intel i9 ou AMD Ryzen 9.

4. **Armazenamento:**
   - SSD com capacidade de 1 TB ou mais, para garantir leitura e escrita rápidas.

5. **Sistema Operacional:**
   - Linux é geralmente preferido por compatibilidade e eficiência, mas também pode ser feito no Windows.

6. **Resfriamento Adequado:**
   - Para manter a temperatura dos componentes sob controle durante operações intensivas.

Essas especificações ajudam a garantir que o modelo funcione de forma eficiente, embora otimizações adicionais possam ser necessárias para ajustar o desempenho.

---

🖐🏼 Para rodar um LLM de **405 bilhões** de parâmetros localmente, você precisaria de um computador com especificações ainda mais avançadas:

1. **GPU de Alta Capacidade:**
   - Placas como NVIDIA A100 ou H100 com 80 GB de VRAM ou mais, preferencialmente em configurações multi-GPU.

2. **Memória RAM:**
   - Pelo menos 512 GB de RAM para suportar o processamento de dados.

3. **Processador (CPU):**
   - Processador de servidor com muitos núcleos, como AMD EPYC ou Intel Xeon.

4. **Armazenamento:**
   - Múltiplos SSDs NVMe com capacidade total de vários terabytes.

5. **Sistema Operacional:**
   - Linux, devido à sua eficiência e compatibilidade com software de ML.

6. **Infraestrutura de Resfriamento:**
   - Sistema avançado de resfriamento para lidar com o calor gerado por operações intensivas.

Além disso, considere a necessidade de um ambiente distribuído ou em cluster para lidar com cargas de trabalho dessa magnitude de forma eficaz.

---

## 🖐🏼 CPU, GPU, LPU, NPU e TPU:

### CPU (Central Processing Unit)
- **Função:** Unidade de processamento geral em computadores.
- **Uso:** Executa tarefas gerais, ótima para operações sequenciais.
- **Vantagens:** Versátil, capaz de lidar com múltiplos tipos de processos.
- **Desvantagens:** Menor desempenho em tarefas massivamente paralelas.

### GPU (Graphics Processing Unit)
- **Função:** Processamento paralelo para gráficos e computação intensiva.
- **Uso:** Treinamento de modelos de machine learning e deep learning.
- **Vantagens:** Excelente para cálculos paralelos e processamento de grandes volumes de dados.
- **Desvantagens:** Consome mais energia e requer programação específica para otimização.

### LPU (Learning Processing Unit)
- **Função:** Projetada especificamente para acelerar o machine learning.
- **Uso:** Ainda emergente, focada em otimizar tarefas de machine learning.
- **Vantagens:** Otimizada para eficiência energética em machine learning.
- **Desvantagens:** Menos comum e menos suportada atualmente.

### NPU (Neural Processing Unit)
- **Função:** Redes neurais
- **Uso:** Dispositivos móveis, inferência de IA
- **Vantagens:** Processamento de operações de IA

### TPU (Tensor Processing Unit)
- **Função:** Desenvolvida pelo Google para acelerar redes neurais.
- **Uso:** Utilizada principalmente em aplicações de machine learning no Google Cloud.
- **Vantagens:** Altamente eficiente em cargas de trabalho de deep learning.
- **Desvantagens:** Limitada a certos ambientes e menos flexível fora de aplicações de ML.

### IPU (Intelligence Processing Unit)
- **Função:** Processamento de IA
- **Vantagens:** Flexibilidade e paralelismo extremo
- **Uso:** Modelos de deep learning complexos, pesquisa em IA

Essas unidades de processamento atendem a diferentes necessidades, desde tarefas gerais (CPU) até processamento intensivo de dados (GPU e TPU), com LPUs sendo uma área emergente focada em machine learning.

Exemplos de cada tipo de unidade:

### CPU (Central Processing Unit)
- **Intel Core i9-12900K**
- **AMD Ryzen 9 5950X**

### GPU (Graphics Processing Unit)
- **NVIDIA RTX 3080**
- **AMD Radeon RX 6800 XT**
- **NVIDIA H100/H200 Tensor Core** (Desenhada para tarefas de computação intensiva, especialmente inteligência artificial e deep learning)

### LPU (Learning Processing Unit)
- **SambaNova SN10-8 (exemplo de uma LPU emergente)**

### TPU (Tensor Processing Unit)
- **Google TPU v4**
- **Edge TPU (para dispositivos de IoT)**

### IPU (Intelligence Processing Unit)
- **Graphcore GC200 IPU:** Usado em sistemas de data center para acelerar IA.
- **Mythic IPU:** Focado em computação de IA de baixa potência e eficiência.
- **Habana Labs Goya:** Oferece desempenho otimizado para inferência de IA.

### NPU (Neural Processing Unit)
- **Huawei Kirin NPU:** Usado em dispositivos móveis para acelerar tarefas de IA.
- **Samsung Exynos NPU:** Integra capacidades de IA em smartphones para processamento eficiente.

Esses exemplos representam opções comuns e emergentes em suas respectivas categorias.

---

O 🔥 **Jetson Nano da NVIDIA** 🔥 é um computador compacto e poderoso projetado para projetos de inteligência artificial e computação de borda. Ele é ideal para desenvolvedores que desejam criar soluções de IA e robótica acessíveis e eficientes com um valor aprox. de 250€.

### Características principais:

1. **Processador**: Possui um CPU quad-core ARM Cortex-A57.
2. **GPU**: Equipada com 128 núcleos CUDA, excelente para tarefas de IA.
3. **Memória**: 4 GB de RAM LPDDR4.
4. **Conectividade**: Oferece USB, Ethernet e suporte para câmera CSI.
5. **Sistema Operacional**: Compatível com Linux, usando a distribuição JetPack da NVIDIA.

### Aplicações comuns:

- **Visão computacional**
- **Robótica**
- **Automação residencial**
- **Drones**

É uma escolha popular para entusiastas e profissionais por sua combinação de potência e preço acessível. Além disso, a comunidade em torno dele é bastante ativa, o que facilita encontrar tutoriais e suporte para projetos.
