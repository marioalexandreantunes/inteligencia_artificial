# 🖥 Entendendo as Bases da Inteligência Artificial (AI) 

A IA é um ramo da ciência da computação que desenvolve sistemas capazes de executar tarefas que normalmente requerem inteligência humana, como aprendizagem, reconhecimento de padrões, tomada de decisões e resolução de problemas. Está presente em diversas aplicações, desde assistentes virtuais até sistemas de recomendação e carros autônomos.

### Categorias de IA
- **Inteligência Artificial Restrita (ANI)**: Focada em tarefas específicas, como vencer um jogo de xadrez ou identificar rostos em fotos. AI Fraca. Nest acategoria que estão as atuais LLMs
- **Inteligência Artificial Geral (AGI)**: Equivalente à inteligência humana em todas as tarefas. AI Forte.
- **Superinteligência Artificial (ASI)**: Superaria a inteligência humana. AI Forte.

### Componentes para Criar IA
1. **Dados**: Essenciais para o treinamento.
2. **Algoritmos**: Processam e aprendem com os dados.
3. **Computação**: Necessária para processar grandes volumes de dados.
4. **Especialistas**: Profissionais em ciência de dados e aprendizagem de máquina.
5. **Infraestrutura**: Servidores e armazenamento.

### Representação gráfica :

<p align="center">
  <img src="https://i.imgur.com/2QMJU20.png" alt="My cool logo" width="350px"/>
</p>

---

## ➡️ Machine Learning (ML)

Machine Learning é uma área da inteligência artificial que permite que sistemas aprendam e façam previsões ou decisões sem serem explicitamente programados. Utiliza algoritmos para identificar padrões em dados e melhorar o desempenho ao longo do tempo com experiência. 
Inclui:
- Coleta e pré-processamento de dados.
- Engenharia de recursos e seleção de modelos.
- Treino e avaliação do modelo.
- Implantação e manutenção contínua.

### Exemplo de Ferramentas

1. **TensorFlow:** Biblioteca de código aberto do Google.
2. **PyTorch:** Popular por sua flexibilidade e facilidade de uso.
3. **Scikit-learn:** Focado em tarefas de aprendizagem supervisionado e não supervisionado.
4. **Keras:** Interface de alto nível para redes neurais.
5. **Apache Spark MLlib:** Biblioteca de machine learning para processamento em larga escala.

### Exemplo de Algoritmos

1. **Regressão Linear:** Modelo simples para prever valores contínuos.
2. **Árvores de Decisão:** Usado para classificação e regressão.
3. **Redes Neurais:** Inspiradas no cérebro humano, excelentes para padrões complexos. Mais usado nas atuais LLMs
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
- 📌 **Transformers**: Para tarefas de linguagem e multimodais. os Transformers também têm demonstrado uma grande versatilidade e são utilizados em várias dessas aplicações, muitas vezes complementando ou até substituindo GANs em certas tarefas. Mais usado nas LLMs mais conhecidas.
- 📌 **Redes Generativas Adversariais (GANs)**: Criar imagens realistas de pessoas, objetos, e cenas que não existem no mundo real. Síntese de imagens médicas para ajudar no treinamento de modelos de diagnóstico, etc. 

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

## Como criar/melhorar um modelo IA com dados próprios e sem custo!?

Melhorar um modelo de IA com seus próprios dados sem custo é viável utilizando ferramentas e recursos gratuitos. A chave é aproveitar bibliotecas de código aberto e plataformas que oferecem computação gratuita, como Google Colab e Kaggle Kernels. Além disso, ferramentas como Streamlit e Flask permitem que você implante seu modelo de maneira simples e acessível.

O **LM Studio** e o **Anything LLM** são ferramentas que podem facilitar o processo de criação e implantação de modelos de linguagem natural. Vamos ver como você pode usá-las para criar seu próprio modelo de IA com dados próprios e sem custos.

Mais importante porque se deve usar este método é para a privacidade dos dados que para muitas empresas é muito importante.

### LM Studio

📀 [LM Studio](https://lmstudio.ai/) é uma plataforma ou ferramenta que facilita o treino e ajuste fino de modelos de linguagem natural. Como utilizá-lo:
1. Configuração Inicial
2. Coleta e Preparo de Dados
3. Treinamento do Modelo
4. Avaliação e Ajustes

### Anything LLM

📀 [Anything LLM](https://anythingllm.com/) é uma plataforma que facilita a construção, ajuste fino e implantação de modelos de linguagem natural. Como usá-la:
1. Instalação e Configuração
2. Treinamento e Ajuste Fino
3. Implantação

Utilizando ferramentas como LM Studio e Anything LLM, é possível criar, ajustar e implantar modelos de IA com seus próprios dados de maneira eficiente e gratuita. Certifique-se de seguir as documentações oficiais dessas ferramentas para detalhes específicos e melhores práticas.

### Videos de ajuda

🔎 [Youtube](https://www.youtube.com/results?search_query=+lm+studio+e+anything+llm+dados+proprios)

### Fine Tunning de modelos LLM?

O “fine-tuning” de um modelo de LLM (Large Language Model) é como dar um “treino especial” para que ele se torne especialista em uma tarefa específica. Imagine que você tem um super-herói com muitos poderes, mas você o treina para ser um detetive especialista em desvendar crimes. O “fine-tuning” faz isso com LLMs, usando dados específicos da sua área de interesse para que ele aprenda a realizar tarefas complexas com mais precisão e criatividade, como escrever textos no estilo de um autor específico, traduzir idiomas com mais naturalidade ou gerar código de programação para diferentes aplicações. É como dar um toque final para que o LLM se torne a ferramenta perfeita para as suas necessidades!

### Videos de ajuda

🔎 [Youtube](https://www.youtube.com/results?search_query=como+fazer+finetunning+llm+)

--- 

### 1. Fine-Tuning

**O que é:**
Fine-tuning é o processo de ajustar um modelo de linguagem pré-treinado em um conjunto de dados específico para melhorar seu desempenho em tarefas ou domínios particulares. O modelo é treinado adicionalmente com dados relevantes para o domínio de interesse.

**Vantagens:**
- **Especialização:** Permite que o modelo se especialize em um domínio específico, aprendendo nuances e terminologias que são relevantes para o contexto.
- **Desempenho em Tarefas Específicas:** Pode melhorar o desempenho em tarefas específicas, como classificação ou geração de texto em um domínio específico.
- **Controle:** Oferece mais controle sobre o comportamento do modelo, já que o treinamento é personalizado para as necessidades específicas.

**Desvantagens:**
- **Requer Dados:** Necessita de um conjunto de dados específico e relevante para o treinamento.
- **Custo Computacional:** O fine-tuning pode ser caro em termos de recursos computacionais e tempo.
- **Menos Flexível:** Pode ser menos flexível para mudanças rápidas em conhecimento ou para abordar uma ampla gama de tópicos sem re-treinamento adicional.

### 2. Retrieval-Augmented Generation (RAG)

**O que é:**
RAG combina a recuperação de informações com a geração de texto. O modelo de recuperação busca documentos relevantes, e o modelo de geração usa essas informações para produzir respostas mais informadas e contextualmente precisas.
Imagine um sistema que combina o poder do machine learning com a capacidade de buscar e combinar informações de fontes externas. 

**Pipeline:**
- Input: O usuário faz uma pergunta.
- Retrieval: O sistema busca documentos ou fragmentos relevantes.
- Generation: O modelo gera uma resposta usando as informações recuperadas.

**Vantagens:**
- **Atualização de Conhecimento:** Pode acessar informações atualizadas e relevantes diretamente do banco de dados.
- **Flexibilidade:** Pode lidar com uma ampla gama de tópicos sem precisar de fine-tuning extensivo.
- **Respostas Contextualizadas:** Melhora a precisão e relevância das respostas ao combinar recuperação de informações com geração de texto.

**Desvantagens:**
- **Complexidade:** Requer a configuração e manutenção de um sistema de recuperação de documentos eficiente.
- **Dependência de Dados Externos:** A qualidade das respostas depende da qualidade e relevância dos dados recuperados.
- **Custo Computacional:** A combinação de recuperação e geração pode ser intensiva em termos de recursos.

### 3. Prompt Engineering

**O que é:**
Prompt engineering envolve a criação de prompts eficazes para guiar o modelo de linguagem a gerar respostas desejadas. Em vez de modificar o modelo, você ajusta a forma como as perguntas ou instruções são apresentadas.

**Vantagens:**
- **Simplicidade:** Não requer treinamento adicional e pode ser implementado rapidamente.
- **Flexibilidade:** Pode adaptar-se a diferentes contextos e tarefas apenas alterando os prompts.
- **Menos Custo Computacional:** Menos intensivo em recursos comparado ao fine-tuning ou RAG.

**Desvantagens:**
- **Limitações de Precisão:** A eficácia dos prompts pode ser limitada e não garantir sempre respostas precisas ou contextualmente adequadas.
- **Dependência do Modelo Base:** A qualidade das respostas ainda depende das capacidades do modelo base e pode não ser suficiente para tarefas muito específicas.

### Qual é Melhor?

- **Fine-Tuning:** Melhor para situações em que você precisa de um modelo altamente especializado para tarefas ou domínios específicos e tem recursos para treinamento adicional.
- **RAG:** Ideal quando você precisa de respostas atualizadas e contextualizadas e pode acessar um grande volume de dados. Também é útil quando a flexibilidade é importante e você deseja integrar informações externas.
- **Prompt Engineering:** Adequado para tarefas onde você pode ajustar a maneira como faz perguntas ou dá instruções ao modelo. É uma solução rápida e de baixo custo para melhorar o desempenho sem treinamento adicional.

**Escolha com base em suas necessidades:**

- **Para tarefas muito específicas e especializadas** com recursos para treinamento, **fine-tuning** pode ser a melhor escolha.
- **Para tarefas que se beneficiam de informações atualizadas e amplas** e onde a integração com um banco de dados externo é viável, **RAG** pode ser mais eficaz.
- **Para ajustes rápidos e flexíveis** sem a necessidade de treinamento adicional, **prompt engineering** é geralmente a abordagem mais prática.

Cada técnica tem suas próprias forças e limitações, e a melhor escolha pode até envolver uma combinação dessas abordagens, dependendo do problema específico que você está tentando resolver.

---

## 📌 Tendencias

## Deep Learning em 2025: Um Resumo

Prepare-se para um futuro onde a IA esteja mais presente do que nunca! 

### Deep Learning mais acessível e eficiente:

* Modelos menores e mais rápidos para dispositivos móveis e IoT.
* Edge computing para processamento local e redução de latência. Arquitetura de TI onde os dados do cliente são processados no limite da rede, ou o mais próximo possível da fonte de dados.
* Hardware especializado tornando a IA mais acessível a todos.

### IA mais confiável e ética:

* Federated Learning para privacidade de dados. É um método de treino de modelos de IA, onde os dados permanecem localizados nos dispositivos originais, e apenas os parâmetros do modelo são compartilhados.
* Inteligência Interpretada para modelos mais transparentes. Permitir que humanos compreendam como a IA chega a suas previsões.
* Robustness contra ataques para garantir segurança e confiabilidade.

### Deep Learning em novas áreas:

* Revolucionando ciência de dados, pesquisa, engenharia e manufatura.
* Expandindo fronteiras da arte e criatividade com IA como ferramenta.

### Deep Learning como serviço:

* Plataformas de IA com acesso fácil a modelos e ferramentas.
* Serviços de inferência para processamento de dados em tempo real.

### Automação e Agentes AI:

A área da automação e agentes AI está em plena **expansão**, com uma rica variedade de APIs, frameworks e plataformas disponíveis.

* Automação de tarefas complexas e fluxos de trabalho.
* Assistentes virtuais avançados e agentes inteligentes autônomos.
* Agentes colaborativos trabalhando em conjunto com humanos.

### Desafios e oportunidades:

* Discussões e regulamentações sobre ética e impacto social da IA.
* Adaptação do mercado de trabalho e necessidade de novas habilidades.
* Distribuição equitativa dos benefícios da automação.

A automação, impulsionada pela inteligência artificial, tem o potencial de transformar diversos setores, impactando a necessidade de certos tipos de trabalho.

- **Tarefas Repetitivas e Padronizadas**:
    - Operadores de Telemarketing: Chatbots e sistemas de atendimento automatizado podem lidar com chamadas simples, agendamentos e respostas a perguntas frequentes.
    - Digitadores e Transcritores: Softwares de reconhecimento de fala e OCR (Optical Character Recognition) podem automatizar a transcrição de áudio e texto.
    - Caixa de supermercado: Caixas self-checkout já estão em uso, e sistemas de reconhecimento de imagem podem automatizar a identificação de produtos.
    - Processamento de dados: Softwares podem automatizar tarefas como entrada de dados, classificação de informações e geração de relatórios.

- **Tarefas que Requerem Análise de Dados Simples**:
    - Analistas de dados básicos: Softwares podem identificar padrões e tendências em dados, liberando analistas para tarefas mais complexas.
    - Assistentes administrativos: Softwares podem automatizar tarefas como agendamento de reuniões, gerenciamento de emails e organização de arquivos.

- **Trabalhos Criativos (impacto gradual)**:
    - Redatores de conteúdo básico: Geração de textos simples, descrições de produtos, roteiros básicos.
    - Criadores de conteúdo visual: Geração de imagens, design de elementos gráficos simples.
    - Tradutores: Tradução de textos simples.

- **Tarefas que Podem Ser Realizadas por Robôs**:
    - Operadores de linha de produção: Robôs industriais podem realizar tarefas repetitivas e perigosas em fábricas.
    - Motoristas de caminhão: Veículos autônomos podem automatizar entregas e transporte de cargas.
    - Trabalhadores de construção: Robôs podem auxiliar em tarefas como pintura, soldagem e demolição.

**Em resumo:** 2025 será um ano crucial para o Deep Learning, com avanços significativos em eficiência, confiabilidade e aplicações. A IA se tornará ainda mais integrada em nossas vidas, trazendo oportunidades e desafios que exigirão reflexão e adaptação.
A IA impactará o mercado de trabalho, mas o nível de impacto é incerto. A adaptação e desenvolvimento de habilidades humanas serão cruciais para o futuro do trabalho. Previsões sobre o impacto do emprego variam, com alguns modelos predizerem aumento de empregos, outros a redução.
O impacto dependerá de fatores como políticas públicas, adaptação da força de trabalho.

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

---

## Servidores GPU

Utilizar VPS (Virtual Private Server) com GPUs (Graphics Processing Units) para implantações de IA pode ser uma escolha poderosa, especialmente para aplicações que exigem processamento intensivo, como treinamento de modelos de deep learning ou inferência em tempo real. Aqui estão algumas considerações e vantagens de usar VPS com GPUs para implantações de IA.

### Principais Fornecedores de VPS com GPUs

1. **Amazon Web Services (AWS)**
   - **EC2 P3 Instances:** GPUs NVIDIA Tesla V100
   - **EC2 G4 Instances:** GPUs NVIDIA T4

2. **Google Cloud Platform (GCP)**
   - **Compute Engine:** GPUs NVIDIA Tesla K80, P100, V100, T4
   - **AI Platform:** Serviço gerenciado com suporte para GPUs

3. **Microsoft Azure**
   - **NC-Series:** GPUs NVIDIA Tesla K80, P100, V100
   - **ND-Series:** GPUs NVIDIA Tesla P40

4. **IBM Cloud**
   - **Virtual Servers for VPC:** GPUs NVIDIA Tesla V100

5. **Oracle Cloud Infrastructure (OCI)**
   - **Compute Instances with GPUs:** GPUs NVIDIA Tesla P100, V100


### Fornecedores de VPS com GPUs 

6. **Paperspace**
   - Instâncias GPU dedicadas e preemptíveis com GPUs NVIDIA Quadro, Tesla, RTX - RTX4000 0.56/hour

7. **Linode**
   - **GPU Instances:** GPUs NVIDIA Tesla T4

8. **runpod.io**
   - Preço pode ir desde $0.13/hr - RTX 3070 - 8GB VRAM

9. **vast.ai**
    - preço pode ir desde $0.17/hr - RTX 3070 - 8GB VRAM

10. **Hetzner**
    - para modelos GGML, aluga Hetzner CAX4 no Falkenstein datacenter. 16 ARM vCPU e 32GB of RAM dar-te-á 10 tokens/sec até 33B 5bit.

11. **Modal**
    - Para modelos GPTQ, aluga Nvidia A10G $1.10/hr, uma unica placa faz 33B (20+ tokens/seg.) ou uma divisão 17,22 suporta 65B em dois cartões (5 tokens/seg.)
   
12. **huggingface.co**

