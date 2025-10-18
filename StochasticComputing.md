# Stochastic Computing

Stochastic computing* é um método de cálculo que representa números através de **fluxos de bits aleatórios**, permitindo realizar operações complexas usando apenas **operações simples entre bits**.
Em vez de procurar a máxima precisão, este método **troca um pouco de exatidão e velocidade por eficiência**, o que pode resultar em circuitos **de baixo consumo, alta densidade e muito tolerantes a falhas**.
Tem voltado a despertar interesse, sobretudo em **inteligência artificial e aprendizagem profunda**, graças ao seu potencial em eficiência energética e simplicidade de hardware.

*Stochastic Computing - Computação estocástica - É a tradução mais direta e correta. “Estocástico” vem do grego stokhastikos (relativo ao acaso), e é usado em matemática e estatística em português europeu. 

---

### Como funciona

* **Representação dos números:**
  Em vez de usar números binários fixos (como 1011), o stochastic computing representa o valor pela **probabilidade de ocorrência de 1s** num fluxo de bits.
  Imagine que cada bit é como o resultado de uma moeda:

  * se a moeda der “cara” (1) quase sempre, o número é próximo de 1;
  * se sair “cara” poucas vezes, o número é pequeno.
    Assim, um fluxo com 80% de 1s representa aproximadamente 0,8.

* **Operações:**
  As operações matemáticas complexas são divididas em **operações muito simples entre bits**.
  Por exemplo:

  * Multiplicar dois números pode ser feito com um **único portão lógico AND** (como se duas moedas tivessem de sair “cara” ao mesmo tempo);
  * Somar pode ser implementado com um **multiplexador**, um componente básico que escolhe entre sinais.

* **Erro e precisão:**
  A **precisão** depende do **comprimento do fluxo de bits**.
  Quanto mais longo o fluxo, mais fiel o resultado — mas também mais tempo demora.
  O sistema é naturalmente **tolerante a falhas**, pois um erro isolado (um bit trocado) tem pouco impacto no valor total, da mesma forma que um lançamento errado não muda muito a média de mil lançamentos de moeda.

---

### Vantagens

* **Baixo consumo e tamanho reduzido:**
  Por usar componentes simples e eliminar circuitos complexos, consome menos energia e ocupa menos espaço no chip.

* **Tolerância a falhas:**
  Mesmo com erros em alguns componentes, o sistema continua a funcionar corretamente, tornando-se **robusto e confiável**.

* **Simplicidade de hardware:**
  As operações básicas (como multiplicar e somar) exigem **blocos de circuito muito simples**, facilitando o design de certas aplicações.

---

### Desafios

* **Velocidade vs. precisão:**
  Há um **equilíbrio inevitável**: resultados mais precisos requerem fluxos mais longos, logo mais tempo de processamento.

* **Complexidade de design:**
  Adaptar algoritmos convencionais para o formato estocástico **não é direto** e requer novas abordagens de projeto.

* **Aplicabilidade limitada:**
  Nem todas as funções matemáticas podem ser facilmente traduzidas para este tipo de representação.

---

### Aplicações

* **Inteligência Artificial (IA):**
  Especialmente promissora em redes neurais profundas (CNNs), que exigem **milhões de multiplicações** e podem aceitar pequenas aproximações sem comprometer o resultado.

* **Processamento Digital de Sinais (DSP):**
  Útil no tratamento em tempo real de dados de sensores e outros sinais, onde **um pequeno erro é aceitável**.

* **Processamento de Imagem:**
  Pode ser aplicado a **filtros e transformações de imagem**, obtendo **boa eficiência com baixo custo de hardware**.

---

### Tabela Comparativa

**Tabela comparativa** entre os três métodos: o tradicional binário (computação digital convencional), o Stochastic computing (computação estocástica) e o Quantum computing (computação quântica).


| Método                   | Como funciona resumido                                                                                                                                                 | Principais vantagens                                                                              | Principais limitações                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Computação binária**   | Utiliza bits “0” ou “1” e executa operações lógicas e aritméticas tradicionais.                                                                                        | Tecnologia madura, previsível, fácil de programar e amplamente suportada.                         | Maior consumo energético, limitações em paralelismo e miniaturização.                          |
| **Computação Estocástica** | Representa números por **fluxos de bits aleatórios**, onde a proporção de 1s indica o valor. As operações complexas são substituídas por simples operações entre bits. | Consumo reduzido de energia, alta densidade de integração e forte tolerância a falhas.            | Menor precisão, desempenho dependente do tamanho do fluxo e aplicabilidade restrita.           |
| **Computação quântica**  | Usa **qubits** que podem estar em superposição (0 e 1 ao mesmo tempo) e exploram fenómenos quânticos como o entrelaçamento.                                            | Capaz de resolver certos problemas de forma exponencialmente mais rápida, com enorme paralelismo. | Tecnologia complexa, alto nível de erros e necessidade de ambientes controlados (criogénicos). |

---

**Resumo comparativo tradicional:**

* A **computação binária** é o método clássico e estável — a base de toda a tecnologia atual.
* A **computação estocástica** é uma alternativa eficiente e tolerante a falhas, adequada para aplicações específicas como IA ou processamento de sinais.
* A **computação quântica** representa o salto para o futuro: promissora, mas ainda imatura e limitada a centros de pesquisa e aplicações experimentais.

Investir em computação estocástica é muito mais prático, acessível e realista no curto e médio prazo do que investir em computação quântica.
Vamos detalhar isso com clareza:

### 🔹 1. Base tecnológica e componentes usados

| Aspeto                             | **Computação binária**                                                                    | **Computação estocástica**                                                                                                                                                               | **Computação quântica**                                                                                                                              |
| ---------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tipo de componente**             | Transístores CMOS tradicionais (os mesmos usados há décadas em processadores e memórias). | Usa exatamente os **mesmos transístores e portas lógicas** (AND, OR, multiplexadores, etc.). Não precisa de novo hardware, apenas uma forma diferente de codificar e processar os dados. | Requer **nova tecnologia física**: qubits baseados em átomos, íons, fotões ou supercondutores, controlados a temperaturas quase absolutas (−273 °C). |
| **Infraestrutura**                 | Totalmente madura.                                                                        | Pode ser implementada nos **chips atuais** com pequenas modificações no design.                                                                                                          | Exige **laboratórios especializados**, isolamento magnético e criogenia.                                                                             |
| **Disponibilidade de componentes** | Produção global em massa.                                                                 | Já disponível — usa o mesmo ecossistema CMOS.                                                                                                                                            | Extremamente limitada, experimental e cara.                                                                                                          |

➡️ **Conclusão:**
A computação estocástica **aproveita o hardware já existente**, enquanto a quântica **precisa reinventar o hardware desde a base**.


### 🔹 2. Custo e maturidade tecnológica

* **Computação estocástica** pode ser vista como uma **evolução dentro do digital tradicional**, não uma revolução completa.
  É como ensinar uma máquina velha a pensar de forma mais probabilística — requer software novo e técnicas de design diferentes, mas **usa a mesma fábrica de chips**.
* **Computação quântica**, por outro lado, ainda está no equivalente ao “período dos tubos de vácuo” da computação clássica — promissora, mas longe da produção industrial.
  Cada qubit é frágil, difícil de controlar e de manter estável.

➡️ Assim, do ponto de vista empresarial e industrial, **a estocástica é um investimento incremental e pragmático**, enquanto a quântica é **um investimento de longo prazo e alto risco**.


### 🔹 3. Aplicabilidade prática

* A computação estocástica é **ideal para tarefas aproximadas** como IA, visão computacional, sensores, e processamento de sinais — todos mercados em forte crescimento e já utilizáveis hoje.
* A quântica tem potencial para **problemas de otimização, criptografia e simulação molecular**, mas **ainda não há uso comercial direto** fora de ambientes de pesquisa.

➡️ Ou seja, se quisermos **ganhos reais nos próximos 5 a 10 anos**, a aposta racional é **na computação estocástica**, não na quântica.


### 🔹 4. Analogia simples

Pense assim:

* O **binário** é como um carro a combustão — confiável, conhecido e universal.
* O **estocástico** é como um carro híbrido — aproveita o motor que já existe, mas usa energia de forma mais eficiente.
* O **quântico** seria um carro a levitação magnética — um salto de paradigma, mas que ainda não tem estrada nem infraestrutura.




