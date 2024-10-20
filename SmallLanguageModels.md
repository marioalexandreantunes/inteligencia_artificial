# Comparação das principais SML (Out de 2024)

## Tabela com as informações mais recentes sobre os modelos Llama 3.2 1B e 3B:

| Modelo | Parâmetros | Contexto (tokens) | MMLU | MT-Bench | Linguagens | Licença | Uso Local | Preço estimado (1M/t)* | Principais Aplicações |
|--------|------------|-------------------|------|----------|------------|---------|-----------|----------------------|----------------------|
| Phi-3-mini | 3,8B | 2048 | 68,8% | 8,38 | Multilíngue | Proprietária (Microsoft) | Sim | $0.1* | Linguagem, código, matemática |
| Phi-3-small | 7B | 2048 | 75,3% | 8,68 | Multilíngue | Proprietária (Microsoft) | Sim | $0.30* | Tarefas de linguagem complexas |
| Llama 3.2 1B | 1,23B | 128K | N/D | N/D | 8 idiomas | Apache 2.0 | Sim | $0.02* | Diálogo multilíngue, recuperação e resumo[1][3] |
| Llama 3.2 3B | 3,21B | 128K | N/D | N/D | 8 idiomas | Apache 2.0 | Sim | $0.05* | Resumos, classificação, tradução[2] |
| Mixtral 7B | 7B | 32K | 68,4% | 8,3 | Multilíngue | Apache 2.0 | Sim | $0.5* | Tarefas gerais, codificação |
| Gemma | 9B | 8K | 63,6% | 7,0 | Inglês | Apache 2.0 | Sim | $0.06* | IA responsável, tarefas gerais |
| OpenELM | Varia | N/D | N/D | N/D | N/D | Apache 2.0 | Sim | Gratuito** | Processamento em dispositivos de borda |
| DeepSeek-Coder-V2 | Varia | 16K | N/D | N/D | Múltiplas | Apache 2.0 | Sim | Gratuito** | Geração e compreensão de código |

Notas adicionais sobre os modelos Llama 3.2 1B e 3B:

1. **Linguagens suportadas**: Ambos os modelos oferecem suporte robusto a oito idiomas: inglês, alemão, francês, italiano, português, hindi, espanhol e tailandês[1][2].
2. **Contexto**: Ambos os modelos mantêm uma capacidade de contexto de 128K tokens, o que é uma melhoria significativa em relação a muitos outros modelos[1][2].
3. **Aplicações**: 
   - O modelo 1B é ideal para assistentes de escrita com IA em dispositivos móveis, aplicações de atendimento ao cliente e gerenciamento de informações pessoais[2][3].
   - O modelo 3B é adequado para resumos de texto, classificação e tarefas de tradução de idiomas[2].
4. **Eficiência**: Esses modelos são projetados para serem mais eficientes em cargas de trabalho de IA, com latência reduzida e desempenho aprimorado[2].
5. **Uso em dispositivos de borda**: Ambos os modelos são otimizados para uso em dispositivos de borda e aplicações móveis, permitindo processamento local com forte privacidade de dados[2][3].
6. **Arquitetura**: Utilizam uma arquitetura de transformador otimizada e foram ajustados usando técnicas de fine-tuning supervisionado (SFT) e aprendizado por reforço com feedback humano (RLHF)[3][4].
7. **Dados de treinamento**: Foram pré-treinados em até 9 trilhões de tokens de dados de fontes publicamente disponíveis, com data de corte em dezembro de 2023[3][4].
