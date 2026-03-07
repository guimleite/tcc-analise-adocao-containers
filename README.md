# Análise Estatística - TCC Engenharia de Software

## Descrição

Aplicação containerizada para análise estatística de dados sobre adoção de containers e orquestradores em empresas de tecnologia.

**Projeto:** Tecnologias de Conteinerização - Curva de Aprendizagem Organizacional
**Autor:** Guilherme Magalhães Leite
**Instituição:** MBA USP/ESALQ - Engenharia de Software

---

## Requisitos

- [Docker](https://www.docker.com/) e [Docker Compose](https://docs.docker.com/compose/)

**Alternativa:** Python 3.10+ e pip

---

## Como Executar

1. **Suba os containers**

   ```sh
   docker-compose up -d
   ```

2. **Acesse o Jupyter Notebook**

   http://localhost:8888

3. **Execute a análise**

   - Abra `notebooks/00_master_analise.ipynb`
   - Execute todas as células: `Cell → Run All`

4. **Verifique os resultados**

   Gráficos salvos em `resultados/graficos/`

5. **Para parar os containers**

   ```sh
   docker-compose down
   ```

---

## Análises Implementadas

- **Análise Descritiva** - Frequências, médias, tabelas de contingência
- **Teste Qui-Quadrado** - Testes de independência entre variáveis
- **ANACOR** - Análise de Correspondência (mapas perceptuais)
- **Regressão Logística** - Fatores preditivos de curva de aprendizado
- **Visualizações** - Gráficos descritivos

---

## Estrutura do Projeto

```
Analise_App/
├── docker-compose.yml          # Configuração Docker
├── Dockerfile                  # Imagem Python + Jupyter
├── requirements.txt            # Dependências Python
│
├── notebooks/                  # Jupyter Notebooks
│   ├── 00_master_analise.ipynb        # Notebook principal
│   ├── 01_analise_descritiva.ipynb
│   ├── 02_qui_quadrado.ipynb
│   ├── 03_anacor.ipynb
│   ├── 04_regressao.ipynb
│   └── 05_visualizacoes.ipynb
│
├── src/                        # Código-fonte
│   └── utils.py                # Funções reutilizáveis
│
├── dados/                      # Dados CSV
│   └── respostas.csv           # CSV do Google Forms
│
└── resultados/                 # Outputs gerados
    └── graficos/               # Gráficos PNG (300 DPI)
        ├── anacor/
        ├── descritivas/
        └── regressao/
```

---

## Dependências

### Ambiente
- Python 3.10
- Docker & Docker Compose

### Bibliotecas Python

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| pandas | 2.1.4 | Manipulação de dados |
| numpy | 1.26.2 | Operações numéricas |
| matplotlib | 3.8.2 | Visualizações básicas |
| seaborn | 0.13.0 | Visualizações estatísticas |
| scipy | 1.11.4 | Teste qui-quadrado |
| prince | 0.13.0 | Análise de Correspondência (ANACOR) |
| statsmodels | 0.14.1 | Regressão Logística |
| jupyter | 1.0.0 | Ambiente interativo |
| notebook | 7.0.6 | Jupyter Notebook |
| ipykernel | 6.27.1 | Kernel Python para Jupyter |

---

## Executar sem Docker

```sh
pip install -r requirements.txt
jupyter notebook
# Abra notebooks/00_master_analise.ipynb
```

---

Uso acadêmico - TCC MBA USP/ESALQ