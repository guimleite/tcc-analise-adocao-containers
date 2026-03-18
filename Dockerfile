# Dockerfile para Análise Estatística - TCC
# Tecnologias de Conteinerização: Curva de Aprendizagem Organizacional
#
# Autor: Guilherme Magalhães Leite
# Instituição: MBA USP/ESALQ - Engenharia de Software

# Imagem base: Python 3.10 (conforme especificado no projeto)
FROM python:3.10-slim

# Metadados
LABEL maintainer="Guilherme Magalhães Leite"
LABEL description="Ambiente containerizado para análise estatística do TCC sobre containers e orquestradores"
LABEL version="1.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JUPYTER_ENABLE_LAB=yes

# Diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema (necessárias para algumas libs Python)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivo de requisitos
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar arquivos do projeto
COPY . .

# Criar diretórios necessários
RUN mkdir -p resultados/graficos resultados/tabelas resultados/relatorios

# Copiar configurações customizadas do Jupyter
RUN mkdir -p /root/.jupyter/custom
COPY .jupyter/custom/custom.css /root/.jupyter/custom/

# Expor porta do Jupyter Notebook
EXPOSE 8888

# Comando padrão: iniciar Jupyter Notebook
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
