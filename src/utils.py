"""
Módulo de Utilitários para Análise Estatística
TCC - Engenharia de Software - MBA USP/ESALQ

Autor: Guilherme Magalhães Leite
Orientador: Arthur Pinheiro de Araújo Costa

Este módulo contém funções reutilizáveis para:
- Pré-processamento de dados
- Teste Qui-Quadrado
- ANACOR (Análise de Correspondência)
- Regressão Logística
- Visualizações
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
import prince
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import csv
import os
import warnings
from IPython.display import display

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

def titulo_secao(texto):
    """Imprime título de seção com separadores visuais"""
    print("\n" + "="*80)
    print(texto.upper())
    print("="*80)


def tabela_frequencia(serie, titulo, nome_coluna="Categoria"):
    """Cria tabela de frequência com percentuais"""
    freq = serie.value_counts()
    pct = (serie.value_counts(normalize=True) * 100).round(2)

    tabela = pd.DataFrame({
        nome_coluna: freq.index,
        'Frequência': freq.values,
        '%': pct.values
    })

    print(f"\n{titulo}\n")

    styled = (tabela.style
              .hide(axis='index')
              .format({'%': '{:.2f}'})
              .set_properties(**{'text-align': 'left'})
              .set_table_styles([
                  {'selector': 'th', 'props': [('text-align', 'left')]},
                  {'selector': '', 'props': [('border', '1px solid black')]},
                  {'selector': 'th', 'props': [('border', '1px solid black')]},
                  {'selector': 'td', 'props': [('border', '1px solid black')]}
              ]))

    display(styled)


def tabela_desafios_likert(df):
    """Exibe estatísticas dos desafios em escala Likert"""
    desafios = DESAFIOS_LIKERT

    print("\nESTATÍSTICAS DESCRITIVAS: Desafios (Escala Likert 1-5)\n")

    # % alto = níveis 4-5, mesma dicotomização declarada na metodologia
    stats = pd.DataFrame({
        'Desafio': desafios.keys(),
        'Média': [df[col].mean() for col in desafios.values()],
        'Desvio': [df[col].std() for col in desafios.values()],
        'Min': [df[col].min() for col in desafios.values()],
        'Max': [df[col].max() for col in desafios.values()],
        '% alto': [(df[col] >= LIMIAR_DESAFIO_ALTO).mean() * 100 for col in desafios.values()],
    }).sort_values('% alto', ascending=False)

    styled = (stats.style
              .hide(axis='index')
              .format({'Média': '{:.2f}', 'Desvio': '{:.2f}', 'Min': '{:.0f}', 'Max': '{:.0f}',
                       '% alto': '{:.1f}'})
              .set_properties(**{'text-align': 'left'})
              .set_table_styles([
                  {'selector': 'th', 'props': [('text-align', 'left')]},
                  {'selector': '', 'props': [('border', '1px solid black')]},
                  {'selector': 'th', 'props': [('border', '1px solid black')]},
                  {'selector': 'td', 'props': [('border', '1px solid black')]}
              ]))
    display(styled)


def tabela_contingencia(df, var1, var2, titulo):
    """Cria tabela de contingência formatada"""
    print(f"\n{'='*80}")
    print(f"{titulo.upper()}")
    print(f"{'='*80}\n")

    tabela = pd.crosstab(df[var1], df[var2])
    tabela.index.name = None
    tabela.columns.name = None

    styled = (tabela.style
              .set_properties(**{'text-align': 'center'})
              .set_table_styles([
                  {'selector': 'th', 'props': [('text-align', 'left')]},
                  {'selector': '', 'props': [('border', '1px solid black')]},
                  {'selector': 'th', 'props': [('border', '1px solid black')]},
                  {'selector': 'td', 'props': [('border', '1px solid black')]}
              ]))

    display(styled)


def configurar_visualizacoes():
    """Configura estilo padrão para visualizações"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams.update({'figure.figsize': (12, 8), 'font.size': 10})
    pd.set_option('display.max_columns', None)


# ============================================================================
# FIGURAS PARA O TCC (manual USP/Esalq, Tabela 8)
# ============================================================================
# Sem grade, sem borda, sem preenchimento, sem título; eixos pretos de 1,5 pt;
# rótulos Arial (ou equivalente) <= 11 pt, pretos. A legenda vai no Word.

def formatar_grafico_tcc(ax, xlabel='', ylabel='', fonte=10):
    """Aplica as regras de formatação de gráficos do manual a um eixo.
    fonte: tamanho dos rótulos dos eixos; os títulos dos eixos usam fonte + 1
    (máximo 11, limite do manual). Painéis de meia página pedem fonte=8."""
    ax.figure.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)
    ax.set_title('')
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)
    for lado in ('left', 'bottom'):
        ax.spines[lado].set_visible(True)
        ax.spines[lado].set_color('black')
        ax.spines[lado].set_linewidth(1.5)
    titulo = min(fonte + 1, 11)
    ax.set_xlabel(xlabel, fontsize=titulo, color='black')
    ax.set_ylabel(ylabel, fontsize=titulo, color='black')
    ax.tick_params(colors='black', labelsize=fonte)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']


COR_BARRA = '#4d7fa8'


def _rotulo_n_pct(v, n):
    """'46 (41,8%)' — vírgula como separador decimal, conforme o manual."""
    return f"{v} ({v / n * 100:.1f}%)".replace('.', ',')


def _letra_painel(ax, letra):
    if letra:
        ax.text(0, 1.02, letra, transform=ax.transAxes, fontsize=12,
                fontweight='bold', ha='left', va='bottom')


def _contagem(serie, ordem):
    cont = serie.value_counts()
    return cont.reindex(ordem).fillna(0).astype(int) if ordem else cont


def barras_horizontais(ax, serie, ordem=None, xlabel='Respondentes', ylabel='', letra=None):
    """Barras horizontais com n e % ao lado de cada barra, no padrão do manual.
    ordem: categorias de baixo para cima; sem ordem, ordena por frequência (maior no topo).
    Indicada para categorias nominais com rótulos longos."""
    cont = _contagem(serie, ordem)
    if not ordem:
        cont = cont.sort_values()
    n = int(cont.sum())
    ax.barh(cont.index, cont.values, color=COR_BARRA, edgecolor='black', linewidth=0.8)
    for i, v in enumerate(cont.values):
        ax.text(v + cont.max() * 0.02, i, _rotulo_n_pct(v, n), va='center', fontsize=9)
    ax.set_xlim(0, cont.max() * 1.3)
    formatar_grafico_tcc(ax, xlabel, ylabel)
    _letra_painel(ax, letra)


def colunas_verticais(ax, serie, ordem, xlabel='', ylabel='Respondentes', letra=None,
                      rotulos=None, fonte=8):
    """Colunas verticais com n e % acima de cada coluna (em duas linhas), no padrão
    do manual. ordem: categorias da esquerda para a direita; rotulos: texto exibido
    no eixo para cada categoria (use '\\n' para quebrar). Indicada para faixas ordinais
    em painéis de meia página (fonte 8)."""
    cont = _contagem(serie, ordem)
    n = int(cont.sum())
    ax.bar(range(len(cont)), cont.values, color=COR_BARRA, edgecolor='black', linewidth=0.8)
    for i, v in enumerate(cont.values):
        ax.text(i, v + cont.max() * 0.02, _rotulo_n_pct(v, n).replace(' (', '\n('),
                ha='center', va='bottom', fontsize=fonte, linespacing=1.1)
    ax.set_ylim(0, cont.max() * 1.3)
    formatar_grafico_tcc(ax, xlabel, ylabel, fonte=fonte)  # antes dos rótulos: define labelsize
    ax.set_xticks(range(len(cont)))
    ax.set_xticklabels(rotulos or cont.index, fontsize=fonte)
    _letra_painel(ax, letra)


# ============================================================================
# PRÉ-PROCESSAMENTO DE DADOS
# ============================================================================

def carregar_dados(arquivo_csv='../dados/respostas.csv'):
    """Carrega o CSV do Google Forms, renomeia colunas e remove espaços nas pontas."""
    try:
        df = pd.read_csv(arquivo_csv, encoding='utf-8-sig')
    except (UnicodeDecodeError, FileNotFoundError):
        df = pd.read_csv(arquivo_csv, encoding='latin-1')

    df = renomear_colunas(df)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    return df


def renomear_colunas(df):
    """Renomeia colunas do Google Forms para nomes curtos"""
    keywords = {
        'carimbo': 'timestamp',
        'cargo': 'q1_cargo',
        'você trabalha na área de ti': 'q2_tempo_ti',
        'você trabalha com containers': 'q3_tempo_containers',
        'certificações': 'q4_certificacoes',
        'participa de comunidades': 'q5_comunidades',
        'porte da empresa': 'q6_porte_empresa',
        'setor de atuação': 'q7_setor',
        'empresa utiliza containers em produção': 'q8_tempo_containers_producao',
        'tecnologia de orquestração': 'q9_tecnologia_orquestracao',
        'em quais ambientes': 'q10_ambientes',
        'processo de adoção': 'q11_processo_adocao',
        'segurança (vulnerabilidades': 'q12_desafio_seguranca',
        'complexidade técnica': 'q13_desafio_complexidade',
        'monitoramento e observabilidade': 'q14_desafio_monitoramento',
        'mudanças organizacionais': 'q15_desafio_mudancas',
        'treinamento e desenvolvimento': 'q16_desafio_treinamento',
        'ci/cd (automação': 'q17_desafio_cicd',
        'principal desafio': 'q18_principal_desafio',
        'estratégias de aprendizado': 'q19_estrategias_aprendizado',
        'empresa ofereceu treinamento': 'q20_empresa_ofereceu_treinamento',
        'alocação de tempo': 'q21_alocacao_tempo',
        'fonte de aprendizado foi mais útil': 'q22_fonte_mais_util',
        'realizou pocs': 'q23_realizou_pocs',
        'você se sentir confortável': 'q24_tempo_individual',
        'equipe levou': 'q25_tempo_equipe',
        'fator foi mais importante': 'q26_fator_mais_importante',
        'retorno sobre o investimento': 'q27_roi',
        'recomendaria': 'q28_recomendaria'
    }

    mapa = {}
    for col in df.columns:
        col_lower = col.lower()
        for keyword, nome_curto in keywords.items():
            if keyword in col_lower:
                mapa[col] = nome_curto
                break

    return df.rename(columns=mapa)


def carregar_e_preparar(arquivo_csv='../dados/respostas.csv', verbose=True,
                        detalhado=False):
    """Pipeline completo. detalhado=True audita cada etapa da limpeza."""
    df = carregar_dados(arquivo_csv)
    if verbose:
        print(f"Dataset carregado: {len(df)} respostas × {len(df.columns)} variáveis")

    for rotulo, funcao in [('consolidação', consolidar_categorias),
                           ('múltipla escolha', explodir_multipla_escolha),
                           ('derivadas', preparar_variaveis)]:
        antes = len(df.columns)
        df = funcao(df, verbose=detalhado)
        if verbose and not detalhado:
            print(f"  {rotulo + ':':20s} {len(df.columns) - antes:2d} colunas")

    if verbose:
        print(f"Pronto para análise: {len(df)} respostas × {len(df.columns)} variáveis")
    return df


def garantir_preparado(df=None, arquivo_csv='../dados/respostas.csv', verbose=False):
    """Devolve o df já tratado; carrega e trata se ainda não passou pelo pipeline."""
    if df is None or 'q1_cargo_c' not in df.columns:
        return carregar_e_preparar(arquivo_csv, verbose=verbose)
    return df


# ============================================================================
# CONSOLIDAÇÃO DE CATEGORIAS
# ============================================================================
# Sufixo _c = versão descritiva; _macro = versão agregada para inferência.

# Q1: cargo (14 -> 6). As 5 opções fixas do questionário recebem rótulo em
# português; respostas do campo "Outro" funcionalmente equivalentes a uma
# opção fixa são recodificadas para ela, as demais vão para Outros.
MAPA_Q1_CARGO = {
    'DevOps Engineer': 'Engenheiro DevOps',
    'Platform Engineer': 'Engenheiro DevOps',
    'backend/devops': 'Engenheiro DevOps',
    'SRE (Site Reliability Engineer)': 'Engenheiro de Confiabilidade (SRE)',
    'Desenvolvedor': 'Desenvolvedor de Software',
    'Líder Técnico/Tech Lead': 'Líder Técnico',
    'Technical Product Manager': 'Líder Técnico',
    'Arquiteto de Soluções': 'Arquiteto de Soluções',
    'Estágio em tecnologia': 'Outros',
    'Freelancer': 'Outros',
    'Não estou trabalhando no momento': 'Outros',
    'Analista de segurança de aplicações': 'Outros',
    'Suporte': 'Outros',
    'Analista de Sistemas': 'Outros',
}

# Q1: 6 -> 4, para inferência
MAPA_Q1_MACRO = {
    'Engenheiro DevOps': 'DevOps/SRE/Plataforma',
    'Engenheiro de Confiabilidade (SRE)': 'DevOps/SRE/Plataforma',
    'Desenvolvedor de Software': 'Desenvolvedor',
    'Líder Técnico': 'Liderança/Arquitetura',
    'Arquiteto de Soluções': 'Liderança/Arquitetura',
    'Outros': 'Outros',
}

# Q7: setor (25 -> 10 -> 5)
MAPA_Q7_SETOR = {
    'Saas/Software': 'Software/SaaS',
    'LawTech': 'Software/SaaS',
    'Segurança da informação': 'Software/SaaS',
    'tecnologia. Infraestrutura e monitoramento.': 'Software/SaaS',
    'Fintech/Serviços Financeiros': 'Fintech/Serviços Financeiros',
    'Benefícios': 'Fintech/Serviços Financeiros',
    'Telecom': 'Telecom',
    'Agro': 'Indústria/Logística',
    'Agronegócio': 'Indústria/Logística',
    'IndTech': 'Indústria/Logística',
    'Manufatura': 'Indústria/Logística',
    'Engenharia Elétrica': 'Indústria/Logística',
    'Automotivo': 'Indústria/Logística',
    'Setor logístico + Software': 'Indústria/Logística',
    'TMS': 'Indústria/Logística',
    'Setor Público': 'Setor Público',
    'E-commerce/Varejo': 'E-commerce/Varejo',
    'Saúde/HealthTech': 'Saúde/HealthTech',
    'Educação/EdTech': 'Educação/EdTech',
    'Consultoria': 'Consultoria/TI',
    'consultoria': 'Consultoria/TI',
    'AWS Partner, foco em migrações cloud.': 'Consultoria/TI',
    'Mídia/Entretenimento': 'Mídia/Games',
    'jogos': 'Mídia/Games',
    'e-gaming': 'Mídia/Games',
}

MAPA_Q7_MACRO = {
    'Software/SaaS': 'Software/SaaS/Consultoria',
    'Consultoria/TI': 'Software/SaaS/Consultoria',
    'Fintech/Serviços Financeiros': 'Fintech/Serviços Financeiros',
    'Setor Público': 'Setor Público, Saúde e Educação',
    'Saúde/HealthTech': 'Setor Público, Saúde e Educação',
    'Educação/EdTech': 'Setor Público, Saúde e Educação',
    'Indústria/Logística': 'Indústria, Logística e Varejo',
    'E-commerce/Varejo': 'Indústria, Logística e Varejo',
    'Telecom': 'Telecom e Mídia',
    'Mídia/Games': 'Telecom e Mídia',
}

# Q18: principal desafio (16 -> 12 -> 8)
MAPA_Q18_DESAFIO = {
    'A adoção foi tranquila.': 'Sem desafio relevante',
    'Não houve desafios, a equipe técnica incluindo o time de desenvolvimento já havia '
    'sido informado que trabalharia com micro serviços, e que ficar em containers era a '
    'cultura da empresa.': 'Sem desafio relevante',
    'migração': 'Integração com sistemas legados',
    'Conseguir realizar a migração sem comprometer produção': 'Integração com sistemas legados',
    'Testes (Debug, E2E, Local env, etc...)': 'CI/CD (automação, fluxos de deploy)',
}

MAPA_Q18_MACRO = {
    'Mudanças organizacionais e culturais (adoção de DevOps, colaboração entre equipes)':
        'Organizacional/Cultural',
    'Treinamento e desenvolvimento de habilidades (falta de capacitação, upskilling)':
        'Organizacional/Cultural',
    'Complexidade técnica (curva de aprendizado íngreme, novos conceitos)': 'Complexidade técnica',
    'Monitoramento e observabilidade (monitoramento, logs, tracing)': 'Monitoramento',
    'Segurança': 'Segurança',
    'Custos de infraestrutura': 'Infra/Escala',
    'Escalabilidade': 'Infra/Escala',
    'Networking': 'Infra/Escala',
    'Persistência de dados e gestão de estado': 'Infra/Escala',
    'Integração com sistemas legados': 'Legado/Migração',
    'CI/CD (automação, fluxos de deploy)': 'CI/CD',
    'Sem desafio relevante': 'Sem desafio relevante',
}

# Q22: fonte mais útil (10 -> 5)
MAPA_Q22_FONTE = {
    'Documentação oficial (Kubernetes, Docker Docs)': 'Documentação e autoestudo',
    'Aprendizado autodirigido (tentativa e erro)': 'Documentação e autoestudo',
    'Tutoriais e artigos online (blogs, Medium, Dev.to)': 'Documentação e autoestudo',
    'Livros técnicos': 'Documentação e autoestudo',
    'Treinamento formal / cursos pagos (ex.: Udemy, Coursera, KodeKloud, Linux Academy)':
        'Treinamento formal',
    'Cursos estruturados (online ou presenciais)': 'Treinamento formal',
    'Mentoria interna (desenvolvedores seniores ou especialistas)': 'Mentoria/Comunidade',
    'Comunidades técnicas (fóruns, Stack Overflow, Reddit)': 'Mentoria/Comunidade',
    'Provas de Conceito (PoCs) e projetos piloto': 'Prática/PoC',
    'Agentes de inteligência artificial (IA)': 'Agentes de IA',
}

# Q22, sensibilidade: 5 -> 3. O par Q22 × tempo individual falha na premissa
# em 5x3; o reagrupamento verifica se a recusa vem da esparsidade ou da
# ausência de associação. Critério: quem conduz o aprendizado — o próprio
# profissional, um instrutor, ou pessoas/prática.
MAPA_Q22_FONTE3 = {
    'Documentação e autoestudo': 'Autoestudo',
    'Agentes de IA': 'Autoestudo',
    'Treinamento formal': 'Treinamento formal',
    'Mentoria/Comunidade': 'Social/Prática',
    'Prática/PoC': 'Social/Prática',
}

# Q26: fator acelerador (7 -> 5; a descritiva usa a coluna original)
MAPA_Q26_MACRO = {
    'Realização de POCs antes de produção': 'Prática/PoC',
    'Experiência prévia com Docker': 'Experiência prévia',
    'Mentoria de especialistas internos ou externos': 'Mentoria/Comunidade',
    'Participação em comunidades técnicas': 'Mentoria/Comunidade',
    'Tempo dedicado exclusivo para estudo': 'Tempo dedicado',
    'Treinamento formal estruturado': 'Formal/Certificação',
    'Certificações profissionais': 'Formal/Certificação',
}


def _aplicar_mapa(df, origem, destino, mapa, rotulo, avisos):
    """Aplica um mapeamento categórico registrando respostas não previstas."""
    if origem not in df.columns:
        avisos.append(f"coluna '{origem}' ausente; '{destino}' não foi criada")
        return
    df[destino] = df[origem].map(mapa)
    orfas = df.loc[df[origem].notna() & df[destino].isna(), origem].unique()
    if len(orfas):
        avisos.append(f"{rotulo}: {len(orfas)} resposta(s) sem mapeamento -> {list(orfas)}")


def consolidar_categorias(df, verbose=True):
    """Cria as colunas consolidadas _c e _macro, preservando as originais."""
    df = df.copy()
    avisos = []

    _aplicar_mapa(df, 'q1_cargo', 'q1_cargo_c', MAPA_Q1_CARGO, 'Q1 cargo', avisos)
    _aplicar_mapa(df, 'q1_cargo_c', 'q1_cargo_macro', MAPA_Q1_MACRO, 'Q1 macro', avisos)
    _aplicar_mapa(df, 'q7_setor', 'q7_setor_c', MAPA_Q7_SETOR, 'Q7 setor', avisos)
    _aplicar_mapa(df, 'q7_setor_c', 'q7_setor_macro', MAPA_Q7_MACRO, 'Q7 macro', avisos)
    _aplicar_mapa(df, 'q22_fonte_mais_util', 'q22_fonte_c', MAPA_Q22_FONTE, 'Q22 fonte', avisos)
    _aplicar_mapa(df, 'q22_fonte_c', 'q22_fonte_3', MAPA_Q22_FONTE3, 'Q22 fonte-3', avisos)
    _aplicar_mapa(df, 'q26_fator_mais_importante', 'q26_fator_macro', MAPA_Q26_MACRO,
                  'Q26 fator', avisos)

    # replace, e não map: respostas fora do mapa permanecem como estão
    if 'q18_principal_desafio' in df.columns:
        df['q18_desafio_c'] = df['q18_principal_desafio'].replace(MAPA_Q18_DESAFIO)
        _aplicar_mapa(df, 'q18_desafio_c', 'q18_desafio_macro', MAPA_Q18_MACRO,
                      'Q18 macro', avisos)

    if verbose:
        titulo_secao("Consolidação de categorias")
        criadas = [('q1_cargo', 'q1_cargo_c'), ('q1_cargo_c', 'q1_cargo_macro'),
                   ('q7_setor', 'q7_setor_c'),
                   ('q7_setor_c', 'q7_setor_macro'), ('q18_principal_desafio', 'q18_desafio_c'),
                   ('q18_desafio_c', 'q18_desafio_macro'),
                   ('q22_fonte_mais_util', 'q22_fonte_c'), ('q22_fonte_c', 'q22_fonte_3'),
                   ('q26_fator_mais_importante', 'q26_fator_macro')]
        for origem, destino in criadas:
            if destino in df.columns:
                perdas = int(df[origem].notna().sum() - df[destino].notna().sum())
                print(f"  {destino:26s} {df[origem].nunique():3d} -> {df[destino].nunique():3d}"
                      + (f"   PERDA: {perdas}" if perdas else ""))
        for a in avisos:
            print(f"  AVISO: {a}")

    return df


# ============================================================================
# MÚLTIPLA ESCOLHA -> VARIÁVEIS BINÁRIAS
# ============================================================================
# Opções não listadas aqui são ignoradas na explosão.

MULTIPLA_ESCOLHA = {
    'q4_certificacoes': {
        'CKA (Certified Kubernetes Administrator)': 'q4_cka',
        'CKAD (Certified Kubernetes Application Developer)': 'q4_ckad',
        'CKS (Certified Kubernetes Security Specialist)': 'q4_cks',
        'Docker Certified Associate': 'q4_docker_certified',
        'Outras certificações cloud (AWS/Azure/GCP relacionadas a containers)': 'q4_outras_cloud',
        'Não possuo certificações': 'q4_nao_possui',
    },
    'q5_comunidades': {
        'Meetups presenciais': 'q5_meetups',
        'Comunidades online (Slack, Discord, Reddit, entre outros)': 'q5_online',
        'Contribuo com projetos open source relacionados': 'q5_open_source',
        'Não participo de comunidades técnicas': 'q5_nao_participa',
    },
    'q10_ambientes': {
        'Desenvolvimento local': 'q10_dev_local',
        'Staging/Homologação': 'q10_staging',
        'Produção': 'q10_producao',
    },
    'q19_estrategias_aprendizado': {
        'Documentação oficial (Kubernetes, Docker Docs)': 'q19_documentacao',
        'Provas de Conceito (PoCs) e projetos piloto': 'q19_pocs',
        'Treinamento formal / cursos pagos (ex.: Udemy, Coursera, KodeKloud, Linux Academy)':
            'q19_treinamento_formal',
        'Aprendizado autodirigido (tentativa e erro)': 'q19_autodirigido',
        'Tutoriais e artigos online (blogs, Medium, Dev.to)': 'q19_tutoriais',
        'Comunidades técnicas (fóruns, Stack Overflow, Reddit)': 'q19_comunidades',
        'Agentes de inteligência artifical (IA)': 'q19_agentes_ia',  # typo do formulário
        'Mentoria interna (desenvolvedores seniores ou especialistas)': 'q19_mentoria',
        'Livros técnicos': 'q19_livros',
        'Consultoria externa': 'q19_consultoria_externa',
    },
}

# Certificações declaradas em campo aberto que contam como válidas
CERTIFICACOES_LIVRES_VALIDAS = [
    'Linux Professional Institute LPIC-3 Virtualization and Containerization',
    'LPI',
    'LPIC-2',
]


def _opcoes_marcadas(serie):
    return serie.fillna('').apply(lambda v: {p.strip() for p in str(v).split(';') if p.strip()})


def explodir_multipla_escolha(df, verbose=True):
    """Converte as questões de múltipla escolha em variáveis binárias."""
    df = df.copy()
    criadas = []

    for coluna, mapa in MULTIPLA_ESCOLHA.items():
        if coluna not in df.columns:
            print(f"AVISO: coluna '{coluna}' ausente")
            continue
        marcadas = _opcoes_marcadas(df[coluna])
        for opcao, nome in mapa.items():
            df[nome] = marcadas.apply(lambda s, o=opcao: int(o in s))
            criadas.append(nome)

    # Nos binários derivados, marcar item específico prevalece sobre a
    # opção negativa genérica ("Não possuo", "Não participo")
    if 'q4_certificacoes' in df.columns:
        df['q4_outras_tecnicas'] = _opcoes_marcadas(df['q4_certificacoes']).apply(
            lambda s: int(bool(s & set(CERTIFICACOES_LIVRES_VALIDAS))))
        cols_cert = ['q4_cka', 'q4_ckad', 'q4_cks', 'q4_docker_certified',
                     'q4_outras_cloud', 'q4_outras_tecnicas']
        df['possui_certificacao'] = (df[cols_cert].sum(axis=1) > 0).astype(int)
        criadas += ['q4_outras_tecnicas', 'possui_certificacao']

    if 'q5_meetups' in df.columns:
        cols_com = ['q5_meetups', 'q5_online', 'q5_open_source']
        df['participa_comunidades'] = (df[cols_com].sum(axis=1) > 0).astype(int)
        criadas.append('participa_comunidades')

    if verbose:
        n = len(df)
        titulo_secao("Múltipla escolha convertida em variáveis binárias")
        for nome in criadas:
            total = int(df[nome].sum())
            alerta = "  constante" if total in (0, n) else ""
            print(f"  {nome:26s} {total:3d}/{n} ({total/n*100:5.1f}%){alerta}")

    return df


# ============================================================================
# VARIÁVEIS DERIVADAS
# ============================================================================

# Escalas ordinais de tempo (1 = mais rápido)
ORDEM_TEMPO_INDIVIDUAL = {
    'Menos de 3 meses': 1,
    '3-6 meses': 2,
    '6-12 meses': 3,
    '12-18 meses': 4,
    'Mais de 18 meses': 5,
}

ORDEM_TEMPO_EQUIPE = {
    'Menos de 1 mês': 1,
    '1-3 meses': 2,
    '3-6 meses': 3,
    '6-12 meses': 4,
    'Mais de 12 meses': 5,
}

DESAFIOS_LIKERT = {
    'Segurança': 'q12_desafio_seguranca',
    'Complexidade': 'q13_desafio_complexidade',
    'Monitoramento': 'q14_desafio_monitoramento',
    'Mudanças organizacionais': 'q15_desafio_mudancas',
    'Treinamento': 'q16_desafio_treinamento',
    'CI/CD': 'q17_desafio_cicd',
}

LIMIAR_DESAFIO_ALTO = 4  # desafio alto = 4-5 na escala Likert


def preparar_variaveis(df, verbose=True):
    """Cria escalas ordinais, desfechos binários, faixas e perfis de desafio."""
    df = df.copy()
    criadas = []

    for col, ordem, destino in [
        ('q24_tempo_individual', ORDEM_TEMPO_INDIVIDUAL, 'tempo_individual_ord'),
        ('q25_tempo_equipe', ORDEM_TEMPO_EQUIPE, 'tempo_equipe_ord'),
    ]:
        if col not in df.columns:
            continue
        df[destino] = df[col].map(ordem)
        nao_mapeados = df[col].notna() & df[destino].isna()
        if nao_mapeados.any():
            print(f"AVISO: {nao_mapeados.sum()} resposta(s) de '{col}' fora da escala "
                  f"esperada e não codificadas: {sorted(df.loc[nao_mapeados, col].unique())}")
        criadas.append(destino)

    # curva rápida: até 6 meses (individual) e até 3 meses (equipe)
    if 'tempo_individual_ord' in df.columns:
        df['curva_rapida_individual'] = (df['tempo_individual_ord'] <= 2).astype(int)
        criadas.append('curva_rapida_individual')

    if 'tempo_equipe_ord' in df.columns:
        df['curva_rapida_equipe'] = (df['tempo_equipe_ord'] <= 2).astype(int)
        criadas.append('curva_rapida_equipe')

    # faixas de 3 níveis: a ANACOR exige tabela de no mínimo 3x3
    if 'tempo_individual_ord' in df.columns:
        df['faixa_tempo_individual'] = pd.cut(
            df['tempo_individual_ord'], bins=[0, 1, 2, 5],
            labels=['Até 3 meses', '3-6 meses', 'Mais de 6 meses'])
        criadas.append('faixa_tempo_individual')

    if 'tempo_equipe_ord' in df.columns:
        df['faixa_tempo_equipe'] = pd.cut(
            df['tempo_equipe_ord'], bins=[0, 2, 3, 5],
            labels=['Até 3 meses', '3-6 meses', 'Mais de 6 meses'])
        criadas.append('faixa_tempo_equipe')

    for rotulo, col in DESAFIOS_LIKERT.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        fora = df[col].notna() & ~df[col].between(1, 5)
        if fora.any():
            print(f"AVISO: {fora.sum()} valor(es) de '{col}' fora da escala 1-5")
        df[col + '_alto'] = (df[col] >= LIMIAR_DESAFIO_ALTO).astype(int)
        criadas.append(col + '_alto')

    cicd = 'q17_desafio_cicd_alto'
    complex_ = 'q13_desafio_complexidade_alto'
    if cicd in df.columns and complex_ in df.columns:
        df['perfil_desafio'] = np.select(
            [(df[cicd] == 1) & (df[complex_] == 1),
             (df[cicd] == 1),
             (df[complex_] == 1)],
            ['CI/CD + Complexidade', 'CI/CD', 'Complexidade'],
            default='Baixo desafio')
        criadas.append('perfil_desafio')

        # 3 níveis: atende à premissa das frequências esperadas na ANACOR
        # rótulos iguais aos usados no texto do TCC
        df['perfil_desafio_3'] = df['perfil_desafio'].replace({
            'CI/CD + Complexidade': 'CI/CD alto',
            'CI/CD': 'CI/CD alto',
            'Complexidade': 'Apenas complexidade alta'})
        criadas.append('perfil_desafio_3')

    # corte alternativo que isola o primeiro mês, equilibrando os grupos
    if 'tempo_equipe_ord' in df.columns:
        df['q25_faixa_deploy'] = pd.cut(
            df['tempo_equipe_ord'], bins=[0, 1, 2, 5],
            labels=['Menos de 1 mês', '1-3 meses', 'Mais de 3 meses'])
        criadas.append('q25_faixa_deploy')

    if verbose:
        print(f"Variáveis derivadas criadas: {len(criadas)}")
        for nome, rot in [('curva_rapida_individual', 'curva rápida individual (até 6 meses)'),
                          ('curva_rapida_equipe', 'curva rápida da equipe (até 3 meses)')]:
            if nome in df.columns:
                n = int(df[nome].sum())
                print(f"  {rot}: {n}/{len(df)} ({n/len(df)*100:.1f}%)")

    return df


# ============================================================================
# TESTE QUI-QUADRADO
# ============================================================================

def residuos_ajustados(tabela, expected):
    """Resíduos padronizados ajustados (Haberman): (O - E) / sqrt(E (1 - p_i)(1 - p_j)).
    Sob H0 ~ N(0,1); |r| > 1,96 indica célula com contribuição relevante."""
    n = tabela.values.sum()
    p_lin = tabela.sum(axis=1).values / n
    p_col = tabela.sum(axis=0).values / n
    denom = np.sqrt(expected * np.outer(1 - p_lin, 1 - p_col))
    return pd.DataFrame((tabela.values - expected) / denom,
                        index=tabela.index, columns=tabela.columns)


def teste_qui_quadrado(df, var1, var2, nome_teste="Teste Qui-Quadrado",
                       alpha=0.05, exibir_tabela=True):
    """Qui-quadrado com checagem da premissa das frequências esperadas e Fisher em 2x2."""
    print(f"\n{'='*80}")
    print(f"{nome_teste.upper()}")
    print(f"{'='*80}\n")

    tabela = pd.crosstab(df[var1], df[var2])
    tabela.index.name = None
    tabela.columns.name = None

    if exibir_tabela:
        styled = (tabela.style
                  .set_properties(**{'text-align': 'center'})
                  .set_table_styles([
                      {'selector': 'th', 'props': [('text-align', 'left')]},
                      {'selector': '', 'props': [('border', '1px solid black')]},
                      {'selector': 'th', 'props': [('border', '1px solid black')]},
                      {'selector': 'td', 'props': [('border', '1px solid black')]}
                  ]))
        display(styled)

    chi2, p_value, dof, expected = chi2_contingency(tabela)

    # premissa: até 20% das células com esperado < 5 e nenhuma < 1
    n_baixo = int((expected < 5).sum())
    pct_baixo = n_baixo / expected.size * 100
    esperado_minimo = float(expected.min())
    premissa_ok = pct_baixo <= 20 and esperado_minimo >= 1

    n = int(tabela.values.sum())
    cramers_v = np.sqrt(chi2 / (n * (min(tabela.shape) - 1)))

    p_fisher = None
    if tabela.shape == (2, 2):
        _, p_fisher = fisher_exact(tabela.values)
    p_ref = p_fisher if p_fisher is not None else p_value
    residuos = residuos_ajustados(tabela, expected)

    print(f"\nQui-quadrado: {chi2:.4f} | p: {p_value:.4f} | "
          f"gl: {dof} | V de Cramér: {cramers_v:.3f}")
    if p_fisher is not None:
        print(f"Fisher exato: p = {p_fisher:.4f}")
    print(f"Esperado < 5: {n_baixo}/{expected.size} ({pct_baixo:.0f}%) | "
          f"mínimo: {esperado_minimo:.2f}")

    if not premissa_ok:
        print("Premissa VIOLADA - p-valor não interpretável.\n")
    elif p_ref < alpha:
        print(f"Associação significativa (p < {alpha}).\n")
        # em 2x2 as quatro células têm o mesmo |r|; só vale mostrar em tabelas maiores
        if tabela.shape != (2, 2):
            print("Resíduos padronizados ajustados (* = |r| > 1,96):")
            marcado = residuos.map(lambda r: f"{r:+.2f}{'*' if abs(r) > 1.96 else ' '}")
            print(marcado.to_string(), "\n")
    else:
        print(f"Sem evidência de associação (p >= {alpha}).\n")

    return {
        'chi2': chi2,
        'residuos': residuos,
        'p_value': p_value,
        'dof': dof,
        'tabela': tabela,
        'expected': expected,
        'premissa_ok': premissa_ok,
        'pct_esperado_baixo': pct_baixo,
        'esperado_minimo': esperado_minimo,
        'cramers_v': cramers_v,
        'p_fisher': p_fisher,
        'n': n,
    }


def bateria_qui_quadrado(df, pares, alpha=0.05, metodo_correcao='fdr_bh',
                         exibir_tabelas=False):
    """Roda a bateria de testes e aplica correção para comparações múltiplas."""
    resultados = {}
    linhas = []

    for var1, var2, rotulo in pares:
        faltando = [v for v in (var1, var2) if v not in df.columns]
        if faltando:
            print(f"AVISO: teste '{rotulo}' ignorado - coluna(s) ausente(s): {faltando}")
            continue

        r = teste_qui_quadrado(df, var1, var2, rotulo, alpha=alpha,
                               exibir_tabela=exibir_tabelas)
        resultados[(var1, var2)] = r
        r['rotulo'] = rotulo
        r['p_referencia'] = r['p_fisher'] if r['p_fisher'] is not None else r['p_value']
        linhas.append(r)

    # corrige só os testes válidos: incluir os inválidos distorceria a correção
    validos = [r for r in linhas if r['premissa_ok']]

    if validos:
        ps = [r['p_referencia'] for r in validos]
        rejeitados, p_ajustados, _, _ = multipletests(ps, alpha=alpha,
                                                      method=metodo_correcao)
        for r, pa, rej in zip(validos, p_ajustados, rejeitados):
            r['p_ajustado'] = float(pa)
            r['significativo'] = bool(rej)
            # significativo antes da correção mas não depois = indício
            r['indicio'] = (not rej) and r['p_referencia'] < alpha

    for r in linhas:
        if not r['premissa_ok']:
            r['p_ajustado'] = np.nan
            r['significativo'] = False
            r['indicio'] = False

    def _classificar(r):
        if not r['premissa_ok']:
            return 'não interpretável'
        return 'sim' if r['significativo'] else ('indício' if r['indicio'] else 'não')

    resumo = pd.DataFrame([{
        'Teste': r['rotulo'],
        'n': r['n'],
        'Tabela': f"{r['tabela'].shape[0]}x{r['tabela'].shape[1]}",
        'gl': r['dof'],
        'Qui-quadrado': round(r['chi2'], 3),
        'V de Cramér': round(r['cramers_v'], 3),
        'p': round(r['p_referencia'], 4),
        'p ajustado': (round(r['p_ajustado'], 4)
                       if not np.isnan(r['p_ajustado']) else '-'),
        'Premissa': 'OK' if r['premissa_ok'] else 'violada',
        'Significativo': _classificar(r),
    } for r in linhas])

    titulo_secao(f"Resumo da bateria ({len(linhas)} testes, correção {metodo_correcao})")
    print(f"\nPremissa atendida em {len(validos)}/{len(linhas)}. "
          f"Em 2x2 o p é o do Fisher exato.\n")
    display(resumo.style.hide(axis='index')
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left')]},
                {'selector': '', 'props': [('border', '1px solid black')]},
                {'selector': 'th', 'props': [('border', '1px solid black')]},
                {'selector': 'td', 'props': [('border', '1px solid black')]}
            ]))

    sig = [r for r in linhas if r['significativo']]
    ind = [r for r in linhas if r['indicio']]

    for rotulo, grupo in [("ACHADOS", sig), ("INDÍCIOS (não sobrevivem à correção)", ind)]:
        print(f"\n{rotulo}: {len(grupo)}")
        for r in grupo:
            print(f"  {r['rotulo']}: p = {r['p_referencia']:.4f} | "
                  f"ajustado = {r['p_ajustado']:.4f} | V = {r['cramers_v']:.3f}")

    return {'resultados': resultados, 'resumo': resumo,
            'achados': sig, 'indicios': ind}


# ============================================================================
# ANACOR (ANÁLISE DE CORRESPONDÊNCIA)
# ============================================================================

def anacor(df, var1, var2, nome_analise="ANACOR", salvar_grafico=True):
    """Gera o mapa perceptual bidimensional do par informado."""
    print(f"\n{'='*80}")
    print(f"ANACOR: {nome_analise.upper()}")
    print(f"{'='*80}\n")

    tabela = pd.crosstab(df[var1], df[var2])
    tabela.index.name = None
    tabela.columns.name = None

    styled = (tabela.style
              .set_properties(**{'text-align': 'center'})
              .set_table_styles([
                  {'selector': 'th', 'props': [('text-align', 'left')]},
                  {'selector': '', 'props': [('border', '1px solid black')]},
                  {'selector': 'th', 'props': [('border', '1px solid black')]},
                  {'selector': 'td', 'props': [('border', '1px solid black')]}
              ]))

    display(styled)

    # a tabela admite min(linhas-1, colunas-1) dimensões
    dim_max = min(tabela.shape[0] - 1, tabela.shape[1] - 1)
    if dim_max < 2:
        print(f"\nTabela {tabela.shape[0]}x{tabela.shape[1]} admite apenas {dim_max} "
              f"dimensão; mapa 2D exige no mínimo 3x3.")
        return {'ca_model': None, 'tabela': tabela, 'dimensoes': dim_max}

    ca = prince.CA(n_components=2, n_iter=10, copy=True, engine='sklearn').fit(tabela)

    # percentage_of_variance_ divide pela inércia total da tabela; usar a soma
    # dos autovalores retidos faria o acumulado dar sempre 100%
    pct_variancia = np.asarray(ca.percentage_of_variance_, dtype=float)
    print(f"\nInércia total: {ca.total_inertia_:.4f}")
    for i, pct in enumerate(pct_variancia[:2]):
        print(f"  Dimensão {i+1}: {pct:.2f}%")
    print(f"  Acumulado: {pct_variancia[:2].sum():.2f}%"
          + ("  (a tabela admite exatamente 2 dimensões)" if dim_max == 2 else ""))

    coord_linhas = ca.row_coordinates(tabela)
    coord_colunas = ca.column_coordinates(tabela)

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(coord_linhas[0], coord_linhas[1],
              s=200, c='steelblue', marker='o',
              edgecolors='black', linewidth=1.5,
              alpha=0.8, label=var1)

    for idx, txt in enumerate(coord_linhas.index):
        ax.annotate(txt, (coord_linhas[0].iloc[idx], coord_linhas[1].iloc[idx]),
                   fontsize=9, fontweight='bold', ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    ax.scatter(coord_colunas[0], coord_colunas[1],
              s=200, c='coral', marker='s',
              edgecolors='black', linewidth=1.5,
              alpha=0.8, label=var2)

    for idx, txt in enumerate(coord_colunas.index):
        ax.annotate(txt, (coord_colunas[0].iloc[idx], coord_colunas[1].iloc[idx]),
                   fontsize=9, fontweight='bold', ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Dimensão 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimensão 2', fontsize=12, fontweight='bold')
    ax.set_title(f'Mapa Perceptual - ANACOR\n{nome_analise}',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if salvar_grafico:
        os.makedirs('../resultados/graficos/anacor', exist_ok=True)
        nome_arquivo = f"../resultados/graficos/anacor/anacor_{var1}_{var2}.png"
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        print(f"\nGráfico salvo: {nome_arquivo}")

    plt.show()

    return {
        'ca_model': ca,
        'coord_linhas': coord_linhas,
        'coord_colunas': coord_colunas,
        'tabela': tabela,
        'inercia_total': float(ca.total_inertia_),
        'pct_variancia': pct_variancia,
        'dimensoes': dim_max,
    }


def anacor_condicionada(df, var1, var2, nome_analise, alpha=0.05,
                        salvar_grafico=True):
    """Aplica ANACOR só se o par passar no qui-quadrado, na premissa e for >= 3x3."""
    qui = teste_qui_quadrado(df, var1, var2, f"Pré-teste: {nome_analise}", alpha=alpha)

    tabela = qui['tabela']
    dim_max = min(tabela.shape[0] - 1, tabela.shape[1] - 1)
    p_ref = qui['p_fisher'] if qui['p_fisher'] is not None else qui['p_value']

    if not qui['premissa_ok']:
        motivo = (f"premissa das frequências esperadas violada "
                  f"({qui['pct_esperado_baixo']:.0f}% das células com esperado < 5)")
    elif p_ref >= alpha:
        motivo = f"associação não significativa (p = {p_ref:.4f})"
    elif dim_max < 2:
        motivo = (f"tabela {tabela.shape[0]}x{tabela.shape[1]} admite apenas "
                  f"{dim_max} dimensão; mapa 2D exige no mínimo 3x3")
    else:
        motivo = None

    if motivo:
        print(f"ANACOR NÃO APLICADA a '{nome_analise}': {motivo}.\n")
        return {'qui': qui, 'anacor': None, 'aplicada': False, 'motivo': motivo}

    print(f"Pré-requisitos atendidos (p = {p_ref:.4f}, V de Cramér = "
          f"{qui['cramers_v']:.3f}). Aplicando ANACOR.\n")
    res = anacor(df, var1, var2, nome_analise, salvar_grafico=salvar_grafico)
    return {'qui': qui, 'anacor': res, 'aplicada': True, 'motivo': None}


# ============================================================================
# REGRESSÃO LOGÍSTICA
# ============================================================================

def preparar_dados_regressao(df):
    """Prepara dados para regressão logística"""
    # mesma definição da bateria (preparar_variaveis): tempo individual até 6 meses
    y = df['curva_rapida_individual']

    # 4 preditores declarados na metodologia; porte e comunidades ficaram fora
    # (não acrescentam ajuste: teste LR 7 vs 4 preditores, p = 0,46). Se
    # comunidades voltar, usar df['participa_comunidades'] criada em
    # explodir_multipla_escolha — a opção do questionário é "Não participo",
    # e um str.contains('Não participa') nunca casa.
    X = pd.DataFrame()
    X['realizou_pocs'] = (df['q23_realizou_pocs'] == 'Sim').astype(int)
    X['possui_certificacao'] = (~df['q4_certificacoes'].str.contains('Não possuo', na=False)).astype(int)
    X['treinamento_formal'] = (df['q20_empresa_ofereceu_treinamento'] == 'Sim').astype(int)
    X['experiencia_conteineres'] = (df['q3_tempo_containers'].isin(['3-5 anos', 'Mais de 5 anos'])).astype(int)
    X = sm.add_constant(X)

    return X, y


def regressao_logistica(df):
    """
    Realiza Regressão Logística para identificar fatores preditivos

    Args:
        df (DataFrame): DataFrame com os dados

    Returns:
        dict: Resultados da regressão
    """
    # Mapeamento de nomes técnicos para descritivos
    nomes_variaveis = {
        'realizou_pocs': 'Realizou provas de conceito antes da produção',
        'possui_certificacao': 'Possui certificação técnica',
        'treinamento_formal': 'Empresa ofereceu treinamento formal',
        'experiencia_conteineres': 'Experiência com contêineres (3 anos ou mais)',
    }

    print(f"\n{'='*80}")
    print("REGRESSÃO LOGÍSTICA - FATORES PREDITIVOS DA CURVA DE APRENDIZADO")
    print(f"{'='*80}")

    print("\nMODELO:")
    print("  Variável Dependente: Curva Rápida individual (até 6 meses)")
    print("  Método: Regressão Logística Binária")

    print("\nVARIÁVEIS PREDITORAS:")
    for var_tech, var_desc in nomes_variaveis.items():
        print(f"  • {var_desc}")

    # Preparar dados
    X, y = preparar_dados_regressao(df)

    # Verificar variância das variáveis
    print(f"\n{'='*80}")
    print("RESUMO DOS DADOS")
    print(f"{'='*80}")
    print(f"  Total de observações: {len(y)}")
    print(f"  Curva rápida (até 6 meses): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Curva lenta (mais de 6 meses): {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

    # Verificar se há variação suficiente
    if y.sum() < 2 or (~y.astype(bool)).sum() < 2:
        print("\nAVISO: Dados insuficientes para regressão logística!")
        print("   Necessário pelo menos 2 casos de cada classe (Y=0 e Y=1)")
        print("   Recomendação: Coletar mais dados ou usar análises descritivas")
        return {
            'modelo': None,
            'odds_ratios': None,
            'X': X,
            'y': y,
            'erro': 'Dados insuficientes'
        }

    # Ajustar modelo com tratamento de erro
    try:
        modelo = Logit(y, X)
        # Usar método mais robusto para datasets pequenos
        resultado = modelo.fit(method='bfgs', maxiter=100, disp=0)

        # Exibir métricas do modelo
        print(f"\n{'='*80}")
        print("MÉTRICAS DO MODELO")
        print(f"{'='*80}")
        print(f"  Pseudo R² (McFadden): {resultado.prsquared:.4f}")
        print(f"  Log-Likelihood: {resultado.llf:.2f}")
        # modelo completo vs. modelo só com intercepto
        print(f"  Teste de razão de verossimilhança: LR = {resultado.llr:.3f} | "
              f"gl = {int(resultado.df_model)} | p = {resultado.llr_pvalue:.4f}")
        print(f"  AIC: {resultado.aic:.2f}")
        print(f"  BIC: {resultado.bic:.2f}")
        print(f"  Convergência: {'Sim' if resultado.mle_retvals['converged'] else 'Não'}")

        # Verificar separação quase-completa
        if any(resultado.pvalues > 0.95):
            print("\nALERTA: Separação Quase-Completa Detectada!")
            print("-" * 80)
            print("O modelo apresenta 'quasi-separation' - as variáveis preditoras separam")
            print("quase perfeitamente os casos, resultando em estimativas não confiáveis.")
            print("\nEste é um problema comum com:")
            print("  - Dados simulados (muito 'perfeitos')")
            print("  - Amostras pequenas com padrões muito claros")
            print("  - Variáveis altamente correlacionadas com o desfecho")
            print("\nRECOMENDAÇÕES:")
            print("  1. Usar análise descritiva (qui-quadrado, tabelas de contingência)")
            print("  2. Coletar dados reais com mais variabilidade")
            print("  3. Os Odds Ratios abaixo NÃO são interpretáveis estatisticamente")
            print("="*80)

        # Interpretação dos coeficientes
        print("\n" + "="*80)
        print("INTERPRETAÇÃO DOS COEFICIENTES (Odds Ratios)")
        print("="*80)

        odds_ratios = np.exp(resultado.params)
        ic_odds = np.exp(resultado.conf_int())  # IC 95% das razões de chance
        variaveis_significativas = []

        # Criar dados para tabela
        resultados_tabela = []
        for var, odds in odds_ratios.items():
            if var == 'const':
                continue

            p_value = resultado.pvalues[var]
            ic_inf, ic_sup = ic_odds.loc[var]
            if p_value < 0.05:
                significativo = "Sim"
                variaveis_significativas.append(nomes_variaveis.get(var, var))
            else:
                significativo = "Não"

            # Formatação melhorada
            if odds < 0.001:
                odds_str = f"{odds:.6f}"
            elif odds > 1000:
                odds_str = f"{odds:.1f}"
            else:
                odds_str = f"{odds:.3f}"

            resultados_tabela.append({
                'Variável': nomes_variaveis.get(var, var),
                'Razão de chances': odds_str,
                'IC 95%': f"[{ic_inf:.2f}; {ic_sup:.2f}]",
                'p-valor': f"{p_value:.4f}",
                'Significativo': significativo
            })

        df_resultados = pd.DataFrame(resultados_tabela)

        styled_resultados = (df_resultados.style
                            .hide(axis='index')
                            .set_properties(**{'text-align': 'left'})
                            .set_table_styles([
                                {'selector': 'th', 'props': [('text-align', 'left')]},
                                {'selector': '', 'props': [('border', '1px solid black')]},
                                {'selector': 'th', 'props': [('border', '1px solid black')]},
                                {'selector': 'td', 'props': [('border', '1px solid black')]}
                            ]))

        display(styled_resultados)

        if variaveis_significativas:
            print(f"\nVariáveis significativas (p < 0.05): {', '.join(variaveis_significativas)}")
        else:
            print("\nRESULTADO: Nenhuma variável significativa (p ≥ 0.05 para todas)")

        # Análise descritiva visual
        print("\n" + "="*80)
        print("ANÁLISE DESCRITIVA: Impacto Real de Cada Variável")
        print("="*80)

        # Criar tabela descritiva
        analise_desc = []
        for var in X.columns:
            if var == 'const':
                continue

            # Calcular proporções
            total_var_1 = X[var].sum()
            total_var_0 = len(X) - total_var_1

            # Curva rápida quando variável = 1
            curva_rapida_quando_1 = y[X[var] == 1].sum() if total_var_1 > 0 else 0
            pct_quando_1 = (curva_rapida_quando_1 / total_var_1 * 100) if total_var_1 > 0 else 0

            # Curva rápida quando variável = 0
            curva_rapida_quando_0 = y[X[var] == 0].sum() if total_var_0 > 0 else 0
            pct_quando_0 = (curva_rapida_quando_0 / total_var_0 * 100) if total_var_0 > 0 else 0

            analise_desc.append({
                'Variável': nomes_variaveis.get(var, var),
                'Com fator': f"{curva_rapida_quando_1}/{total_var_1} ({pct_quando_1:.1f}%)",
                'Sem fator': f"{curva_rapida_quando_0}/{total_var_0} ({pct_quando_0:.1f}%)",
                'Diferença': f"{pct_quando_1 - pct_quando_0:+.1f}pp"
            })

        df_analise = pd.DataFrame(analise_desc)

        styled_analise = (df_analise.style
                         .hide(axis='index')
                         .set_properties(**{'text-align': 'left'})
                         .set_table_styles([
                             {'selector': 'th', 'props': [('text-align', 'left')]},
                             {'selector': '', 'props': [('border', '1px solid black')]},
                             {'selector': 'th', 'props': [('border', '1px solid black')]},
                             {'selector': 'td', 'props': [('border', '1px solid black')]}
                         ]))

        display(styled_analise)

        print("\nComo interpretar:")
        print("  • 'Com fator': % de curva rápida quando a característica está presente")
        print("  • 'Sem fator': % de curva rápida quando a característica está ausente")
        print("  • 'Diferença': Impacto em pontos percentuais (pp)")
        print("  • Diferença positiva = fator favorece curva rápida")
        print("  • Diferença negativa = fator dificulta curva rápida")

        return {
            'modelo': resultado,
            'odds_ratios': odds_ratios,
            'X': X,
            'y': y,
            'quasi_separacao': any(resultado.pvalues > 0.95),
            'analise_descritiva': df_analise
        }

    except Exception as e:
        print(f"\nERRO ao ajustar modelo: {str(e)}")
        print("\nPOSSÍVEIS CAUSAS:")
        print("   1. Dataset muito pequeno (< 30 observações)")
        print("   2. Separação perfeita dos dados")
        print("   3. Multicolinearidade entre variáveis")
        print("\nSOLUÇÃO:")
        print("   - Coletar mais dados (recomendado: 100+ respostas)")
        print("   - Usar análises descritivas e qui-quadrado")
        print("   - Simplificar modelo (remover variáveis correlacionadas)")

        return {
            'modelo': None,
            'odds_ratios': None,
            'X': X,
            'y': y,
            'erro': str(e)
        }


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def salvar_resultados(resultados, nome_arquivo='resultados_analise.txt'):
    """
    Salva resumo dos resultados em arquivo texto

    Args:
        resultados (dict): Dicionário com resultados das análises
        nome_arquivo (str): Nome do arquivo de saída
    """
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMO DOS RESULTADOS - ANÁLISE ESTATÍSTICA\n")
        f.write("TCC - Engenharia de Software - MBA USP/ESALQ\n")
        f.write("="*80 + "\n\n")

        for chave, valor in resultados.items():
            f.write(f"\n{chave}:\n")
            f.write(f"{valor}\n")
            f.write("-"*80 + "\n")

    print(f"\n💾 Resultados salvos em: {nome_arquivo}")
