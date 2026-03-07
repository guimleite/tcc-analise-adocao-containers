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
from scipy.stats import chi2_contingency
import prince
from statsmodels.discrete.discrete_model import Logit
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
    desafios = {
        'Segurança': 'q12_desafio_seguranca',
        'Complexidade': 'q13_desafio_complexidade',
        'Monitoramento': 'q14_desafio_monitoramento',
        'Mudanças Organizacionais': 'q15_desafio_mudancas',
        'Treinamento': 'q16_desafio_treinamento',
        'CI/CD': 'q17_desafio_cicd'
    }

    print("\nESTATÍSTICAS DESCRITIVAS: Desafios (Escala Likert 1-5)\n")

    stats = pd.DataFrame({
        'Desafio': desafios.keys(),
        'Média': [df[col].mean() for col in desafios.values()],
        'Desvio': [df[col].std() for col in desafios.values()],
        'Min': [df[col].min() for col in desafios.values()],
        'Max': [df[col].max() for col in desafios.values()]
    })

    styled = (stats.style
              .hide(axis='index')
              .format({'Média': '{:.2f}', 'Desvio': '{:.2f}', 'Min': '{:.0f}', 'Max': '{:.0f}'})
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
# PRÉ-PROCESSAMENTO DE DADOS
# ============================================================================

def carregar_dados(arquivo_csv='../dados/respostas.csv'):
    """Carrega e renomeia colunas do CSV do Google Forms"""
    try:
        df = pd.read_csv(arquivo_csv, encoding='utf-8-sig')
    except (UnicodeDecodeError, FileNotFoundError):
        df = pd.read_csv(arquivo_csv, encoding='latin-1')
    return renomear_colunas(df)


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


# ============================================================================
# TESTE QUI-QUADRADO
# ============================================================================

def teste_qui_quadrado(df, var1, var2, nome_teste="Teste Qui-Quadrado"):
    """
    Realiza teste Qui-Quadrado de independência entre duas variáveis

    Args:
        df (DataFrame): DataFrame com os dados
        var1 (str): Nome da primeira variável
        var2 (str): Nome da segunda variável
        nome_teste (str): Nome descritivo do teste

    Returns:
        dict: Resultados do teste (chi2, p-value, graus de liberdade, tabela)
    """
    print(f"\n{'='*80}")
    print(f"{nome_teste.upper()}")
    print(f"{'='*80}\n")

    # Criar e exibir tabela de contingência formatada
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

    # Aplicar teste qui-quadrado
    chi2, p_value, dof, expected = chi2_contingency(tabela)

    print(f"\nQui-quadrado: {chi2:.4f} | P-value: {p_value:.4f} | Graus de liberdade: {dof}")

    # Interpretação
    alpha = 0.05
    if p_value < alpha:
        print(f"Associação significativa (p < {alpha}) - Variáveis NÃO são independentes\n")
    else:
        print(f"Associação NÃO significativa (p >= {alpha}) - Variáveis são independentes\n")

    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'tabela': tabela,
        'expected': expected
    }


# ============================================================================
# ANACOR (ANÁLISE DE CORRESPONDÊNCIA)
# ============================================================================

def anacor(df, var1, var2, nome_analise="ANACOR", salvar_grafico=True):
    """
    Realiza Análise de Correspondência (ANACOR)

    Args:
        df (DataFrame): DataFrame com os dados
        var1 (str): Nome da primeira variável
        var2 (str): Nome da segunda variável
        nome_analise (str): Nome descritivo da análise
        salvar_grafico (bool): Se True, salva o gráfico em arquivo

    Returns:
        dict: Resultados da ANACOR (coordenadas, inércia)
    """
    print(f"\n{'='*80}")
    print(f"ANACOR: {nome_analise.upper()}")
    print(f"{'='*80}\n")

    # Criar e exibir tabela de contingência formatada
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

    # Aplicar ANACOR
    ca = prince.CA(n_components=2, n_iter=3, copy=True, engine='sklearn')
    ca = ca.fit(tabela)

    # Exibir inércia explicada
    print(f"\nInércia Explicada por Dimensão:")
    if hasattr(ca, 'eigenvalues_'):
        total_inercia = ca.eigenvalues_.sum()
        for i, inercia in enumerate(ca.eigenvalues_[:2]):
            pct = (inercia / total_inercia) * 100
            print(f"   Dimensão {i+1}: {pct:.2f}%")
        print(f"   Total acumulado: {sum((ca.eigenvalues_[:2] / total_inercia) * 100):.2f}%")

    # Obter coordenadas
    coord_linhas = ca.row_coordinates(tabela)
    coord_colunas = ca.column_coordinates(tabela)

    # Criar mapa perceptual
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plotar linhas (primeira variável)
    ax.scatter(coord_linhas[0], coord_linhas[1],
              s=200, c='steelblue', marker='o',
              edgecolors='black', linewidth=1.5,
              alpha=0.8, label=var1)

    for idx, txt in enumerate(coord_linhas.index):
        ax.annotate(txt, (coord_linhas[0].iloc[idx], coord_linhas[1].iloc[idx]),
                   fontsize=9, fontweight='bold', ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # Plotar colunas (segunda variável)
    ax.scatter(coord_colunas[0], coord_colunas[1],
              s=200, c='coral', marker='s',
              edgecolors='black', linewidth=1.5,
              alpha=0.8, label=var2)

    for idx, txt in enumerate(coord_colunas.index):
        ax.annotate(txt, (coord_colunas[0].iloc[idx], coord_colunas[1].iloc[idx]),
                   fontsize=9, fontweight='bold', ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    # Adicionar linhas de referência
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Configurar labels e título
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
        'tabela': tabela
    }


# ============================================================================
# REGRESSÃO LOGÍSTICA
# ============================================================================

def preparar_dados_regressao(df):
    """Prepara dados para regressão logística"""
    df['curva_rapida'] = (df['q24_tempo_individual'] == 'Menos de 3 meses').astype(int)

    X = pd.DataFrame()
    X['empresa_grande'] = (df['q6_porte_empresa'] == 'Grande (mais de 250 funcionários)').astype(int)
    X['empresa_media'] = (df['q6_porte_empresa'] == 'Média (50-250 funcionários)').astype(int)
    X['possui_certificacao'] = (~df['q4_certificacoes'].str.contains('Não possuo', na=False)).astype(int)
    X['experiencia_docker'] = (df['q3_tempo_containers'].isin(['3-5 anos', 'Mais de 5 anos'])).astype(int)
    X['participa_comunidades'] = (~df['q5_comunidades'].str.contains('Não participa', na=False)).astype(int)
    X['treinamento_formal'] = (df['q20_empresa_ofereceu_treinamento'] == 'Sim').astype(int)
    X['realizou_pocs'] = (df['q23_realizou_pocs'] == 'Sim').astype(int)
    X = sm.add_constant(X)

    return X, df['curva_rapida']


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
        'empresa_grande': 'Empresa Grande (>250 funcionários)',
        'empresa_media': 'Empresa Média (50-250 funcionários)',
        'possui_certificacao': 'Possui Certificações',
        'experiencia_docker': 'Experiência com Containers (3+ anos)',
        'participa_comunidades': 'Participa de Comunidades Técnicas',
        'treinamento_formal': 'Empresa Ofereceu Treinamento',
        'realizou_pocs': 'Realizou POCs antes de Produção'
    }

    print(f"\n{'='*80}")
    print("REGRESSÃO LOGÍSTICA - FATORES PREDITIVOS DA CURVA DE APRENDIZADO")
    print(f"{'='*80}")

    print("\nMODELO:")
    print("  Variável Dependente: Curva Rápida (< 3 meses)")
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
    print(f"  Curva rápida (<3 meses): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Curva lenta (≥3 meses): {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

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
        print(f"  Pseudo R²: {resultado.prsquared:.4f}")
        print(f"  Log-Likelihood: {resultado.llf:.2f}")
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
        variaveis_significativas = []

        # Criar dados para tabela
        resultados_tabela = []
        for var, odds in odds_ratios.items():
            if var == 'const':
                continue

            p_value = resultado.pvalues[var]
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
                'Odds Ratio': odds_str,
                'P-value': f"{p_value:.4f}",
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
