#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise Estatística - TCC Engenharia de Software
Tecnologias de Conteinerização: Curva de Aprendizagem Organizacional

Autor: Guilherme Magalhães Leite
Orientador: Arthur Pinheiro de Araújo Costa
Instituição: MBA USP/ESALQ - Engenharia de Software

Este script contém o MESMO código do notebook Jupyter (analise_estatistica.ipynb)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import prince  # Para ANACOR
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm
import csv
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

print("="*80)
print("ANÁLISE ESTATÍSTICA - TCC")
print("Tecnologias de Conteinerização: Curva de Aprendizagem Organizacional")
print("="*80)
print("✅ Bibliotecas importadas com sucesso!\n")

# ==============================================================================
# 1. PRÉ-PROCESSAMENTO DO CSV
# ==============================================================================

def preprocessar_csv(arquivo_original='dados/brutos/respostas_simuladas.csv'):
    """
    Lê e corrige o CSV usando csv.reader (MESMA lógica do notebook)
    """
    print("="*80)
    print("1. PRÉ-PROCESSAMENTO DO CSV")
    print("="*80)

    arquivo_corrigido = 'dados/processados/respostas_corrigido.csv'

    print(f"🔧 Processando CSV com parser robusto...")

    # Ler arquivo usando csv.reader
    linhas_lidas = []
    with open(arquivo_original, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"',
                           skipinitialspace=True, strict=False)

        for i, row in enumerate(reader):
            linhas_lidas.append(row)
            if i == 0:
                num_colunas = len(row)
                print(f"📋 Header: {num_colunas} colunas")

    print(f"📊 Total de linhas lidas: {len(linhas_lidas)}")

    # Verificar consistência
    linhas_validas = []
    linhas_problematicas = []

    for i, linha in enumerate(linhas_lidas):
        if i == 0:
            linhas_validas.append(linha)  # Header
            continue

        if len(linha) == num_colunas:
            linhas_validas.append(linha)
        else:
            linhas_problematicas.append((i+1, len(linha)))

    print(f"✅ Linhas válidas: {len(linhas_validas)-1}")
    print(f"⚠️  Linhas problemáticas: {len(linhas_problematicas)}")

    if linhas_problematicas:
        print(f"\nPrimeiras 5 linhas com problemas:")
        for linha_num, num_campos in linhas_problematicas[:5]:
            print(f"   Linha {linha_num}: {num_campos} campos (esperado {num_colunas})")

    # Salvar CSV corrigido
    with open(arquivo_corrigido, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(linhas_validas)

    print(f"\n✅ CSV corrigido salvo: {arquivo_corrigido}")
    print(f"📊 Dados válidos: {len(linhas_validas)-1} respostas\n")

    return arquivo_corrigido


# ==============================================================================
# 2. CARREGAMENTO DE DADOS
# ==============================================================================

def carregar_dados(arquivo):
    """
    Carrega CSV usando pandas (MESMA lógica do notebook)
    """
    print("="*80)
    print("2. CARREGAMENTO DE DADOS")
    print("="*80)

    df = pd.read_csv(arquivo, encoding='utf-8')

    print(f"✅ Dados carregados com sucesso!")
    print(f"📊 Total de respostas: {len(df)}")
    print(f"📊 Total de colunas: {len(df.columns)}")

    # Renomear colunas para nomes curtos (facilita manipulação)
    nomes_curtos = {
        df.columns[0]: 'timestamp',
        df.columns[1]: 'q1_cargo',
        df.columns[2]: 'q2_tempo_ti',
        df.columns[3]: 'q3_tempo_containers',
        df.columns[4]: 'q4_certificacoes',
        df.columns[5]: 'q5_comunidades',
        df.columns[6]: 'q6_porte_empresa',
        df.columns[7]: 'q7_setor',
        df.columns[8]: 'q8_tempo_containers_producao',
        df.columns[9]: 'q9_tecnologia_orquestracao',
        df.columns[10]: 'q10_ambientes',
        df.columns[11]: 'q11_processo_adocao',
        df.columns[12]: 'q12_desafio_seguranca',
        df.columns[13]: 'q13_desafio_complexidade',
        df.columns[14]: 'q14_desafio_monitoramento',
        df.columns[15]: 'q15_desafio_networking',
        df.columns[16]: 'q16_desafio_gestao_estado',
        df.columns[17]: 'q17_principal_desafio',
        df.columns[18]: 'q18_estrategias_aprendizado',
        df.columns[19]: 'q19_empresa_ofereceu_treinamento',
        df.columns[20]: 'q20_alocacao_tempo',
        df.columns[21]: 'q21_fonte_mais_util',
        df.columns[22]: 'q22_realizou_pocs',
        df.columns[23]: 'q23_tempo_individual',
        df.columns[24]: 'q24_tempo_equipe',
        df.columns[25]: 'q25_fator_mais_importante',
        df.columns[26]: 'q26_roi',
        df.columns[27]: 'q27_recomendaria'
    }

    df = df.rename(columns=nomes_curtos)

    print(f"✅ Colunas renomeadas para nomes curtos (q1, q2, ...)")

    if len(df) > 0:
        print(f"\nPrimeiras 3 linhas:")
        print(df.head(3))
    else:
        print("\n⚠️  ATENÇÃO: Nenhuma linha de dados carregada!")
        print("   O CSV pode estar malformatado. Verifique o arquivo original.")

    print()
    return df


# ==============================================================================
# 3. ANÁLISE DESCRITIVA
# ==============================================================================

def analise_descritiva(df):
    """
    Análise exploratória completa (MESMA lógica do notebook)
    """
    if len(df) == 0:
        print("❌ Não há dados para analisar!\n")
        return

    print("="*80)
    print("3. ANÁLISE DESCRITIVA")
    print("="*80)

    # 3.1 Perfil Profissional
    print("\n" + "="*80)
    print("SEÇÃO 1: PERFIL PROFISSIONAL")
    print("="*80)

    print("\n1.1. Distribuição por Cargo:")
    print(df['q1_cargo'].value_counts())

    print("\n1.2. Tempo de Experiência em TI:")
    print(df['q2_tempo_ti'].value_counts())

    print("\n1.3. Tempo com Containers/Orquestradores:")
    print(df['q3_tempo_containers'].value_counts())

    cert_sem = (df['q4_certificacoes'] == 'Não possuo certificações').sum()
    cert_com = len(df) - cert_sem
    print(f"\n1.4. Certificações:")
    print(f"Com certificações: {cert_com} ({cert_com/len(df)*100:.1f}%)")
    print(f"Sem certificações: {cert_sem} ({cert_sem/len(df)*100:.1f}%)")

    # 3.2 Contexto Empresarial
    print("\n" + "="*80)
    print("SEÇÃO 2: CONTEXTO EMPRESARIAL")
    print("="*80)

    print("\n2.1. Porte da Empresa:")
    print(df['q6_porte_empresa'].value_counts())

    print("\n2.2. Setor de Atuação:")
    print(df['q7_setor'].value_counts())

    print("\n2.3. Tecnologia de Orquestração:")
    print(df['q9_tecnologia_orquestracao'].value_counts())

    # 3.3 Desafios
    print("\n" + "="*80)
    print("SEÇÃO 3: DESAFIOS ENFRENTADOS (Escala Likert 1-5)")
    print("="*80)

    desafios_cols = ['q12_desafio_seguranca', 'q13_desafio_complexidade',
                     'q14_desafio_monitoramento', 'q15_desafio_networking',
                     'q16_desafio_gestao_estado']
    desafios_labels = ['Segurança', 'Complexidade Técnica', 'Monitoramento',
                       'Networking', 'Gestão de Estado']

    print("\n3.1. Médias por Desafio:")
    for col, label in zip(desafios_cols, desafios_labels):
        if df[col].dtype in ['int64', 'float64']:
            print(f"   {label}: {df[col].mean():.2f}")

    print("\n3.2. Principal Desafio Identificado:")
    print(df['q17_principal_desafio'].value_counts())

    # 3.4 Estratégias
    print("\n" + "="*80)
    print("SEÇÃO 4: ESTRATÉGIAS DE APRENDIZADO")
    print("="*80)

    print("\n4.1. Empresa ofereceu treinamento:")
    print(df['q19_empresa_ofereceu_treinamento'].value_counts())

    print("\n4.2. Fonte mais útil:")
    print(df['q21_fonte_mais_util'].value_counts())

    # 3.5 Curva de Aprendizado
    print("\n" + "="*80)
    print("SEÇÃO 5: CURVA DE APRENDIZADO E PERCEPÇÕES")
    print("="*80)

    print("\n5.1. Tempo INDIVIDUAL:")
    print(df['q23_tempo_individual'].value_counts())

    print("\n5.2. Tempo da EQUIPE:")
    print(df['q24_tempo_equipe'].value_counts())

    print(f"\n5.3. ROI médio: {df['q26_roi'].mean():.2f} (escala 1-5)")

    recom_sim = (df['q27_recomendaria'] == 'Sim').sum()
    print(f"\n5.4. Recomendaria: {recom_sim}/{len(df)} ({recom_sim/len(df)*100:.1f}%)")

    print()


# ==============================================================================
# 4. TESTE QUI-QUADRADO
# ==============================================================================

def teste_qui_quadrado(df, var1, var2, nome_teste):
    """
    Teste qui-quadrado de independência (MESMA lógica do notebook)
    """
    print(f"\n{'='*80}")
    print(f"TESTE QUI-QUADRADO: {nome_teste}")
    print(f"{'='*80}")

    tabela = pd.crosstab(df[var1], df[var2])
    print("\nTabela de Contingência:")
    print(tabela)

    chi2, p_value, dof, expected = chi2_contingency(tabela)

    print(f"\nResultados:")
    print(f"Chi-quadrado: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Graus de liberdade: {dof}")

    if p_value < 0.05:
        print(f"\n✅ Associação significativa (p < 0.05)")
        print(f"   → Rejeita-se H0: as variáveis são DEPENDENTES")
    else:
        print(f"\n❌ Não há associação significativa (p >= 0.05)")
        print(f"   → Não se rejeita H0: as variáveis são INDEPENDENTES")

    return {'chi2': chi2, 'p_value': p_value, 'dof': dof, 'tabela': tabela}


def aplicar_testes_qui_quadrado(df):
    """
    Aplica todos os testes qui-quadrado (MESMA lógica do notebook)
    """
    if len(df) == 0:
        print("❌ Não há dados para testes qui-quadrado!\n")
        return

    print("="*80)
    print("4. TESTE QUI-QUADRADO DE INDEPENDÊNCIA")
    print("="*80)

    try:
        teste_qui_quadrado(df, 'q6_porte_empresa', 'q17_principal_desafio',
                          'Porte da Empresa × Principal Desafio')
    except Exception as e:
        print(f"❌ Erro: {e}")

    try:
        teste_qui_quadrado(df, 'q19_empresa_ofereceu_treinamento', 'q23_tempo_individual',
                          'Treinamento Oferecido × Tempo Individual')
    except Exception as e:
        print(f"❌ Erro: {e}")

    try:
        teste_qui_quadrado(df, 'q22_realizou_pocs', 'q24_tempo_equipe',
                          'Realização de POCs × Tempo da Equipe')
    except Exception as e:
        print(f"❌ Erro: {e}")

    print()


# ==============================================================================
# 5. ANACOR
# ==============================================================================

def anacor(df, var1, var2, nome_analise, salvar=True):
    """
    Análise de Correspondência (MESMA lógica do notebook)
    """
    print(f"\n{'='*80}")
    print(f"ANACOR: {nome_analise}")
    print(f"{'='*80}")

    tabela = pd.crosstab(df[var1], df[var2])
    print("\nTabela de Contingência:")
    print(tabela)

    ca = prince.CA(n_components=2, n_iter=3, copy=True, engine='auto')
    ca = ca.fit(tabela)

    print(f"\nInércia explicada:")
    for i, inercia in enumerate(ca.explained_inertia_):
        print(f"Dimensão {i+1}: {inercia:.2%}")

    coord_linhas = ca.row_coordinates(tabela)
    coord_colunas = ca.column_coordinates(tabela)

    # Plotar mapa perceptual
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(coord_linhas[0], coord_linhas[1], s=200, c='blue',
              alpha=0.6, edgecolors='black', label=var1, marker='o')

    for i, txt in enumerate(coord_linhas.index):
        ax.annotate(txt, (coord_linhas[0].iloc[i], coord_linhas[1].iloc[i]),
                   fontsize=9, ha='right', va='bottom', color='darkblue',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))

    ax.scatter(coord_colunas[0], coord_colunas[1], s=200, c='red',
              alpha=0.6, edgecolors='black', label=var2, marker='s')

    for i, txt in enumerate(coord_colunas.index):
        ax.annotate(txt, (coord_colunas[0].iloc[i], coord_colunas[1].iloc[i]),
                   fontsize=9, ha='left', va='top', color='darkred',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.5))

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    ax.set_xlabel(f'Dimensão 1 ({ca.explained_inertia_[0]:.1%})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Dimensão 2 ({ca.explained_inertia_[1]:.1%})', fontsize=12, fontweight='bold')
    ax.set_title(f'Mapa Perceptual - ANACOR\n{nome_analise}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if salvar:
        os.makedirs('resultados/graficos/anacor', exist_ok=True)
        nome_arquivo = nome_analise.replace(' ', '_').replace('×', 'x').lower()
        plt.savefig(f'resultados/graficos/anacor/anacor_{nome_arquivo}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Salvo: resultados/graficos/anacor/anacor_{nome_arquivo}.png")
    else:
        plt.show()

    plt.close()
    return ca


def aplicar_anacor(df):
    """
    Aplica todas as análises ANACOR (MESMA lógica do notebook)
    """
    if len(df) == 0:
        print("❌ Não há dados para ANACOR!\n")
        return

    print("="*80)
    print("5. ANACOR (ANÁLISE DE CORRESPONDÊNCIA)")
    print("="*80)

    try:
        anacor(df, 'q6_porte_empresa', 'q17_principal_desafio',
              'Porte da Empresa × Principal Desafio')
    except Exception as e:
        print(f"❌ Erro: {e}")

    try:
        anacor(df, 'q21_fonte_mais_util', 'q23_tempo_individual',
              'Fonte Mais Útil × Tempo Individual')
    except Exception as e:
        print(f"❌ Erro: {e}")

    try:
        anacor(df, 'q25_fator_mais_importante', 'q24_tempo_equipe',
              'Fator Mais Importante × Tempo da Equipe')
    except Exception as e:
        print(f"❌ Erro: {e}")

    print()


# ==============================================================================
# 6. REGRESSÃO LOGÍSTICA
# ==============================================================================

def regressao_logistica(df):
    """
    Regressão logística completa (MESMA lógica do notebook)
    """
    if len(df) == 0:
        print("❌ Não há dados para regressão!\n")
        return

    print("="*80)
    print("6. REGRESSÃO LOGÍSTICA")
    print("="*80)

    df_reg = df.copy()

    # Criar VD binária
    curva_rapida = []
    for tempo in df_reg['q23_tempo_individual']:
        if tempo in ['Menos de 3 meses', '3-6 meses']:
            curva_rapida.append(1)
        else:
            curva_rapida.append(0)

    df_reg['curva_rapida'] = curva_rapida

    print(f"\nDistribuição VD (curva_rapida):")
    print(f"Rápida (< 6 meses): {sum(curva_rapida)} ({sum(curva_rapida)/len(curva_rapida)*100:.1f}%)")
    print(f"Lenta (>= 6 meses): {len(curva_rapida) - sum(curva_rapida)}")

    # Criar VIs binárias
    df_reg['possui_cert'] = (df_reg['q4_certificacoes'] != 'Não possuo certificações').astype(int)
    df_reg['participa_comunidades'] = (df_reg['q5_comunidades'] != 'Não participo de comunidades técnicas').astype(int)
    df_reg['empresa_treinou'] = (df_reg['q19_empresa_ofereceu_treinamento'] == 'Sim').astype(int)
    df_reg['realizou_pocs'] = (df_reg['q22_realizou_pocs'] == 'Sim').astype(int)

    # Criar dummies
    porte_dummies = pd.get_dummies(df_reg['q6_porte_empresa'], prefix='porte', drop_first=True)
    setor_dummies = pd.get_dummies(df_reg['q7_setor'], prefix='setor', drop_first=True)

    # Dataset final
    df_final = pd.concat([
        df_reg[['curva_rapida', 'possui_cert', 'participa_comunidades',
                'empresa_treinou', 'realizou_pocs']],
        porte_dummies,
        setor_dummies
    ], axis=1)

    print(f"\nDataset: {len(df_final)} linhas, {len(df_final.columns)} colunas")

    # Modelo
    y = df_final['curva_rapida']
    X = df_final.drop('curva_rapida', axis=1)
    X = sm.add_constant(X)

    modelo = Logit(y, X)
    resultado = modelo.fit(disp=False)

    print("\n" + "="*80)
    print("RESULTADOS DA REGRESSÃO LOGÍSTICA")
    print("="*80)
    print(resultado.summary())

    # Odds Ratios
    odds_ratios = np.exp(resultado.params)
    odds_df = pd.DataFrame({
        'Variável': odds_ratios.index,
        'Odds Ratio': odds_ratios.values,
        'P-value': resultado.pvalues.values
    })

    print("\n" + "="*80)
    print("ODDS RATIOS")
    print("="*80)
    print(odds_df.sort_values('Odds Ratio', ascending=False))

    print()


# ==============================================================================
# 7. VISUALIZAÇÕES
# ==============================================================================

def gerar_visualizacoes(df):
    """
    Gera todos os gráficos (MESMA lógica do notebook)
    """
    if len(df) == 0:
        print("❌ Não há dados para gráficos!\n")
        return

    print("="*80)
    print("7. VISUALIZAÇÕES")
    print("="*80)

    os.makedirs('resultados/graficos/descritivas', exist_ok=True)

    # Gráfico 1: Desafios médios
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        desafios = ['Segurança', 'Complexidade\nTécnica', 'Monitoramento',
                   'Networking', 'Gestão de\nEstado']
        medias = [
            df['q12_desafio_seguranca'].mean(),
            df['q13_desafio_complexidade'].mean(),
            df['q14_desafio_monitoramento'].mean(),
            df['q15_desafio_networking'].mean(),
            df['q16_desafio_gestao_estado'].mean()
        ]

        colors = sns.color_palette("husl", 5)
        bars = ax.bar(desafios, medias, color=colors, edgecolor='black', linewidth=1.5)

        ax.axhline(3, color='red', linestyle='--', linewidth=2, label='Moderado (3.0)', alpha=0.7)
        ax.set_ylabel('Média (Escala Likert 1-5)', fontsize=12, fontweight='bold')
        ax.set_title('Nível Médio de Desafios Enfrentados', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 5)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('resultados/graficos/descritivas/desafios_medios.png', dpi=300, bbox_inches='tight')
        print("✅ Salvo: resultados/graficos/descritivas/desafios_medios.png")
        plt.close()

    except Exception as e:
        print(f"❌ Erro: {e}")

    # Gráfico 2: Curva de aprendizado
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        tempo_ind_count = df['q23_tempo_individual'].value_counts()
        ordem_tempo = ['Menos de 3 meses', '3-6 meses', '6-12 meses', '12-18 meses', 'Mais de 18 meses']
        tempo_ind_ordenado = tempo_ind_count.reindex(ordem_tempo, fill_value=0)

        axes[0].bar(range(len(tempo_ind_ordenado)), tempo_ind_ordenado.values,
                   color=sns.color_palette("Blues_d", len(tempo_ind_ordenado)),
                   edgecolor='black', linewidth=1.2)
        axes[0].set_xticks(range(len(tempo_ind_ordenado)))
        axes[0].set_xticklabels(tempo_ind_ordenado.index, rotation=45, ha='right')
        axes[0].set_ylabel('Frequência', fontsize=12, fontweight='bold')
        axes[0].set_title('Tempo Individual', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        for i, v in enumerate(tempo_ind_ordenado.values):
            if v > 0:
                axes[0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

        tempo_eq_count = df['q24_tempo_equipe'].value_counts()
        ordem_eq = ['Menos de 1 mês', '1-3 meses', '3-6 meses', '6-12 meses', 'Mais de 12 meses']
        tempo_eq_ordenado = tempo_eq_count.reindex(ordem_eq, fill_value=0)

        axes[1].bar(range(len(tempo_eq_ordenado)), tempo_eq_ordenado.values,
                   color=sns.color_palette("Greens_d", len(tempo_eq_ordenado)),
                   edgecolor='black', linewidth=1.2)
        axes[1].set_xticks(range(len(tempo_eq_ordenado)))
        axes[1].set_xticklabels(tempo_eq_ordenado.index, rotation=45, ha='right')
        axes[1].set_ylabel('Frequência', fontsize=12, fontweight='bold')
        axes[1].set_title('Tempo da Equipe', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        for i, v in enumerate(tempo_eq_ordenado.values):
            if v > 0:
                axes[1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('resultados/graficos/descritivas/curva_aprendizado.png', dpi=300, bbox_inches='tight')
        print("✅ Salvo: resultados/graficos/descritivas/curva_aprendizado.png")
        plt.close()

    except Exception as e:
        print(f"❌ Erro: {e}")

    # Gráfico 3: ROI
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        roi_count = df['q26_roi'].value_counts().sort_index()
        labels_roi = ['1 - Muito\nNegativo', '2 - Negativo', '3 - Neutro',
                     '4 - Positivo', '5 - Muito\nPositivo']
        colors_roi = ['darkred', 'orange', 'gray', 'lightgreen', 'darkgreen']

        ax.bar(range(1, 6), [roi_count.get(i, 0) for i in range(1, 6)],
               color=colors_roi, edgecolor='black', linewidth=1.5)

        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(labels_roi)
        ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
        ax.set_title(f'Avaliação do ROI (Média: {df["q26_roi"].mean():.2f})',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for i in range(1, 6):
            val = roi_count.get(i, 0)
            ax.text(i, val, str(val), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('resultados/graficos/descritivas/roi.png', dpi=300, bbox_inches='tight')
        print("✅ Salvo: resultados/graficos/descritivas/roi.png")
        plt.close()

    except Exception as e:
        print(f"❌ Erro: {e}")

    print()


# ==============================================================================
# 8. RESUMO
# ==============================================================================

def gerar_resumo(df):
    """
    Gera resumo executivo (MESMA lógica do notebook)
    """
    if len(df) == 0:
        print("❌ Não há dados para resumo!\n")
        return

    print("="*80)
    print("8. RESUMO EXECUTIVO DOS RESULTADOS")
    print("="*80)

    print("\n📊 ANÁLISE DESCRITIVA")
    print("-" * 80)
    print(f"• Total de respondentes: {len(df)}")

    if len(df['q6_porte_empresa'].mode()) > 0:
        print(f"• Porte predominante: {df['q6_porte_empresa'].mode()[0]}")
    if len(df['q7_setor'].mode()) > 0:
        print(f"• Setor predominante: {df['q7_setor'].mode()[0]}")
    if len(df['q17_principal_desafio'].mode()) > 0:
        print(f"• Principal desafio: {df['q17_principal_desafio'].mode()[0]}")

    print(f"• ROI médio: {df['q26_roi'].mean():.2f}")
    recom_pct = (df['q27_recomendaria'] == 'Sim').sum() / len(df) * 100
    print(f"• Recomendaria: {recom_pct:.1f}%")

    print("\n📊 CURVA DE APRENDIZADO")
    print("-" * 80)
    curva_rapida = df['q23_tempo_individual'].isin(['Menos de 3 meses', '3-6 meses']).sum()
    print(f"• Curva rápida (< 6 meses): {curva_rapida/len(df)*100:.1f}%")

    print("\n" + "="*80)
    print("✅ ANÁLISE COMPLETA FINALIZADA!")
    print("="*80)
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """
    Função principal - executa todo o pipeline de análise
    """
    print("\n🚀 Iniciando análise completa...\n")

    # 1. Pré-processar CSV
    arquivo_corrigido = preprocessar_csv()

    # 2. Carregar dados
    df = carregar_dados(arquivo_corrigido)

    # 3. Análise descritiva
    analise_descritiva(df)

    # 4. Testes qui-quadrado
    aplicar_testes_qui_quadrado(df)

    # 5. ANACOR
    aplicar_anacor(df)

    # 6. Regressão Logística
    regressao_logistica(df)

    # 7. Visualizações
    gerar_visualizacoes(df)

    # 8. Resumo
    gerar_resumo(df)

    print("="*80)
    print("📊 RESULTADOS SALVOS")
    print("="*80)
    print("  - Gráficos: resultados/graficos/")
    print("  - CSV corrigido: dados/processados/respostas_corrigido.csv")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
