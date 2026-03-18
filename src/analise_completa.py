#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise Estatística Completa - TCC Engenharia de Software
Tecnologias de Conteinerização: Curva de Aprendizagem Organizacional

Autor: Guilherme Magalhães Leite
Orientador: Arthur Pinheiro de Araújo Costa
Instituição: MBA USP/ESALQ - Engenharia de Software

Este script executa EXATAMENTE as mesmas análises do notebook 00_master_analise.ipynb
Espelha a lógica dos notebooks individuais usando o módulo utils.py
"""

import sys
import os
import pandas as pd
import warnings

# Importar módulo utils
sys.path.append(os.path.dirname(__file__))
import utils

warnings.filterwarnings('ignore')

# ============================================================================
# BANNER INICIAL
# ============================================================================

print("\n" + "="*80)
print("ANÁLISE ESTATÍSTICA - TCC")
print("Tecnologias de Conteinerização: Curva de Aprendizagem Organizacional")
print("="*80)
print("Autor: Guilherme Magalhães Leite")
print("Orientador: Arthur Pinheiro de Araújo Costa")
print("Instituição: MBA USP/ESALQ - Engenharia de Software")
print("="*80 + "\n")

# ============================================================================
# CONFIGURAÇÃO INICIAL
# ============================================================================

utils.configurar_visualizacoes()
print("Configurações de visualização aplicadas\n")

# ============================================================================
# CARREGAR DADOS
# ============================================================================

utils.titulo_secao("Carregamento de Dados")

# Ajustar caminho relativo (script está em src/, dados em dados/)
caminho_dados = os.path.join(os.path.dirname(__file__), '..', 'dados', 'respostas.csv')
df = utils.carregar_dados(caminho_dados)

print(f"\nDataset carregado: {len(df)} respostas × {len(df.columns)} variáveis")
print(f"Período: {df['timestamp'].iloc[0]} até {df['timestamp'].iloc[-1]}")

# ============================================================================
# 1. ANÁLISE DESCRITIVA
# ============================================================================

utils.titulo_secao("1. Análise Descritiva")

print("\n--- PERFIL PROFISSIONAL ---\n")

utils.tabela_frequencia(df['q1_cargo'], "DISTRIBUIÇÃO: Cargo", "Cargo")
utils.tabela_frequencia(df['q2_tempo_ti'], "DISTRIBUIÇÃO: Tempo de Experiência em TI", "Tempo")
utils.tabela_frequencia(df['q3_tempo_containers'], "DISTRIBUIÇÃO: Tempo de Experiência com Containers", "Tempo")

print("\n--- CONTEXTO EMPRESARIAL ---\n")

utils.tabela_frequencia(df['q6_porte_empresa'], "DISTRIBUIÇÃO: Porte da Empresa", "Porte")
utils.tabela_frequencia(df['q7_setor'], "DISTRIBUIÇÃO: Setor de Atuação", "Setor")
utils.tabela_frequencia(df['q9_tecnologia_orquestracao'], "DISTRIBUIÇÃO: Tecnologia de Orquestração", "Tecnologia")

print("\n--- DESAFIOS ENFRENTADOS ---\n")

utils.tabela_desafios_likert(df)
utils.tabela_frequencia(df['q18_principal_desafio'], "DISTRIBUIÇÃO: Principal Desafio Enfrentado", "Desafio")

print("\n--- ESTRATÉGIAS DE APRENDIZADO ---\n")

utils.tabela_frequencia(df['q20_empresa_ofereceu_treinamento'], "DISTRIBUIÇÃO: Empresa Ofereceu Treinamento?", "Resposta")
utils.tabela_frequencia(df['q22_fonte_mais_util'], "DISTRIBUIÇÃO: Fonte Mais Útil de Aprendizado", "Fonte")
utils.tabela_frequencia(df['q23_realizou_pocs'], "DISTRIBUIÇÃO: Realizou POCs?", "Resposta")

print("\n--- CURVA DE APRENDIZADO ---\n")

utils.tabela_frequencia(df['q24_tempo_individual'], "DISTRIBUIÇÃO: Tempo Individual para se Sentir Confortável", "Tempo")
utils.tabela_frequencia(df['q25_tempo_equipe'], "DISTRIBUIÇÃO: Tempo da Equipe para Colocar em Produção", "Tempo")
utils.tabela_frequencia(df['q26_fator_mais_importante'], "DISTRIBUIÇÃO: Fator Mais Importante para Acelerar Aprendizado", "Fator")

print("\n--- PERCEPÇÃO DE ROI E RECOMENDAÇÃO ---\n")

print(f"DISTRIBUIÇÃO: Avaliação do ROI\n")
print(f"Média: {df['q27_roi'].mean():.2f} | Mediana: {df['q27_roi'].median():.2f}\n")

utils.tabela_frequencia(df['q28_recomendaria'], "DISTRIBUIÇÃO: Recomendaria a Adoção de Containers?", "Resposta")

print("\n--- TABELAS DE CONTINGÊNCIA ---\n")

utils.tabela_contingencia(
    df, 'q24_tempo_individual', 'q22_fonte_mais_util',
    "TABELA DE CONTINGÊNCIA: Tempo para se Sentir Confortável × Fonte de Aprendizado Mais Útil"
)

utils.tabela_contingencia(
    df, 'q23_realizou_pocs', 'q25_tempo_equipe',
    "TABELA DE CONTINGÊNCIA: Realização de POCs × Tempo da Equipe para Produção"
)

utils.titulo_secao("Análise Descritiva Concluída")

# ============================================================================
# 2. TESTE QUI-QUADRADO
# ============================================================================

utils.titulo_secao("2. Teste Qui-Quadrado")

# Teste 1: Porte da Empresa × Principal Desafio
utils.teste_qui_quadrado(
    df, 'q6_porte_empresa', 'q18_principal_desafio',
    "Porte da Empresa × Principal Desafio"
)

# Teste 2: Empresa Ofereceu Treinamento × Tempo Individual
utils.teste_qui_quadrado(
    df, 'q20_empresa_ofereceu_treinamento', 'q24_tempo_individual',
    "Empresa Ofereceu Treinamento × Tempo Individual"
)

# Teste 3: Realizou POCs × Tempo da Equipe
utils.teste_qui_quadrado(
    df, 'q23_realizou_pocs', 'q25_tempo_equipe',
    "Realizou POCs × Tempo da Equipe"
)

# Teste 4: Fonte Mais Útil × Tempo Individual
utils.teste_qui_quadrado(
    df, 'q22_fonte_mais_util', 'q24_tempo_individual',
    "Fonte Mais Útil × Tempo Individual"
)

# Teste 5: Cargo × Tempo com Containers
utils.teste_qui_quadrado(
    df, 'q1_cargo', 'q3_tempo_containers',
    "Cargo × Tempo com Containers"
)

# Teste 6: Setor × Tecnologia de Orquestração
utils.teste_qui_quadrado(
    df, 'q7_setor', 'q9_tecnologia_orquestracao',
    "Setor × Tecnologia de Orquestração"
)

print("\n" + "="*80)
print("RESUMO DOS TESTES QUI-QUADRADO")
print("="*80)
print("\nNível de significância: α = 0.05")
print("\nTodos os testes realizados estão documentados acima.")
print("Verifique os p-valores para identificar associações significativas.")

utils.titulo_secao("Testes Qui-Quadrado Concluídos")

# ============================================================================
# 3. ANACOR (ANÁLISE DE CORRESPONDÊNCIA)
# ============================================================================

utils.titulo_secao("3. ANACOR - Análise de Correspondência")

print("\nSerão gerados 3 mapas perceptuais bidimensionais:")
print("  1. Porte da Empresa × Principal Desafio")
print("  2. Fonte Mais Útil × Tempo Individual")
print("  3. Fator Mais Importante × Tempo da Equipe")
print()

# ANACOR 1
anacor1 = utils.anacor(
    df, 'q6_porte_empresa', 'q18_principal_desafio',
    "Porte da empresa × Principal desafio",
    salvar_grafico=True
)

# ANACOR 2
anacor2 = utils.anacor(
    df, 'q22_fonte_mais_util', 'q24_tempo_individual',
    "Fonte de conhecimento mais útil × Tempo até o profissional se sentir confortável",
    salvar_grafico=True
)

# ANACOR 3
anacor3 = utils.anacor(
    df, 'q26_fator_mais_importante', 'q25_tempo_equipe',
    "Fator mais importante no aprendizado da equipe × Tempo até o primeiro deploy em produção",
    salvar_grafico=True
)

print("\n" + "="*80)
print("INTERPRETAÇÃO DOS MAPAS PERCEPTUAIS")
print("="*80)
print("""
Os mapas perceptuais da ANACOR mostram:

• Categorias PRÓXIMAS no mapa: Tendem a estar associadas
• Categorias DISTANTES: Apresentam pouca ou nenhuma associação
• DISTÂNCIA ao centro: Indica contribuição para a variância total

Como interpretar:
1. Observe quais categorias aparecem próximas no mapa
2. Identifique padrões de agrupamento
3. Relacione os achados com a teoria sobre curva de aprendizado

IMPORTANTE: Estas são interpretações exploratórias. Sempre mencione que
são padrões visuais, não confirmações estatísticas causais.
""")

utils.titulo_secao("ANACOR Concluída")

# ============================================================================
# 4. REGRESSÃO LOGÍSTICA
# ============================================================================

utils.titulo_secao("4. Regressão Logística")

print("\nAnálise de fatores preditivos de curva de aprendizado rápida (<3 meses)\n")

resultado_regressao = utils.regressao_logistica(df)

if resultado_regressao['modelo'] is not None:
    print("\nModelo de Regressão Logística ajustado com sucesso!")
    print("   Consulte os Odds Ratios e p-valores acima para interpretação.")
else:
    print("\nModelo não pôde ser ajustado (dados insuficientes ou separação perfeita)")
    print("   Consulte mensagens de erro acima para detalhes.")

utils.titulo_secao("Regressão Logística Concluída")

# ============================================================================
# 5. RESUMO EXECUTIVO
# ============================================================================

utils.titulo_secao("5. Resumo Executivo dos Resultados")

print("\nAMOSTRA")
print("-" * 80)
print(f"Total de respondentes: {len(df)}")
print(f"Período de coleta: {df['timestamp'].iloc[0]} até {df['timestamp'].iloc[-1]}")

print("\nPERFIL PREDOMINANTE")
print("-" * 80)
if len(df['q1_cargo'].mode()) > 0:
    print(f"Cargo: {df['q1_cargo'].mode()[0]}")
if len(df['q6_porte_empresa'].mode()) > 0:
    print(f"Porte de empresa: {df['q6_porte_empresa'].mode()[0]}")
if len(df['q7_setor'].mode()) > 0:
    print(f"Setor: {df['q7_setor'].mode()[0]}")

print("\nDESAFIOS")
print("-" * 80)
if len(df['q18_principal_desafio'].mode()) > 0:
    print(f"Principal desafio: {df['q18_principal_desafio'].mode()[0]}")

# Médias dos desafios Likert
desafios_medias = {
    'Segurança': df['q12_desafio_seguranca'].mean(),
    'Complexidade Técnica': df['q13_desafio_complexidade'].mean(),
    'Monitoramento': df['q14_desafio_monitoramento'].mean(),
    'Mudanças Organizacionais': df['q15_desafio_mudancas'].mean(),
    'Treinamento': df['q16_desafio_treinamento'].mean(),
    'CI/CD': df['q17_desafio_cicd'].mean()
}
desafio_maior_media = max(desafios_medias, key=desafios_medias.get)
print(f"Maior média (Likert 1-5): {desafio_maior_media} ({desafios_medias[desafio_maior_media]:.2f})")

print("\nCURVA DE APRENDIZADO")
print("-" * 80)
curva_rapida_individual = df['q24_tempo_individual'].isin(['Menos de 3 meses', '3-6 meses']).sum()
print(f"Profissionais com curva rápida individual (<6 meses): {curva_rapida_individual}/{len(df)} ({curva_rapida_individual/len(df)*100:.1f}%)")

curva_rapida_equipe = df['q25_tempo_equipe'].isin(['Menos de 1 mês', '1-3 meses']).sum()
print(f"Equipes com curva rápida (<3 meses produção): {curva_rapida_equipe}/{len(df)} ({curva_rapida_equipe/len(df)*100:.1f}%)")

print("\nPERCEPÇÃO DE ROI E RECOMENDAÇÃO")
print("-" * 80)
print(f"ROI médio: {df['q27_roi'].mean():.2f}/5.00 (escala Likert)")
recom_sim = (df['q28_recomendaria'] == 'Sim').sum()
print(f"Recomendaria adoção: {recom_sim}/{len(df)} ({recom_sim/len(df)*100:.1f}%)")

# ============================================================================
# FINALIZAÇÃO
# ============================================================================

print("\n" + "="*80)
print("ANÁLISE COMPLETA FINALIZADA!")
print("="*80)
print("\nARQUIVOS GERADOS:")
print("   - Gráficos ANACOR: ../resultados/graficos/anacor/")
print("   - Dados processados: DataFrame 'df' em memória")
print("="*80 + "\n")
