# ==========================================
# ETAPA 2 - ANÁLISE PREDITIVA E PRESCRITIVA
# AUTOR: GUILHERME (PESSOA 3)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from IPython.display import display # <- ADICIONADO: Corrige o aviso amarelo do 'display'
import warnings
warnings.filterwarnings('ignore')

# RESOLVIDO: Removidos os '#' para o código carregar o banco de dados sozinho e tirar o aviso do 'df'
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
df = pd.read_csv(url)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df_cluster = df.dropna(subset=['TotalCharges']).copy() 

# Selecionando as variáveis contínuas cruciais para a análise de valor e tempo
features = ['tenure', 'MonthlyCharges']
X = df_cluster[features]

# O K-Means calcula distâncias euclidianas. Se não padronizarmos, a variável 
# 'MonthlyCharges' (que vai até 120) terá um peso diferente de 'tenure' (que vai até 72).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definindo K=3 com base na lógica de negócios (Iniciantes, Intermediários, Premium)
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['Cluster_Risco'] = kmeans.fit_predict(X_scaled)

nome_clusters = {
    0: 'Grupo 1: Risco de Evasão (Novos / Ticket Baixo-Médio)',
    1: 'Grupo 2: Premium Exigentes (Alta Fatura)',
    2: 'Grupo 3: Fiéis Consolidados (Longo Tempo / Retenção Alta)'
}
df_cluster['Perfil'] = df_cluster['Cluster_Risco'].map(nome_clusters)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cluster, x='tenure', y='MonthlyCharges', hue='Perfil', palette='Set1', alpha=0.7)
plt.title('Clusterização de Clientes K-Means (Perfil de Risco e Valor)')
plt.xlabel('Tempo de Contrato (Meses)')
plt.ylabel('Valor Mensal ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

def sistema_recomendacao_retencao(cliente):
    """
    Função que avalia as características multidimensionais do cliente 
    e prescreve uma ação de retenção focada em ROI (Retorno sobre Investimento).
    """
    recomendacoes = []
    
    # Regra 1: Blindagem de Segurança (Focada na descoberta da Etapa 1)
    if cliente['OnlineSecurity'] == 'No' and cliente['Contract'] == 'Month-to-month':
        recomendacoes.append("[AÇÃO CIRÚRGICA]: Oferecer 3 meses GRÁTIS de Segurança Online em troca de fidelidade de 12 meses.")
        
    # Regra 2: Mitigação de Risco Digital (Fatura Paperless)
    if cliente['PaperlessBilling'] == 'Yes':
        recomendacoes.append("[PREVENÇÃO]: Disparar SMS/WhatsApp de alerta de vencimento 2 dias antes para evitar churn por inadimplência acidental.")
        
    # Regra 3: Tratamento Personalizado por Cluster
    if cliente['Cluster_Risco'] == 0: # Risco de evasão precoce
        recomendacoes.append("[ONBOARDING]: Acionar equipe de Sucesso do Cliente (CS) para contato humano nas primeiras 4 semanas.")
    elif cliente['Cluster_Risco'] == 1: # Grupo Premium
        recomendacoes.append("[VIP]: Direcionar para fila de suporte técnico de Nível 2 (sem tempo de espera) para garantir excelência.")
        
    # Regra de Manutenção (Baseline)
    if not recomendacoes:
        recomendacoes.append("[MANUTENÇÃO]: Cliente em zona de estabilidade. Incluir no fluxo padrão de NPS trimestral.")
        
    return " | ".join(recomendacoes)

df_cluster['Acao_Recomendada'] = df_cluster.apply(sistema_recomendacao_retencao, axis=1)

print("\n--- DEMONSTRAÇÃO DO SISTEMA PRESCRITIVO DE RECOMENDAÇÃO ---")
display(df_cluster[['customerID', 'tenure', 'OnlineSecurity', 'PaperlessBilling', 'Perfil', 'Acao_Recomendada']].head(5))