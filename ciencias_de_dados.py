import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# ==========================================
# TRAVA DE DIRETÓRIO (ANTI-BUG DE PASTA Oculta)
# Garante que os arquivos sejam salvos exatamente onde este script .py está
# ==========================================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Iniciando o motor preditivo...")

# ==========================================
# 1. GERAÇÃO DOS DADOS DA PROVA DE CONCEITO (MOCK DATA)
# ==========================================
np.random.seed(42)
n_samples = 1000

data = {
    'ID_Cliente': range(1001, 1001 + n_samples),
    'Tempo_Contrato_Meses': np.random.randint(1, 72, n_samples),
    'Valor_Mensal': np.random.uniform(30, 150, n_samples),
    'Total_Gasto': np.random.uniform(100, 8000, n_samples),
    'Chamados_Suporte': np.random.randint(0, 10, n_samples),
    'Atraso_Pagamento_Dias': np.random.randint(0, 30, n_samples)
}
df = pd.DataFrame(data)

churn_prob = (
    0.35 * (df['Chamados_Suporte'] / 10) +
    0.30 * (df['Atraso_Pagamento_Dias'] / 30) -
    0.25 * (df['Tempo_Contrato_Meses'] / 72) +
    np.random.normal(0, 0.1, n_samples)
)
df['Churn'] = (churn_prob > 0.3).astype(int)

# ==========================================
# 2. PRÉ-PROCESSAMENTO E DIVISÃO (TREINO/TESTE)
# ==========================================
X = df.drop(columns=['ID_Cliente', 'Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ==========================================
# 3. TREINAMENTO DO MODELO
# ==========================================
print("Treinando o modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
model.fit(X_train, y_train)

# ==========================================
# 4. CÁLCULO DO SCORE DE RISCO E EXPORTAÇÃO
# ==========================================
y_probs = model.predict_proba(X_test)[:, 1]

base_final = X_test.copy()
base_final.insert(0, 'ID_Cliente', df.loc[X_test.index, 'ID_Cliente'])
base_final['Status_Real_Cancelamento'] = y_test
base_final['Score_Risco_Churn'] = np.round(y_probs, 4) 

nome_arquivo_saida = "base_com_score_risco_poc.csv"
base_final.to_csv(nome_arquivo_saida, index=False)
print(f"✅ SUCESSO! Planilha '{nome_arquivo_saida}' gerada na mesma pasta.")

# ==========================================
# 5. GERAÇÃO DE GRÁFICOS PARA OS SLIDES
# ==========================================
# Gráfico 1: Curva ROC
auc_score = roc_auc_score(y_test, y_probs)
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Capacidade do Modelo de Prever o Cancelamento')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('grafico_curva_roc.png')
print("✅ Gráfico 'grafico_curva_roc.png' salvo com sucesso.")
plt.close()

# Gráfico 2: Importância das Variáveis
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 5))
plt.barh(range(len(indices)), importances[indices], color='#2ecc71')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title('Fatores que mais impactam no Risco de Churn')
plt.tight_layout()
plt.savefig('grafico_importancia.png')
print("✅ Gráfico 'grafico_importancia.png' salvo com sucesso.")
plt.close()

print("\n=== RESUMO DAS MÉTRICAS ===")
print(f"Área sob a Curva (AUC): {auc_score:.2f}")
print("Processamento 100% concluído.")