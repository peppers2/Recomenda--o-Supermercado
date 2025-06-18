import pandas as pd

# Carregar dados one-hot
df = pd.read_csv('supermercado_onehot.csv', parse_dates=['Data'])

# Carregar transações brutas
transacoes = pd.read_csv('supermercado_transacoes.csv', parse_dates=['Data'])

# Carregar catálogo
import json
with open('catalogo_produtos.json', encoding='utf-8-sig') as f:
    catalogo = json.load(f)