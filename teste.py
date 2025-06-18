import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
import json
import random
from datetime import datetime, timedelta

# Configurações
np.random.seed(42)
random.seed(42)
num_transactions = 10000  # 10,000 transações
min_itens = 1
max_itens = 20

# Catálogo completo de produtos (100 produtos)
produtos = {
    # Bebidas Alcoólicas
    'Cervejas': [
        'Cerveja Pilsen', 'Cerveja IPA', 'Cerveja Weiss',  
        'Cerveja Lager', 'Cerveja Artesanal', 'Cerveja Zero Álcool'
    ],
    'Destilados': [
        'Vodka', 'Whisky', 'Gin', 'Rum', 'Tequila', 'Cachaça'
    ],
    'Vinhos': [
        'Vinho Tinto Seco', 'Vinho Branco', 'Vinho Rosé', 'Espumante', 'Vinho do Porto'
    ],
    
    # Acompanhamentos para Bebidas
    'Acompanhamentos': [
        'Amendoim', 'Batata Chips', 'Biscoito Salgado', 'Castanha de Caju',
        'Tábua de Queijos', 'Azeitona', 'Salgadinhos', 'Pasta de Amendoim',
        'Torradas', 'Biscoito de Polvilho'
    ],
    
    # Produtos para Bebês
    'Bebês': [
        'Fralda P', 'Fralda M', 'Fralda G', 'Fralda XG', 'Lenço Umedecido',
        'Pomada Assadura', 'Mamadeira', 'Chupeta', 'Leite em Pó', 'Papinha'
    ],
    
    # Limpeza
    'Limpeza Pesada': [
        'Desinfetante', 'Água Sanitária', 'Detergente Industrial', 'Limpa-Fornos',
        'Inseticida', 'Desentupidor'
    ],
    'Limpeza Geral': [
        'Sabão em Pó', 'Amaciante', 'Álcool 70%', 'Limpa Vidros',
        'Multiuso', 'Saco de Lixo'
    ],
    'Limpeza Pessoal': [
        'Sabonete Líquido', 'Shampoo', 'Condicionador', 'Pasta de Dente',
        'Fio Dental', 'Creme de Barbear'
    ],
    
    # Mercearia Básica
    'Grãos': [
        'Arroz Branco', 'Arroz Integral', 'Feijão Carioca', 'Feijão Preto',
        'Lentilha', 'Grão-de-Bico'
    ],
    'Massas': [
        'Macarrão Espaguete', 'Macarrão Parafuso', 'Macarrão Penne', 'Macarrão Instantâneo',
        'Farinha de Trigo', 'Farinha de Mandioca'
    ],
    
    # ... (outras categorias como laticínios, carnes, hortifruti)
}

# Padrões de compra realistas
padroes = [
    # Padrões com cerveja
    (['Cerveja Pilsen', 'Amendoim', 'Batata Chips'], 0.12),
    (['Cerveja IPA', 'Tábua de Queijos', 'Castanha de Caju'], 0.08),
    (['Cerveja Weiss', 'Azeitona', 'Biscoito de Polvilho'], 0.06),
    
    # Padrões com fralda
    (['Fralda P', 'Lenço Umedecido', 'Pomada Assadura'], 0.10),
    (['Fralda M', 'Leite em Pó', 'Mamadeira'], 0.09),
    (['Fralda G', 'Papinha', 'Sabonete Líquido'], 0.07),
    
    # Padrões de limpeza
    (['Sabão em Pó', 'Amaciante', 'Água Sanitária'], 0.11),
    (['Desinfetante', 'Saco de Lixo', 'Multiuso'], 0.09),
    (['Álcool 70%', 'Limpa Vidros', 'Pasta de Dente'], 0.08),
    
    # Padrões tradicionais
    (['Arroz Branco', 'Feijão Carioca', 'Óleo de Soja'], 0.15),
    (['Pão Francês', 'Manteiga', 'Leite Integral'], 0.13),
    (['Café', 'Açúcar', 'Biscoito Maizena'], 0.12),
    
    # Padrões especiais
    (['Vinho Tinto Seco', 'Queijo Brie', 'Presunto Parma'], 0.05),
    (['Whisky', 'Gelo', 'Água Mineral'], 0.04)
]

# Gerar transações com datas (últimos 6 meses)
def generate_transactions():
    transactions = []
    dates = []
    base_date = datetime.now() - timedelta(days=180)
    
    for _ in range(num_transactions):
        # Escolhe um padrão ou gera aleatório
        if random.random() < 0.7:  # 70% seguem padrões
            chosen_pattern = None
            rand = random.random()
            cumulative_prob = 0
            
            for items, prob in padroes:
                cumulative_prob += prob
                if rand < cumulative_prob:
                    chosen_pattern = items
                    break
            
            if chosen_pattern:
                # Adiciona itens do padrão + aleatórios
                transaction = list(chosen_pattern)
                extra_items = random.randint(0, 5)
                
                # Escolhe categoria relacionada ao padrão
                if 'Cerveja' in ' '.join(chosen_pattern):
                    category = random.choice(['Acompanhamentos', 'Destilados'])
                elif 'Fralda' in ' '.join(chosen_pattern):
                    category = random.choice(['Bebês', 'Limpeza Pessoal'])
                elif any(item in ' '.join(chosen_pattern) for item in ['Sabão', 'Álcool', 'Desinfetante']):
                    category = random.choice(['Limpeza Geral', 'Limpeza Pesada'])
                else:
                    category = random.choice(list(produtos.keys()))
                
                # Adiciona itens extras
                transaction.extend(random.sample(produtos[category], min(extra_items, len(produtos[category]))))
                
                # Limita tamanho
                transaction = list(set(transaction))[:max_itens]
                transactions.append(transaction)
                
                # Gera data (mais transações nos fins de semana)
                if random.random() < 0.4:  # 40% chance de ser fim de semana
                    date = base_date + timedelta(days=random.randint(0, 180), 
                                               hours=random.choice([18, 19, 20, 21]))  # Horário de happy hour
                else:
                    date = base_date + timedelta(days=random.randint(0, 180), 
                                               hours=random.randint(10, 20))
                dates.append(date)
                continue
        
        # Transação totalmente aleatória
        num_items = random.randint(min_itens, max_itens)
        transaction = random.sample([item for sublist in produtos.values() for item in sublist], 
                                  min(num_items, len([item for sublist in produtos.values() for item in sublist])))
        transactions.append(transaction)
        
        # Data aleatória
        date = base_date + timedelta(days=random.randint(0, 180), 
                                   hours=random.randint(8, 22))
        dates.append(date)
    
    return transactions, dates

# Gerar transações
transacoes, datas = generate_transactions()

# Salvar catálogo completo
with open('catalogo_produtos.json', 'w', encoding='utf-8') as f:
    json.dump(produtos, f, ensure_ascii=False, indent=2)

# Transformar para DataFrame one-hot
te = TransactionEncoder()
te_ary = te.fit(transacoes).transform(transacoes)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

# Adicionar data
df_onehot['Data'] = datas

# Salvar em CSV
df_onehot.to_csv('supermercado_onehot.csv', index=False)

# Salvar transações brutas com data
df_transacoes = pd.DataFrame({
    'TransacaoID': [f'T{i+1}' for i in range(len(transacoes))],
    'Data': datas,
    'Itens': [', '.join(t) for t in transacoes],
    'TotalItens': [len(t) for t in transacoes]
})

# Adicionar colunas de categoria (para análise)
def detect_categories(items):
    categories = set()
    for item in items:
        for cat, prods in produtos.items():
            if item in prods:
                categories.add(cat)
    return ', '.join(categories)

df_transacoes['Categorias'] = df_transacoes['Itens'].apply(lambda x: detect_categories(x.split(', ')))

df_transacoes.to_csv('supermercado_transacoes.csv', index=False)

# Salvar metadados
metadata = {
    'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'num_transacoes': num_transactions,
    'num_produtos': len(te.columns_),
    'categorias': list(produtos.keys()),
    'padroes': [{'itens': p[0], 'probabilidade': p[1]} for p in padroes]
}

with open('metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"""
✅ Base de dados gerada com sucesso!
📊 Total de transações: {num_transactions}
🛍️ Produtos únicos: {len(te.columns_)}
📁 Arquivos gerados:
   - supermercado_onehot.csv (formato one-hot para análise)
   - supermercado_transacoes.csv (transações completas com metadados)
   - catalogo_produtos.json (categorias e produtos)
   - metadata.json (informações sobre a geração)
""")



