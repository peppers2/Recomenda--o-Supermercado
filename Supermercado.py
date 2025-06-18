import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# Configuração da página
st.set_page_config(
    page_title="Smart Basket - Recomendador de Produtos",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .header {
        font-size: 50px !important;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 24px !important;
        color: #43a047;
        border-bottom: 2px solid #c8e6c9;
        padding-bottom: 10px;
    }
    .recommendation-card {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .basket-item {
        padding: 8px 15px;
        background-color: #c8e6c9;
        border-radius: 20px;
        margin: 5px;
        display: inline-block;
        font-weight: 500;
    }
    .stButton>button {
        background-color: #4caf50 !important;
        color: white !important;
        border-radius: 20px !important;
        padding: 8px 20px !important;
        font-weight: bold !important;
        transition: all 0.3s !important;
    }
    .stButton>button:hover {
        background-color: #388e3c !important;
        transform: scale(1.05);
    }
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }
    .product-card {
        border: 1px solid #c8e6c9;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
    }
    .product-card:hover {
        background-color: #e8f5e9;
        transform: translateY(-3px);
    }
    .product-card img {
        max-width: 80px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Gerar dados de produtos e regras de associação
@st.cache_resource
def generate_product_data():
    # Catálogo de produtos com categorias e emojis
    products = {
        'Laticínios': ['Leite 🥛', 'Queijo 🧀', 'Manteiga 🧈', 'Iogurte 🍶', 'Requeijão'],
        'Padaria': ['Pão 🍞', 'Biscoito 🍪', 'Rosca', 'Baguete', 'Croissant 🥐'],
        'Bebidas': ['Cerveja 🍺', 'Refrigerante 🥤', 'Água 💧', 'Suco 🧃', 'Vinho 🍷'],
        'Mercearia': ['Arroz 🍚', 'Feijão', 'Macarrão 🍝', 'Óleo', 'Sal', 'Açúcar'],
        'Carnes': ['Frango 🍗', 'Carne 🥩', 'Peixe 🐟', 'Linguiça', 'Bacon 🥓'],
        'Hortifruti': ['Tomate 🍅', 'Alface 🥬', 'Cebola', 'Batata 🥔', 'Maçã 🍎'],
        'Limpeza': ['Sabão 🧼', 'Detergente', 'Desinfetante', 'Esponja', 'Álcool'],
        'Higiene': ['Shampoo', 'Sabonete🧴', 'Pasta Dental', 'Papel Higiênico🧻', 'Fralda']
    }
    
    return products

@st.cache_resource
def generate_association_rules():
    # Gerar transações simuladas com padrões
    transactions = []
    products_flat = [item for sublist in generate_product_data().values() for item in sublist]
    
    for _ in range(500):
        # Padrões de compra
        if np.random.random() < 0.3:
            transactions.append(['Pão 🍞', 'Manteiga 🧈', 'Leite 🥛'])
        elif np.random.random() < 0.2:
            transactions.append(['Arroz 🍚', 'Feijão', 'Frango 🍗'])
        elif np.random.random() < 0.15:
            transactions.append(['Cerveja 🍺', 'Amendoim', 'Batata Chips'])
        elif np.random.random() < 0.1:
            transactions.append(['Fralda', 'Leite 🥛', 'Papel Higiênico🧻'])
        else:
            size = np.random.randint(1, 5)
            transactions.append(list(np.random.choice(products_flat, size=size, replace=False)))
    
    # Pré-processamento
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Gerar regras de associação
    frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    # Formatar regras
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)) if len(x) > 0 else '')
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)) if len(x) > 0 else '')
    
    return rules

# Inicializar a cesta de compras na sessão
if 'basket' not in st.session_state:
    st.session_state.basket = []

# Função para adicionar produto à cesta
def add_to_basket(product):
    if product not in st.session_state.basket:
        st.session_state.basket.append(product)

# Função para remover produto da cesta
def remove_from_basket(product):
    if product in st.session_state.basket:
        st.session_state.basket.remove(product)

# Função para gerar recomendações
def get_recommendations(basket, rules, top_n=3):
    if not basket or rules.empty:
        return []
    
    # Converter cesta para string
    basket_str = ', '.join(basket)
    
    # Encontrar regras onde os antecedentes estão na cesta
    relevant_rules = rules[rules['antecedents'].apply(
        lambda x: all(item in basket_str for item in x.split(', ')) if x else False)]
    
    # Ordenar por confiança e lift
    relevant_rules = relevant_rules.sort_values(
        by=['confidence', 'lift'], ascending=False)
    
    # Coletar recomendações
    recommendations = []
    for _, row in relevant_rules.iterrows():
        conseqs = row['consequents'].split(', ')
        for item in conseqs:
            if item not in basket and item not in recommendations and item != '':
                recommendations.append(item)
                if len(recommendations) >= top_n:
                    return recommendations
    
    return recommendations

# Interface do aplicativo
def main():
    # Carregar dados
    products = generate_product_data()
    rules = generate_association_rules()
    
    # Header
    st.markdown('<h1 class="header">🛒 Smart Basket</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px; font-size: 18px; color: #555;'>
        Adicione produtos à sua cesta e receba recomendações inteligentes de produtos que combinam!
        </div>
    """, unsafe_allow_html=True)
    
    # Layout em colunas
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="subheader">Sua Cesta de Compras</div>', unsafe_allow_html=True)
        
        # Mostrar cesta atual
        if st.session_state.basket:
            for product in st.session_state.basket:
                col_b1, col_b2 = st.columns([5, 1])
                with col_b1:
                    st.markdown(f'<div class="basket-item">{product}</div>', unsafe_allow_html=True)
                with col_b2:
                    if st.button(f"❌", key=f"remove_{product}"):
                        remove_from_basket(product)
        else:
            st.info("Sua cesta está vazia. Adicione produtos usando os botões ao lado.")
            
        # Botão para limpar cesta
        if st.session_state.basket:
            if st.button("🧹 Limpar Cesta", use_container_width=True):
                st.session_state.basket = []
                st.experimental_rerun()
        
        # Mostrar recomendações
        if st.session_state.basket:
            st.markdown('<div class="subheader" style="margin-top: 30px;">Recomendações</div>', unsafe_allow_html=True)
            st.write("Produtos que combinam com sua seleção:")
            
            recommendations = get_recommendations(st.session_state.basket, rules)
            
            if recommendations:
                for product in recommendations:
                    if st.button(f"➕ {product}", key=f"rec_{product}", 
                                use_container_width=True, 
                                help=f"Adicionar {product} à cesta"):
                        add_to_basket(product)
            else:
                st.info("Adicione mais produtos para receber recomendações")
    
    with col2:
        st.markdown('<div class="subheader">Produtos Disponíveis</div>', unsafe_allow_html=True)
        
        # Mostrar produtos por categoria
        for category, items in products.items():
            st.markdown(f"**{category}**")
            cols = st.columns(5)
            
            for i, product in enumerate(items):
                with cols[i % 5]:
                    if st.button(product, key=product, 
                                help=f"Adicionar {product} à cesta",
                                use_container_width=True):
                        add_to_basket(product)
        
        # Visualização de associações
        if st.session_state.basket:
            st.markdown('<div class="subheader" style="margin-top: 30px;">Rede de Associações</div>', unsafe_allow_html=True)
            
            # Criar grafo
            G = nx.Graph()
            basket_str = ', '.join(st.session_state.basket)
            
            # Adicionar nós da cesta
            for product in st.session_state.basket:
                G.add_node(product, color='#4caf50', size=3000)
            
            # Adicionar recomendações
            recommendations = get_recommendations(st.session_state.basket, rules, top_n=10)
            for product in recommendations:
                G.add_node(product, color='#ff9800', size=2000)
            
            # Adicionar arestas
            for _, row in rules.iterrows():
                antecedents = row['antecedents'].split(', ')
                consequents = row['consequents'].split(', ')
                
                for ant in antecedents:
                    for cons in consequents:
                        if ant in st.session_state.basket and cons in recommendations:
                            G.add_edge(ant, cons, weight=row['lift']*2)
            
            # Layout do grafo
            pos = nx.spring_layout(G, seed=42)
            
            # Desenhar o grafo
            plt.figure(figsize=(10, 6))
            node_colors = [G.nodes[n]['color'] for n in G.nodes]
            node_sizes = [G.nodes[n]['size'] for n in G.nodes]
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=node_sizes, alpha=0.9)
            nx.draw_networkx_edges(G, pos, width=[d['weight'] for u, v, d in G.edges(data=True)],
                                  edge_color='#888', alpha=0.6)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            plt.title("Relações entre produtos na sua cesta e recomendações")
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.info("Adicione produtos à cesta para ver as relações entre eles")

# Rodar o aplicativo
if __name__ == "__main__":
    main()