import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from datetime import datetime
from collections import defaultdict
import time
import plotly.express as px

# Configuração da página
st.set_page_config(
    page_title="Supermercado do Pedro - Recomendação IA",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .header {
        font-size: 3rem !important;
        font-weight: 700;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #c8e6c9;
    }
    .subheader {
        font-size: 1.5rem !important;
        color: #43a047;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .product-card {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
        background-color: #f8f9fa;
    }
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        border-color: #c8e6c9;
    }
    .basket-item {
        display: inline-flex;
        align-items: center;
        background-color: #e8f5e9;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 0.5rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .recommendation-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4caf50 !important;
        color: white !important;
        border-radius: 20px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s !important;
        border: none !important;
    }
    .stButton>button:hover {
        background-color: #388e3c !important;
        transform: scale(1.05);
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
    }
    .tabs .stTab {
        border-bottom: 2px solid transparent;
    }
    .tabs .stTab:hover {
        color: #2e7d32;
    }
    .tabs .stTab[aria-selected="true"] {
        color: #2e7d32;
        border-bottom-color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Carregar dados
@st.cache_resource
def load_data():
    # Carregar dados one-hot
    df = pd.read_csv('supermercado_onehot.csv', parse_dates=['Data'])
    
    # Carregar catálogo
    with open('catalogo_produtos.json', 'r', encoding='utf-8') as f:
        catalogo = json.load(f)
    
    # Carregar transações brutas
    transacoes = pd.read_csv('supermercado_transacoes.csv', parse_dates=['Data'])
    
    return df, catalogo, transacoes

@st.cache_resource
def prepare_rules(df):
    # Remover coluna de data para análise
    df_items = df.drop(columns=['Data'])
    
    # Gerar regras de associação
    frequent_itemsets = apriori(df_items, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    # Filtrar regras relevantes
    rules = rules[(rules['lift'] > 1.5) & (rules['confidence'] > 0.3)]
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=False)
    
    # 🔧 Corrigir erro de serialização JSON: transformar frozensets em listas
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    
    return rules

# Função de recomendação
def get_recommendations(basket, rules, catalogo, top_n=10):
    if not basket:
        return []
    
    # Converter cesta para conjunto
    basket_set = set(basket)
    
    # Encontrar regras relevantes
    recommendations = []
    scores = []
    
    for _, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        
        if antecedents.issubset(basket_set):
            for item in consequents:
                if item not in basket_set and item not in recommendations:
                    # Score baseado em lift e confidence
                    score = rule['lift'] * rule['confidence']
                    recommendations.append(item)
                    scores.append(score)
    
    # Ordenar recomendações por score
    if recommendations:
        sorted_recs = [x for _, x in sorted(zip(scores, recommendations), reverse=True)]
        return sorted_recs[:top_n]
    
    # Se não encontrar recomendações, sugerir itens populares
    popular_items = [
        item for sublist in catalogo.values() 
        for item in sublist 
        if item not in basket
    ][:top_n]
    
    return popular_items

# Função para calcular métricas
def calculate_metrics(basket, recommendations, rules):
    metrics = {
        'total_items': len(basket),
        'total_recommendations': len(recommendations),
        'avg_confidence': 0,
        'avg_lift': 0,
        'coverage': 0
    }
    
    if not basket or not recommendations:
        return metrics
    
    # Calcular métricas das regras aplicadas
    relevant_rules = []
    basket_set = set(basket)
    
    for _, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        
        if antecedents.issubset(basket_set) and any(item in consequents for item in recommendations):
            relevant_rules.append(rule)
    
    if relevant_rules:
        metrics['avg_confidence'] = np.mean([r['confidence'] for r in relevant_rules])
        metrics['avg_lift'] = np.mean([r['lift'] for r in relevant_rules])
        metrics['coverage'] = len(relevant_rules) / len(rules)
    
    return metrics

# Interface principal
def main():
    if 'basket' not in st.session_state:
        st.session_state.basket = []
    st.markdown('<div class="header">🛒 Supermercado do Pedro IA</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; font-size: 1.1rem; color: #555;'>
    Adicione produtos à sua cesta e receba recomendações inteligentes baseadas em machine learning
    </div>
    """, unsafe_allow_html=True)
    
    # Carregar dados
    df, catalogo, transacoes = load_data()
    rules = prepare_rules(df)
    
    # Inicializar cesta na sessão
    if 'basket' not in st.session_state:
        st.session_state.basket = []
    
    # Layout em colunas
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown('<div class="subheader">Sua Cesta de Compras</div>', unsafe_allow_html=True)
        
        # Mostrar cesta atual
        if st.session_state.basket:
            for product in st.session_state.basket:
                cols = st.columns([5, 1])
                with cols[0]:
                    st.markdown(f'<div class="basket-item">{product}</div>', unsafe_allow_html=True)
                with cols[1]:
                    if st.button(f"❌", key=f"remove_{product}"):
                        st.session_state.basket.remove(product)
                        #st.experimental_rerun()
            
            st.markdown("---")
            
            # Botão para limpar cesta
            if st.button("🧹 Limpar Cesta", use_container_width=True):
                st.session_state.basket = []
                #st.experimental_rerun()
            
            # Mostrar recomendações
            recommendations = get_recommendations(st.session_state.basket, rules, catalogo)
            metrics = calculate_metrics(st.session_state.basket, recommendations, rules)
            
            if recommendations:
                st.markdown('<div class="subheader">Recomendações para Você</div>', unsafe_allow_html=True)
                
                for i, product in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f'<div class="recommendation-card">'
                                  f'<h4>#{i} {product}</h4>'
                                  f'<p style="color: #555; margin-bottom: 0;">Recomendado com base em seus itens</p>'
                                  f'</div>', unsafe_allow_html=True)
                        
                        if st.button(f"➕ Adicionar {product}", key=f"add_{product}"):
                            st.session_state.basket.append(product)
                            #st.experimental_rerun()
            else:
                st.info("Adicione mais produtos à cesta para receber recomendações personalizadas")
        else:
            st.info("Sua cesta está vazia. Selecione produtos abaixo para começar.")
    
    with col2:
        # Abas para navegação
        








        
        with tab1:
            st.markdown('<div class="subheader">Catálogo de Produtos</div>', unsafe_allow_html=True)

            selected_category = st.selectbox(
                "Selecione uma categoria:",
                ["Todos"] + list(catalogo.keys()),
                index=0,
                key="category_selector"
            )

            # CSS: estilo do botão transparente e dos cards
            st.markdown("""
            <style>
                .product-btn {
                    background: none;
                    border: none;
                    width: 100%;
                    text-align: left;
                    padding: 0;
                }

                .product-card {
                    border-radius: 12px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    border: 1px solid #e0e0e0;
                    background-color: #f8f9fa;
                    transition: all 0.3s ease;
                }

                .product-card:hover {
                    transform: translateY(-3px);
                    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                    border-color: #c8e6c9;
                    cursor: pointer;
                }

                .product-card.added {
                    background-color: #e8f5e9;
                    border-color: #81c784;
                }

                .product-card h4 {
                    margin: 0;
                    color: #333;
                }

                .added-badge {
                    color: #2e7d32;
                    font-size: 0.8rem;
                    margin-top: 0.5rem;
                }
            </style>
            """, unsafe_allow_html=True)

            # Produtos por categoria
            products_to_show = (
                [item for sublist in catalogo.values() for item in sublist]
                if selected_category == "Todos"
                else catalogo[selected_category]
            )

            # Layout responsivo
            cols = st.columns(4)
            for i, product in enumerate(products_to_show):
                with cols[i % 4]:
                    is_added = product in st.session_state.basket

                    # Card inteiro como botão visível estilizado
                    if st.button(
                        label=f"""
                        <div class="product-card{' added' if is_added else ''}">
                            <h4>{product}</h4>
                            {'<div class="added-badge">✔️ Adicionado</div>' if is_added else ''}
                        </div>
                        """,
                        key=f"product_btn_{i}",
                        help="Clique para adicionar/remover",
                        unsafe_allow_html=True
                    ):
                        if is_added:
                            st.session_state.basket.remove(product)
                        else:
                            st.session_state.basket.append(product)
                        st.experimental_rerun()

                                    














        
        with tab2:
            st.markdown('<div class="subheader">Análise de Associações</div>', unsafe_allow_html=True)
            
            if st.session_state.basket:
                # Criar grafo de associações
                G = nx.Graph()
                basket_set = set(st.session_state.basket)
                recommendations = get_recommendations(st.session_state.basket, rules, catalogo, top_n=10)
                
                # Adicionar nós
                for product in st.session_state.basket:
                    G.add_node(product, size=20, color='#2e7d32')
                
                for product in recommendations:
                    G.add_node(product, size=15, color='#43a047')
                
                # Adicionar arestas
                for _, rule in rules.iterrows():
                    antecedents = set(rule['antecedents'])
                    consequents = set(rule['consequents'])
                    
                    if antecedents.issubset(basket_set):
                        for ant in antecedents:
                            for cons in consequents:
                                if cons in recommendations:
                                    G.add_edge(ant, cons, weight=rule['lift'], 
                                              label=f"Lift: {rule['lift']:.2f}")
                
                # Plotar grafo
                if G.nodes():
                    pos = nx.spring_layout(G, k=0.8, seed=42)
                    
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1.5, color='#888'),
                        hoverinfo='none',
                        mode='lines')
                    
                    node_x = []
                    node_y = []
                    node_text = []
                    node_color = []
                    node_size = []
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(node)
                        node_color.append(G.nodes[node]['color'])
                        node_size.append(G.nodes[node]['size'])
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition="top center",
                        hoverinfo='text',
                        marker=dict(
                            color=node_color,
                            size=node_size,
                            line_width=2))
                    
                    fig = go.Figure(data=[edge_trace, node_trace],
                                   layout=go.Layout(
                                       showlegend=False,
                                       hovermode='closest',
                                       margin=dict(b=0,l=0,r=0,t=0),
                                       height=500,
                                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Adicione mais produtos para visualizar as associações")
                
                # Novo: Gráfico de Sankey para fluxo de associações
                st.markdown('<div class="subheader">Fluxo de Associações</div>', unsafe_allow_html=True)
                if G.edges():
                    # Preparar dados para Sankey
                    sources = []
                    targets = []
                    values = []
                    labels = list(basket_set) + recommendations
                    
                    for edge in G.edges(data=True):
                        sources.append(labels.index(edge[0]))
                        targets.append(labels.index(edge[1]))
                        values.append(edge[2]['weight'])
                    
                    fig = go.Figure(go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=labels,
                            color="#4caf50"
                        ),
                        link=dict(
                            source=sources,
                            target=targets,
                            value=values
                        )))
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Novo: Heatmap de correlação entre itens
                st.markdown('<div class="subheader">Correlação entre Itens</div>', unsafe_allow_html=True)
                if len(st.session_state.basket) > 1:
                    # Criar matriz de correlação
                    basket_items = list(basket_set) + recommendations
                    correlation_matrix = pd.DataFrame(index=basket_items, columns=basket_items)
                    
                    for item1 in basket_items:
                        for item2 in basket_items:
                            if item1 == item2:
                                correlation_matrix.loc[item1, item2] = 1
                            else:
                                # Encontrar regras que conectam os itens
                                matching_rules = rules[
                                    ((rules['antecedents'].apply(lambda x: item1 in x) & 
                                    rules['consequents'].apply(lambda x: item2 in x)) |
                                    (rules['antecedents'].apply(lambda x: item2 in x) & 
                                    rules['consequents'].apply(lambda x: item1 in x)))
                                ]

                                if not matching_rules.empty:
                                    correlation_matrix.loc[item1, item2] = matching_rules['lift'].max()
                                else:
                                    correlation_matrix.loc[item1, item2] = 0
                    
                    fig = px.imshow(
                        correlation_matrix.astype(float) ,  # converter para percentual
                        labels=dict(x="Item", y="Item", color="Associação (%)"),
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        color_continuous_scale='Greens',
                        text_auto='.2f'  # mostrar valores com 2 casas decimais
)
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Adicione mais itens à cesta para ver correlações")
            
            # Top associações
            st.markdown('<div class="subheader">Principais Associações no Modelo</div>', unsafe_allow_html=True)
            #top_rules = rules.head(10).copy()
            #top_rules['Associação'] = top_rules['antecedents'].astype(str) + " → " + top_rules['consequents'].astype(str)
            # Filtrar regras que envolvem produtos no carrinho
            carrinho_set = set(st.session_state.basket)
            top_rules = rules[
                rules['antecedents'].apply(lambda x: len(carrinho_set.intersection(x)) > 0)
            ].copy()


            # Ordenar por lift e confiança
            top_rules = top_rules.sort_values(by=['lift', 'confidence'], ascending=False).head(15)

            # Criar coluna de descrição da associação
            top_rules['Associação'] = top_rules['antecedents'].apply(lambda x: ', '.join(x)) + " → " + \
                                    top_rules['consequents'].apply(lambda x: ', '.join(x))


            fig = px.bar(top_rules, x='lift', y='Associação', 
                         color='confidence',
                         color_continuous_scale='Greens',
                         labels={'lift': 'Força da Associação (Lift)', 'confidence': 'Confiança'},
                         height=500)
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)





            
            # Novo: Treemap de categorias de produtos
            st.markdown('<div class="subheader">Distribuição por Categoria</div>', unsafe_allow_html=True)
            category_counts = {cat: len(prods) for cat, prods in catalogo.items()}
            df_categories = pd.DataFrame({
                'Categoria': list(category_counts.keys()),
                'Quantidade': list(category_counts.values()),
                'Parent': [''] * len(category_counts)
            })

            
            fig = px.treemap(df_categories, path=['Parent', 'Categoria'], values='Quantidade',
                            color='Quantidade', color_continuous_scale='Greens')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)






        
        with tab3:
            st.markdown('<div class="subheader">Desempenho do Modelo</div>', unsafe_allow_html=True)
            
            if st.session_state.basket:
                metrics = calculate_metrics(st.session_state.basket, 
                                           get_recommendations(st.session_state.basket, rules, catalogo), 
                                           rules)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">'
                            '<h3>📊 Confiança Média</h3>'
                            f'<h1>{ (metrics["avg_confidence"] * 100) :.2f}</h1>'   # <-- f-string está sozinha aqui
                            '<p>Probabilidade média de acerto</p>'
                            '</div>', unsafe_allow_html=True)

               
               
                with col2:
                    st.markdown('<div class="metric-card">'
                               '<h3>📈 Lift Médio</h3>'
                               f'<h1>{metrics["avg_lift"]:.2f}</h1>'
                               '<p>Força das associações</p>'
                               '</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">'
                               '<h3>🛒 Cobertura</h3>'
                               f'<h1>{metrics["coverage"]:.2%}</h1>'
                               '<p>Regras aplicáveis</p>'
                               '</div>', unsafe_allow_html=True)
                
               
                # Configuração do tema verde
                green_theme = {
                    'colorway': ['#2e7d32', '#388e3c', '#43a047', '#4caf50', '#66bb6a'],
                    'plot_bgcolor': '#f5f9f5',
                    'paper_bgcolor': '#ffffff'
                }

                if not rules.empty:
                    with st.container():
                        st.subheader("📊 Análise Completa das Associações entre Produtos")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Gráfico de Violino Verde
                            fig1 = px.violin(rules, y='lift', box=True, color_discrete_sequence=['#2e7d32'])
                            fig1.update_layout(
                                title='Distribuição da Força das Associações',
                                **green_theme
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                            
                        with col2:
                            # Heatmap Verde
                            fig2 = px.density_heatmap(
                                rules, x="confidence", y="lift",
                                color_continuous_scale='Greens',
                                title='Relação Confiança vs Força'
                            )
                            fig2.update_layout(**green_theme)
                            st.plotly_chart(fig2, use_container_width=True)
                        

                else:
                    st.info("Adicione produtos à cesta para ver as métricas de recomendação")







if __name__ == "__main__":
    main()

        # Rodapé fixo no final de todas as páginas
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #e8f5e9;
        color: #2e7d32;
        text-align: center;
        padding: 1rem 0;
        font-size: 1rem;
        font-weight: 500;
        border-top: 1px solid #c8e6c9;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
        z-index: 9999;
    }
    </style>

    <div class="footer">
        Desenvolvido por <strong>Pedro Siqueira</strong> © 2025
    </div>
    """, unsafe_allow_html=True)
