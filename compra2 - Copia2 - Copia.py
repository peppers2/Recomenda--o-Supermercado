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
    /* Estilos base (light theme) */
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
    
    /* Ajustes de layout responsivo */
    @media (max-width: 1600px) {
        .header {
            font-size: 2.5rem !important;
        }
        .subheader {
            font-size: 1.3rem !important;
        }
    }
    
    @media (max-width: 1366px) {
        .header {
            font-size: 2.2rem !important;
        }
        .subheader {
            font-size: 1.1rem !important;
        }
        .product-card {
            padding: 0.8rem;
        }
        .recommendation-card {
            padding: 1rem;
        }
    }
    
    /* Dark Theme Overrides */
    @media (prefers-color-scheme: dark) {
        /* Fundo geral */
        .stApp, .main {
            background-color: #121212 !important;
            color: #e0e0e0 !important;
        }
        
        /* Componentes */
        .header {
            color: #81c784 !important;
            border-bottom-color: #2e7d32 !important;
        }
        
        .subheader {
            color: #66bb6a !important;
        }
        
        .product-card {
            background-color: #1e1e1e !important;
            border-color: #333 !important;
            color: #e0e0e0 !important;
        }
        
        .basket-item {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
        }
        
        .recommendation-card {
            background: linear-gradient(135deg, #2d2d2d 0%, #1e3b1e 100%) !important;
            color: #e0e0e0 !important;
        }
        
        .metric-card {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
        }
        
        .stButton>button {
            background-color: #2e7d32 !important;
        }
        
        .stButton>button:hover {
            background-color: #1b5e20 !important;
        }
        
        .tabs .stTab:hover {
            color: #81c784 !important;
        }
        
        .tabs .stTab[aria-selected="true"] {
            color: #81c784 !important;
            border-bottom-color: #81c784 !important;
        }
        
        /* Gráficos */
        .js-plotly-plot .plotly, 
        .plot-container .plotly {
            background-color: transparent !important;
        }
        
        /* Rodapé */
        .footer {
            background-color: #1e1e1e !important;
            color: #81c784 !important;
            border-top-color: #2e7d32 !important;
        }
        
        /* Ajustes específicos para elementos do Streamlit */
        .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, 
        .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, 
        .st-as, .st-bh, .st-bi, .st-bj, .st-bk, .st-bl, .st-bm, .st-bn, 
        .st-bo, .st-bp, .st-bq, .st-br, .st-bs, .st-bt, .st-bu, .st-bv, 
        .st-bw, .st-bx, .st-by, .st-bz, .st-c0, .st-c1, .st-c2, .st-c3, 
        .st-c4, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9, .st-ca, .st-cb, 
        .st-cc, .st-cd, .st-ce, .st-cf, .st-cg, .st-ch, .st-ci, .st-cj, 
        .st-ck, .st-cl, .st-cm, .st-cn, .st-co, .st-cp, .st-cq, .st-cr, 
        .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz, 
        .st-d0, .st-d1, .st-d2, .st-d3, .st-d4, .st-d5, .st-d6, .st-d7, 
        .st-d8, .st-d9, .st-da, .st-db, .st-dc, .st-dd, .st-de, .st-df, 
        .st-dg, .st-dh, .st-di, .st-dj, .st-dk, .st-dl, .st-dm, .st-dn, 
        .st-do, .st-dp, .st-dq, .st-dr, .st-ds, .st-dt, .st-du, .st-dv, 
        .st-dw, .st-dx, .st-dy, .st-dz, .st-e0, .st-e1, .st-e2, .st-e3, 
        .st-e4, .st-e5, .st-e6, .st-e7, .st-e8, .st-e9, .st-ea, .st-eb, 
        .st-ec, .st-ed, .st-ee, .st-ef, .st-eg, .st-eh, .st-ei, .st-ej, 
        .st-ek, .st-el, .st-em, .st-en, .st-eo, .st-ep, .st-eq, .st-er, 
        .st-es, .st-et, .st-eu, .st-ev, .st-ew, .st-ex, .st-ey, .st-ez, 
        .st-f0, .st-f1, .st-f2, .st-f3, .st-f4, .st-f5, .st-f6, .st-f7, 
        .st-f8, .st-f9, .st-fa, .st-fb, .st-fc, .st-fd, .st-fe, .st-ff, 
        .st-fg, .st-fh, .st-fi, .st-fj, .st-fk, .st-fl, .st-fm, .st-fn, 
        .st-fo, .st-fp, .st-fq, .st-fr, .st-fs, .st-ft, .st-fu, .st-fv, 
        .st-fw, .st-fx, .st-fy, .st-fz, .st-g0, .st-g1, .st-g2, .st-g3, 
        .st-g4, .st-g5, .st-g6, .st-g7, .st-g8, .st-g9, .st-ga, .st-gb, 
        .st-gc, .st-gd, .st-ge, .st-gf, .st-gg, .st-gh, .st-gi, .st-gj, 
        .st-gk, .st-gl, .st-gm, .st-gn, .st-go, .st-gp, .st-gq, .st-gr, 
        .st-gs, .st-gt, .st-gu, .st-gv, .st-gw, .st-gx, .st-gy, .st-gz, 
        .st-h0, .st-h1, .st-h2, .st-h3, .st-h4, .st-h5, .st-h6, .st-h7, 
        .st-h8, .st-h9, .st-ha, .st-hb, .st-hc, .st-hd, .st-he, .st-hf, 
        .st-hg, .st-hh, .st-hi, .st-hj, .st-hk, .st-hl, .st-hm, .st-hn, 
        .st-ho, .st-hp, .st-hq, .st-hr, .st-hs, .st-ht, .st-hu, .st-hv, 
        .st-hw, .st-hx, .st-hy, .st-hz, .st-i0, .st-i1, .st-i2, .st-i3, 
        .st-i4, .st-i5, .st-i6, .st-i7, .st-i8, .st-i9, .st-ia, .st-ib, 
        .st-ic, .st-id, .st-ie, .st-if, .st-ig, .st-ih, .st-ii, .st-ij, 
        .st-ik, .st-il, .st-im, .st-in, .st-io, .st-ip, .st-iq, .st-ir, 
        .st-is, .st-it, .st-iu, .st-iv, .st-iw, .st-ix, .st-iy, .st-iz, 
        .st-j0, .st-j1, .st-j2, .st-j3, .st-j4, .st-j5, .st-j6, .st-j7, 
        .st-j8, .st-j9, .st-ja, .st-jb, .st-jc, .st-jd, .st-je, .st-jf, 
        .st-jg, .st-jh, .st-ji, .st-jj, .st-jk, .st-jl, .st-jm, .st-jn, 
        .st-jo, .st-jp, .st-jq, .st-jr, .st-js, .st-jt, .st-ju, .st-jv, 
        .st-jw, .st-jx, .st-jy, .st-jz, .st-k0, .st-k1, .st-k2, .st-k3, 
        .st-k4, .st-k5, .st-k6, .st-k7, .st-k8, .st-k9, .st-ka, .st-kb, 
        .st-kc, .st-kd, .st-ke, .st-kf, .st-kg, .st-kh, .st-ki, .st-kj, 
        .st-kk, .st-kl, .st-km, .st-kn, .st-ko, .st-kp, .st-kq, .st-kr, 
        .st-ks, .st-kt, .st-ku, .st-kv, .st-kw, .st-kx, .st-ky, .st-kz, 
        .st-l0, .st-l1, .st-l2, .st-l3, .st-l4, .st-l5, .st-l6, .st-l7, 
        .st-l8, .st-l9, .st-la, .st-lb, .st-lc, .st-ld, .st-le, .st-lf, 
        .st-lg, .st-lh, .st-li, .st-lj, .st-lk, .st-ll, .st-lm, .st-ln, 
        .st-lo, .st-lp, .st-lq, .st-lr, .st-ls, .st-lt, .st-lu, .st-lv, 
        .st-lw, .st-lx, .st-ly, .st-lz, .st-m0, .st-m1, .st-m2, .st-m3, 
        .st-m4, .st-m5, .st-m6, .st-m7, .st-m8, .st-m9, .st-ma, .st-mb, 
        .st-mc, .st-md, .st-me, .st-mf, .st-mg, .st-mh, .st-mi, .st-mj, 
        .st-mk, .st-ml, .st-mm, .st-mn, .st-mo, .st-mp, .st-mq, .st-mr, 
        .st-ms, .st-mt, .st-mu, .st-mv, .st-mw, .st-mx, .st-my, .st-mz, 
        .st-n0, .st-n1, .st-n2, .st-n3, .st-n4, .st-n5, .st-n6, .st-n7, 
        .st-n8, .st-n9, .st-na, .st-nb, .st-nc, .st-nd, .st-ne, .st-nf, 
        .st-ng, .st-nh, .st-ni, .st-nj, .st-nk, .st-nl, .st-nm, .st-nn, 
        .st-no, .st-np, .st-nq, .st-nr, .st-ns, .st-nt, .st-nu, .st-nv, 
        .st-nw, .st-nx, .st-ny, .st-nz, .st-o0, .st-o1, .st-o2, .st-o3, 
        .st-o4, .st-o5, .st-o6, .st-o7, .st-o8, .st-o9, .st-oa, .st-ob, 
        .st-oc, .st-od, .st-oe, .st-of, .st-og, .st-oh, .st-oi, .st-oj, 
        .st-ok, .st-ol, .st-om, .st-on, .st-oo, .st-op, .st-oq, .st-or, 
        .st-os, .st-ot, .st-ou, .st-ov, .st-ow, .st-ox, .st-oy, .st-oz, 
        .st-p0, .st-p1, .st-p2, .st-p3, .st-p4, .st-p5, .st-p6, .st-p7, 
        .st-p8, .st-p9, .st-pa, .st-pb, .st-pc, .st-pd, .st-pe, .st-pf, 
        .st-pg, .st-ph, .st-pi, .st-pj, .st-pk, .st-pl, .st-pm, .st-pn, 
        .st-po, .st-pp, .st-pq, .st-pr, .st-ps, .st-pt, .st-pu, .st-pv, 
        .st-pw, .st-px, .st-py, .st-pz, .st-q0, .st-q1, .st-q2, .st-q3, 
        .st-q4, .st-q5, .st-q6, .st-q7, .st-q8, .st-q9, .st-qa, .st-qb, 
        .st-qc, .st-qd, .st-qe, .st-qf, .st-qg, .st-qh, .st-qi, .st-qj, 
        .st-qk, .st-ql, .st-qm, .st-qn, .st-qo, .st-qp, .st-qq, .st-qr, 
        .st-qs, .st-qt, .st-qu, .st-qv, .st-qw, .st-qx, .st-qy, .st-qz, 
        .st-r0, .st-r1, .st-r2, .st-r3, .st-r4, .st-r5, .st-r6, .st-r7, 
        .st-r8, .st-r9, .st-ra, .st-rb, .st-rc, .st-rd, .st-re, .st-rf, 
        .st-rg, .st-rh, .st-ri, .st-rj, .st-rk, .st-rl, .st-rm, .st-rn, 
        .st-ro, .st-rp, .st-rq, .st-rr, .st-rs, .st-rt, .st-ru, .st-rv, 
        .st-rw, .st-rx, .st-ry, .st-rz, .st-s0, .st-s1, .st-s2, .st-s3, 
        .st-s4, .st-s5, .st-s6, .st-s7, .st-s8, .st-s9, .st-sa, .st-sb, 
        .st-sc, .st-sd, .st-se, .st-sf, .st-sg, .st-sh, .st-si, .st-sj, 
        .st-sk, .st-sl, .st-sm, .st-sn, .st-so, .st-sp, .st-sq, .st-sr, 
        .st-ss, .st-st, .st-su, .st-sv, .st-sw, .st-sx, .st-sy, .st-sz, 
        .st-t0, .st-t1, .st-t2, .st-t3, .st-t4, .st-t5, .st-t6, .st-t7, 
        .st-t8, .st-t9, .st-ta, .st-tb, .st-tc, .st-td, .st-te, .st-tf, 
        .st-tg, .st-th, .st-ti, .st-tj, .st-tk, .st-tl, .st-tm, .st-tn, 
        .st-to, .st-tp, .st-tq, .st-tr, .st-ts, .st-tt, .st-tu, .st-tv, 
        .st-tw, .st-tx, .st-ty, .st-tz, .st-u0, .st-u1, .st-u2, .st-u3, 
        .st-u4, .st-u5, .st-u6, .st-u7, .st-u8, .st-u9, .st-ua, .st-ub, 
        .st-uc, .st-ud, .st-ue, .st-uf, .st-ug, .st-uh, .st-ui, .st-uj, 
        .st-uk, .st-ul, .st-um, .st-un, .st-uo, .st-up, .st-uq, .st-ur, 
        .st-us, .st-ut, .st-uu, .st-uv, .st-uw, .st-ux, .st-uy, .st-uz, 
        .st-v0, .st-v1, .st-v2, .st-v3, .st-v4, .st-v5, .st-v6, .st-v7, 
        .st-v8, .st-v9, .st-va, .st-vb, .st-vc, .st-vd, .st-ve, .st-vf, 
        .st-vg, .st-vh, .st-vi, .st-vj, .st-vk, .st-vl, .st-vm, .st-vn, 
        .st-vo, .st-vp, .st-vq, .st-vr, .st-vs, .st-vt, .st-vu, .st-vv, 
        .st-vw, .st-vx, .st-vy, .st-vz, .st-w0, .st-w1, .st-w2, .st-w3, 
        .st-w4, .st-w5, .st-w6, .st-w7, .st-w8, .st-w9, .st-wa, .st-wb, 
        .st-wc, .st-wd, .st-we, .st-wf, .st-wg, .st-wh, .st-wi, .st-wj, 
        .st-wk, .st-wl, .st-wm, .st-wn, .st-wo, .st-wp, .st-wq, .st-wr, 
        .st-ws, .st-wt, .st-wu, .st-wv, .st-ww, .st-wx, .st-wy, .st-wz, 
        .st-x0, .st-x1, .st-x2, .st-x3, .st-x4, .st-x5, .st-x6, .st-x7, 
        .st-x8, .st-x9, .st-xa, .st-xb, .st-xc, .st-xd, .st-xe, .st-xf, 
        .st-xg, .st-xh, .st-xi, .st-xj, .st-xk, .st-xl, .st-xm, .st-xn, 
        .st-xo, .st-xp, .st-xq, .st-xr, .st-xs, .st-xt, .st-xu, .st-xv, 
        .st-xw, .st-xx, .st-xy, .st-xz, .st-y0, .st-y1, .st-y2, .st-y3, 
        .st-y4, .st-y5, .st-y6, .st-y7, .st-y8, .st-y9, .st-ya, .st-yb, 
        .st-yc, .st-yd, .st-ye, .st-yf, .st-yg, .st-yh, .st-yi, .st-yj, 
        .st-yk, .st-yl, .st-ym, .st-yn, .st-yo, .st-yp, .st-yq, .st-yr, 
        .st-ys, .st-yt, .st-yu, .st-yv, .st-yw, .st-yx, .st-yy, .st-yz, 
        .st-z0, .st-z1, .st-z2, .st-z3, .st-z4, .st-z5, .st-z6, .st-z7, 
        .st-z8, .st-z9, .st-za, .st-zb, .st-zc, .st-zd, .st-ze, .st-zf, 
        .st-zg, .st-zh, .st-zi, .st-zj, .st-zk, .st-zl, .st-zm, .st-zn, 
        .st-zo, .st-zp, .st-zq, .st-zr, .st-zs, .st-zt, .st-zu, .st-zv, 
        .st-zw, .st-zx, .st-zy, .st-zz {
            background-color: transparent !important;
        }
    }
    
    /* Ajustes específicos para gráficos Plotly no dark mode */
    @media (prefers-color-scheme: dark) {
        .js-plotly-plot .plotly .modebar {
            background-color: #1e1e1e !important;
        }
        
        .js-plotly-plot .plotly .modebar-btn svg {
            fill: #e0e0e0 !important;
        }
        
        .js-plotly-plot .plotly .main-svg {
            background-color: transparent !important;
        }
        
        .js-plotly-plot .plotly .legend text {
            fill: #e0e0e0 !important;
        }
        
        .js-plotly-plot .plotly .cartesianlayer .xaxis text, 
        .js-plotly-plot .plotly .cartesianlayer .yaxis text {
            fill: #e0e0e0 !important;
        }
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
        tab1, tab2, tab3 = st.tabs(["📦 Todos os Produtos", "📊 Análises", "⚙️ Métricas do Modelo"])
        
        with tab1:
            st.markdown('<div class="subheader">Catálogo de Produtos</div>', unsafe_allow_html=True)
            
            # Seletor de categoria
            selected_category = st.selectbox(
                "Selecione uma categoria:",
                ["Todos"] + list(catalogo.keys()),
                index=0
            )
            
            # Mostrar produtos
            if selected_category == "Todos":
                products_to_show = [item for sublist in catalogo.values() for item in sublist]
            else:
                products_to_show = catalogo[selected_category]
            
            # Grid de produtos
            cols = st.columns(4)
            for i, product in enumerate(products_to_show):
                with cols[i % 4]:
                    with st.container():
                        st.markdown(f'<div class="product-card">'
                                  f'<h4>{product}</h4>'
                                  f'</div>', unsafe_allow_html=True)
                        
                        if st.button(f"➕ Adicionar", key=f"add_{product}_catalog"):
                            if product not in st.session_state.basket:
                                st.session_state.basket.append(product)
                                #st.experimental_rerun()
        
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
                               f'<h1>{metrics["avg_confidence"]:.2f}</h1>'
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
