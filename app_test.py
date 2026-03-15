import streamlit as st
from nlp_fonctions import load_models
import pandas as pd
import re
import base64
import os

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = 0  # 0 = Accueil
if "selected_cluster_filter" not in st.session_state:
    st.session_state["selected_cluster_filter"] = []


if "selected_recipe" not in st.session_state:
    st.session_state["selected_recipe"] = None


if "recipe_source" not in st.session_state:
    st.session_state["recipe_source"] = 1 # Par défaut, on revient sur l'onglet 1


# --- INITIALISATION MÉMOIRE RECHERCHE ---
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="ViteUneRecette",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force le mode clair visuellement pour les widgets
st.markdown('<style>div[data-testid="stAppViewContainer"]{color-scheme: light !important;}</style>', unsafe_allow_html=True)

# =========================
# CHARGEMENT DES DONNÉES
# =========================
# 1. INITIALISATION 
# On prépare la mémoire de l'app avant de charger quoi que ce soit
if "selected_cluster_filter" not in st.session_state:
    st.session_state["selected_cluster_filter"] = []


# Dans nlp_fonctions.py (recommandé) ou autour de votre import
@st.cache_resource
def get_cached_models():
    # Cette fonction ne s'exécutera qu'UNE SEULE FOIS pour tous les utilisateurs
    return load_models()

# Remplacez votre ligne actuelle par :
df, df_chatbot, VOCAB_INGREDIENTS, tfidf_both, vec_both = get_cached_models()


# =========================
# CSS
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --k-red: #c8102e;
        --k-red-dark: #a20d26;
        --k-text: #1d1d1d;
        --k-muted: #666666;
        --k-bg: #f3f3f3;
        --k-white: #ffffff;
        --k-border: #e1e1e1;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--k-text);
    }

    .stApp {
        background: var(--k-bg);
    }

    
    /* Retire les marges de sécurité de Streamlit */
    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
        margin-top: 0 !important;
    }

    /* Force le contenu à utiliser 100% de la largeur de l'écran */
    .block-container {
        max-width: 100% !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        margin: 0 !important;
    }

    /* Ajuste le Hero pour qu'il soit parfaitement bord à bord */
    .hero-wrap {
        position: relative;
        width: 100%;
        margin-left: 0;
        margin-right: 0;
        margin-top: 0;
        height: 500px;
    }

    /* Suppression du header et toolbar Streamlit */
    header[data-testid="stHeader"], div[data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stAppViewContainer"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* --- NAVIGATION STYLE PILL (Étiré & Centré) --- */
    
    /* 1. Conteneur principal : Fond blanc et centrage total */
    div[data-testid="stRadio"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #e1e1e1 !important; /* Ligne grise fine en dessous */
        padding: 15px 0 !important; /* Espace vertical */
        margin-top: -1.5rem !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }

    /* 2. Groupe de boutons : Aligné horizontalement et centré */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 15px !important; /* Espace entre les pilules */
        background-color: transparent !important;
    }

    /* 3. Cache les cercles radio par défaut */
    div[data-testid="stRadio"] label div:first-child {
        display: none !important;
    }

    /* 4. STYLE DES ONGLET "PILULE" (Le cœur du design) */
   
    div[data-testid="stRadio"] label {
        background-color: #f1f1f1 !important;
        border: 1px solid #e1e1e1 !important;
        border-radius: 50px !important;
        
        /* On retire le min-width fixe pour laisser les pilules s'étirer selon le texte */
        padding: 8px 25px !important; 
        margin: 0 !important;
        cursor: pointer !important;
        
        /* Force l'alignement horizontal */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 45px !important; /* FORCE UNE HAUTEUR IDENTIQUE POUR TOUTES */
        width: auto !important;
    }

    /* 5. STYLE DU TEXTE (Empêche le retour à la ligne) */
    div[data-testid="stRadio"] label p {
        color: #1d1d1d !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
        padding: 0 !important;
        
        /* LES LIGNES CLÉS ICI : */
        white-space: nowrap !important; 
        overflow: visible !important;   
        display: block !important;
    }
    /* 5. STYLE DU TEXTE (Noir & Centré) */
    div[data-testid="stRadio"] label p {
        color: #1d1d1d !important; 
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
        opacity: 1 !important;
    }

    /* 6. L'ONGLET ACTIF : Pilule Rouge */
    div[data-testid="stRadio"] div[data-checked="true"] {
        background-color: #c8102e !important; 
        border-color: #c8102e !important;
        box-shadow: 0 4px 6px rgba(200, 16, 46, 0.2) !important; /* Petite ombre rouge */
    }

    /* Texte en Blanc pour l'onglet actif */
    div[data-testid="stRadio"] div[data-checked="true"] label p {
        color: #ffffff !important;
    }

    /* 7. EFFET AU SURVOL (Hover) */
    div[data-testid="stRadio"] label:hover {
        background-color: #e8e8e8 !important; /* Gris un peu plus foncé au survol */
        border-color: #d1d1d1 !important;
    }

    /* Pas d'effet de survol sur l'onglet déjà actif */
    div[data-testid="stRadio"] div[data-checked="true"] label:hover {
        background-color: #c8102e !important;
    }

    /* --- HERO MIX VIDÉO & IMAGES (CORRIGÉ) --- */
    .hero-wrap {
        position: relative;
        width: calc(100% + 4rem);
        margin-left: -2rem;
        margin-right: -2rem;
        margin-top: -1.5rem;
        height: 600px;
        overflow: hidden;;
        /* On met l'image 1 en fond de secours */
        background-image: url('https://images.unsplash.com/photo-1607662255484-562ef2863903?q=80&w=1474');
        background-size: cover;
        background-position: center;
    }

    .hero-element {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        opacity: 0;
        transform: scale(1.15);
        animation: heroCrossFade 40s infinite ease-in-out;
    }

    .el-1 { animation-delay: 0s; opacity: 1 !important;}
    .el-2 { animation-delay: 4s; }
    .el-3 { animation-delay: 8s; }
    .el-4 { animation-delay: 12s; }
    .el-5 { animation-delay: 16s; }
    .el-6 { animation-delay: 20s; }
    .el-7 { animation-delay: 24s; }
    .el-8 { animation-delay: 28s; }
    .el-9 { animation-delay: 32s; }
    .el-10 { animation-delay: 36s; }

    @keyframes heroCrossFade {
    0% { opacity: 0; transform: scale(1.15); }
    5% { opacity: 1; transform: scale(1.08); }
    15% { opacity: 1; transform: scale(1.05); }
    25% { opacity: 0; transform: scale(1.0); }
    100% { opacity: 0; }
}

    /* Animation du texte tournant (Flipper) */
    .word-flipper {
        display: inline-block;
        position: relative;
        text-align: left;
        min-width: 500px;
    }

    .word-flipper span {
        position: absolute;
        left: 0;
        top: 0;
        opacity: 0;
        color: #c8102e;
        white-space: nowrap;
        animation: flipWord 8s infinite;
    }

    /* 4 mots qui tournent toutes les 2 secondes */
    .word-flipper span:nth-child(1) { animation-delay: 0s; }
    .word-flipper span:nth-child(2) { animation-delay: 2s; }
    .word-flipper span:nth-child(3) { animation-delay: 4s; }
    .word-flipper span:nth-child(4) { animation-delay: 6s; }

    @keyframes flipWord {
        0% { opacity: 0; transform: translateY(20px); }
        5% { opacity: 1; transform: translateY(0); }
        20% { opacity: 1; transform: translateY(0); }
        25% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 0; }
    }

    .hero-overlay {
        position: absolute;
        inset: 0;
        background: radial-gradient(circle, rgba(0,0,0,0) 20%, rgba(0,0,0,0.7) 100%);
        z-index: 2;
    }

    .hero-content {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 3;
        text-align: center;
        width: 100%;
    }

    /* SECTION */
    .section-title {
        text-align: center;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        color: #222 !important;
        margin-top: 0.4rem;
        margin-bottom: 0.45rem;
    }

    .section-subtitle {
        text-align: center;
        color: var(--k-muted) !important;
        font-size: 1.02rem !important;
        margin-bottom: 2.1rem;
    }

   
    /* CARDS AREA */
    .cards-wrap {
        max-width: 1040px;
        margin: 0 auto;
    }

    div[data-testid="stHorizontalBlock"] {
        max-width: 1040px;
        margin-left: auto;
        margin-right: auto;
        align-items: flex-start;
        
    }

    .cluster-card {
        background: var(--k-white);
        border: 1px solid var(--k-border);
        overflow: hidden;
        box-shadow: none;
        width: 100%;
        max-width: 470px;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 0.45rem;
    }

    .cluster-image-wrap {
        width: 100%;
        height: 250px;
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center center;
        display: block;
    }

    .cluster-body {
        padding: 0.95rem 1rem 1.05rem 1rem;
        min-height: 230px;   /* hauteur fixe pour aligner les boutons */
    }


    .cluster-tag {
        color: #666;
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.45rem;
    }

    .cluster-name {
        color: #202020 !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        line-height: 1.15 !important;
        margin-bottom: 0.65rem;
    }


    .cluster-desc {
        color: #3f3f3f !important;
        font-size: 1rem !important;
        line-height: 1.55 !important;
        margin-bottom: 0.75rem;
        min-height: 72px;
    }


    .cluster-meta {
        color: #333 !important;
        font-size: 0.92rem !important;
        line-height: 1.55 !important;
        min-height: 58px;   /* réserve la même zone meta */
        margin-bottom: 0;
    }
    .cluster-button-wrap {
    margin-top: auto;
    padding-top: 0.4rem;
}

    div[data-testid="column"] {
    display: flex;
    flex-direction: column;
}

    div[data-testid="column"] > div {
        width: 100%;
    }

    /* BUTTON */
    /* --- NOUVEAU STYLE BOUTONS NAVIGATION --- */
    .stButton > button {
        background:  #c8102e  !important; 
        color:  white !important;      /* Texte noir */
        border: 1px solid #e1e1e1 !important;
        border-radius: 50px !important; /* Forme pilule */
        margin: 0 10px !important; /* Ajoute 10px d'espace à gauche et à droite de chaque bouton */

        
        /* FORCE LA STRUCTURE */
        height: 45px !important;        /* TOUS la même hauteur exacte */
        width: auto !important;         /* S'étire selon le texte */
        min-width: 140px !important;    /* Mais pas trop petit non plus */
        padding: 0 25px !important;     /* Espace sur les côtés */
        
        /* EMPECHE LE RETOUR A LA LIGNE */
        white-space: nowrap !important; 
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }

    /* Effet au survol */
    .stButton > button:hover {
        background: #e8e8e8 !important;
        border-color: #c8102e !important;
        color: #c8102e !important;
    }

    /* Pour identifier le bouton "actif" (optionnel, selon ta logique) */
    .stButton > button:active, .stButton > button:focus {
        background: #f1f1f1 !important;
        color: black !important;
    }

    .small-note {
        text-align: center;
        color: var(--k-muted);
        font-size: 0.95rem;
        margin-top: 0.8rem;
    }

    /* Streamlit headings in other tabs */
    h1, h2, h3 {
        color: #222 !important;
    }

    @media (max-width: 900px) {
        .hero-wrap {
            height: 390px;
        }

        .hero-title,
        .hero-wrap h1,
        .hero-content h1 {
            font-size: 3rem !important;
        }

        .cluster-card {
            max-width: 100%;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .hero-wrap {
            width: calc(100% + 2rem);
            margin-left: -1rem;
            margin-right: -1rem;
            height: 330px;
        }

        .hero-title,
        .hero-wrap h1,
        .hero-content h1 {
            font-size: 2.25rem !important;
        }

        .hero-subtitle,
        .hero-wrap p,
        .hero-content p {
            font-size: 1rem !important;
        }

        .section-title {
            font-size: 1.8rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# DONNÉES VISUELLES DES CLUSTERS
# =========================
clusters_home = [
    {
        "name": "Desserts fruités et lactés",
        "tag": "Douceur",
        "image": "https://plus.unsplash.com/premium_photo-1713551474697-15fe83485bc7?q=80&w=1469&auto=format&fit=crop",
        "image_position": "center center",
        "desc": "Des recettes douces, fraîches et gourmandes autour des fruits, crèmes et textures lactées.",
        "count": "Top cluster sucré",
        "top_ing": "fruits • lait • crème"
    },
    {
        "name": "Desserts chocolatés",
        "tag": "Gourmandise",
        "image": "https://images.unsplash.com/photo-1511381939415-e44015466834?q=80&w=2072&auto=format&fit=crop",
        "image_position": "center center",
        "desc": "L'univers du cacao sous toutes ses formes : gâteaux fondants, mousses et douceurs intenses.",
        "count": "100% Plaisir",
        "top_ing": "chocolat • beurre • sucre"
    },
    {
        "name": "Cuisine Marine et Exotique",
        "tag": "Évasion",
        "image": "https://images.unsplash.com/photo-1514326640560-7d063ef2aed5?q=80&w=2080&auto=format&fit=crop",
        "image_position": "center center",
        "desc": "Un voyage gustatif entre poissons frais, saveurs marines et touches asiatiques parfumées.",
        "count": "Saveurs d'ailleurs",
        "top_ing": "saumon • citron • soja"
    },
    {
        "name": "Pâtisserie et gâteaux de base",
        "tag": "Boulangerie",
        "image": "https://images.unsplash.com/photo-1555507036-ab1f4038808a?q=80&w=1926&auto=format&fit=crop",
        "image_position": "center center",
        "desc": "Les indispensables de la pâtisserie : gâteaux, tartes et pâtes levées faites maison.",
        "count": "Les essentiels",
        "top_ing": "farine • levure • œufs"
    },
    {
        "name": "Plats salés du quotidien",
        "tag": "Famille",
        "image": "https://plus.unsplash.com/premium_photo-1723662129814-99431b2f57cd?q=80&w=836&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "image_position": "center center",
        "desc": "Une cuisine simple et efficace à base de volailles et légumes pour tous les jours.",
        "count": "Pratique & Bon",
        "top_ing": "poulet • oignon • poivron"
    },
    {
        "name": "Accompagnements et féculents",
        "tag": "Réconfort",
        "image": "https://plus.unsplash.com/premium_photo-1669261882830-1e504a9abf1d?q=80&w=687&auto=format&fit=crop",
        "image_position": "center center",
        "desc": "Des accompagnements généreux où la pomme de terre se décline sous toutes ses formes.",
        "count": "Cuisine terroir",
        "top_ing": "pomme de terre • crème • beurre"
    },
    {
        "name": "Plats italien",
        "tag": "Populaire",
        "image": "https://images.unsplash.com/photo-1635264685671-739e75e73e0f?q=80&w=1964&auto=format&fit=crop",
        "image_position": "center center",
        "desc": "Pizzas, mozzarella et saveurs ensoleillées inspirées par la tradition italienne.",
        "count": "Cluster convivial",
        "top_ing": "pizza • jambon • mozzarella"
    },
    {
        "name": "Cuisine méditerranéenne",
        "tag": "Soleil",
        "image": "https://images.unsplash.com/photo-1574788032365-69e929e3ec68?q=80&w=1964&auto=format&fit=crop",
        "image_position": "center center",
        "desc": "Une cuisine colorée avec viandes mijotées, herbes fraîches et huile d’olive.",
        "count": "Cuisine saine",
        "top_ing": "huile d'olive • ail • bœuf"
    },
]

# =========================
# NAVIGATION (STYLE TABS)
# =========================
menu_options = ["Accueil", "Recommandations", "Assistant Gourmi", "Communauté"]

# On crée une structure de colonnes pour centrer les boutons
# Les [2, 1, 1, 1, 1, 2] servent de marges à gauche et à droite
# On donne plus de poids (2 au lieu de 1) aux colonnes du milieu
_, c1, c2, c3, c4, _ = st.columns([1.5, 1.5, 1.8, 1.8, 1.8, 1.5])

with c1:
    if st.button("Accueil", use_container_width=True, key="nav_0"):
        st.session_state["active_tab"] = 0
        st.rerun()
with c2:
    if st.button("Recommandations", use_container_width=True, key="nav_1"):
        st.session_state["active_tab"] = 1
        st.session_state["selected_recipe"] = None
        
        # --- SOLUTION : RESET DES CLÉS DES WIDGETS ---
        st.session_state["search_query"] = "" 
        st.session_state["selected_cluster_filter"] = []
        
        # On vide aussi les clés internes des widgets s'ils existent
        if "search_input_widget" in st.session_state:
            st.session_state["search_input_widget"] = ""
        if "cluster_widget" in st.session_state:
            st.session_state["cluster_widget"] = []
            
        st.rerun()
        
with c3:
    if st.button("Assistant Gourmi", use_container_width=True, key="nav_2"):
        st.session_state["active_tab"] = 2
        st.rerun()
with c4:
    if st.button("Communauté", use_container_width=True, key="nav_3"):
        st.session_state["active_tab"] = 3
        st.rerun()

# On ajoute une ligne de séparation propre en dessous
st.markdown("<hr style='margin:0; border:0; border-bottom:1px solid #ddd;'>", unsafe_allow_html=True)


# =========================
# TAB 1 — ACCUEIL
# =========================
if st.session_state["active_tab"] == 0:
    hero_html = """
    <div class="hero-wrap">
        <img class="hero-element el-1" src="https://images.unsplash.com/photo-1607662255484-562ef2863903?q=80&w=1474&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D">
        <img class="hero-element el-2" src="https://images.unsplash.com/photo-1594385200330-b73b2fc3ca84?q=80&w=2070">
        <img class="hero-element el-3" src="https://images.unsplash.com/photo-1473093226795-af9932fe5856?q=80&w=2070">
        <img class="hero-element el-4" src="https://images.unsplash.com/photo-1506368249639-73a05d6f6488?q=80&w=2070">
        <img class="hero-element el-5" src="https://images.unsplash.com/photo-1414235077428-338989a2e8c0?q=80&w=2070">
        <img class="hero-element el-6" src="https://images.unsplash.com/photo-1455619452474-d2be8b1e70cd?q=80&w=2070">
        <img class="hero-element el-7" src="https://images.unsplash.com/photo-1546069901-ba9599a7e63c?q=80&w=2070">
        <img class="hero-element el-8" src="https://images.unsplash.com/photo-1466637574441-749b8f19452f?q=80&w=2070">
        <img class="hero-element el-9" src="https://images.unsplash.com/photo-1482049016688-2d3e1b311543?q=80&w=2070">
        <img class="hero-element el-10" src="https://images.unsplash.com/photo-1490645935967-10de6ba17061?q=80&w=2070">

        <div class="hero-overlay"></div>
        <div class="hero-content">
            <h1 style="color: white !important; font-size: 5rem !important; font-weight: 900; letter-spacing: -0.05em; line-height: 1.1; margin: 0;">
                VOTRE MENU <br>
                <span class="word-flipper">
                    <span>SÉLECTIF</span>
                    <span>SUR-MESURE</span>
                    <span>OPTIMISÉ</span>
                    <span>INSPIRÉ</span>
                </span>
            </h1>
            <div style="margin-top: 80px; text-shadow: 2px 2px 10px rgba(0,0,0,0.6);">
                <p style="
                    color: #FFFFFF !important; 
                    font-size: 0.9rem !important; 
                    letter-spacing: 0.3em; /* Espacement très large pour le côté luxe */
                    margin-bottom: 20px; 
                    font-weight: 1000; 
                    text-transform: uppercase;
                    opacity: 1;
                    border-top: 1px solid rgba(255,255,255,0.3); /* Petite ligne fine au-dessus */
                    border-bottom: 1px solid rgba(255,255,255,0.3); /* Petite ligne fine en-dessous */
                    display: inline-block; /* Pour que les lignes ne prennent que la largeur du texte */
                    padding: 8px 0;
                    line-height: 1;
                ">
                    Fini les 1000 résultats génériques
                </p>

                <h1 style="color: #c8102e !important; font-weight: 950 !important; font-size: 3.5rem !important; letter-spacing: -0.05em; margin-bottom: 0; line-height: 1;">
                        L'IA GOURMI
                    </h1>

                    <p style="color: white !important; font-weight: 800; font-size: 1.6rem; margin-top: 10px; letter-spacing: -0.01em;">
                        LE TOP 5 DE <span style="border-bottom: 4px solid #c8102e;">VOS INGRÉDIENTS</span>
                    </p>

                    <div style="margin-top: 30px; display: flex; justify-content: center; gap: 20px; color: rgba(255,255,255,0.9);">
                        <span style="font-weight: 700; font-size: 0.9rem; border: 1px solid #c8102e; padding: 5px 15px; border-radius: 50px;">
                            ALGORITHMES NLP
                        </span>
                        <span style="font-weight: 700; font-size: 0.9rem; border: 1px solid #c8102e; padding: 5px 15px; border-radius: 50px;">
                            SÉLECTION INTELLIGENTE
                        </span>
                    </div>
            </div>
        </div>
    </div>
    """
    # ON APPELLE LE RENDU UNE SEULE FOIS
    st.html(hero_html)

    st.markdown('<h2 class="section-title">Explorer les cuisines</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Cliquez sur une catégorie pour commencer votre exploration culinaire.</p>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="cards-wrap">', unsafe_allow_html=True)

    cols = st.columns([1, 1], gap="small")

    for i, cluster in enumerate(clusters_home):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="cluster-card">
                    <div 
                        class="cluster-image-wrap"
                        style="
                            background-image: url('{cluster['image']}');
                            background-position: {cluster['image_position']};
                        ">
                    </div>
                    <div class="cluster-body">
                        <div class="cluster-tag">{cluster['tag']}</div>
                        <div class="cluster-name">{cluster['name']}</div>
                        <div class="cluster-desc">{cluster['desc']}</div>
                        <div class="cluster-meta">
                            <strong>Profil :</strong> {cluster['count']}<br>
                            <strong>Top ingrédients :</strong> {cluster['top_ing']}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(f"Découvrir : {cluster['name']}", key=f"btn_{i}"):
                # 1. On stocke le cluster
                st.session_state["selected_cluster_filter"] = [cluster['name']]
                # 2. ON FORCE LE CHANGEMENT D'ONGLET (vers l'index 1 : Recommandation)
                st.session_state["active_tab"] = 1
                # 3. On relance l'app
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

  

# =========================
# TAB 2 — RECOMMANDATION
# =========================
elif st.session_state["active_tab"] == 1:

    st.markdown("""
    <style>
        .cluster-title {
            font-size: 1.7rem;
            font-weight: 800;
            color: #000000;
            margin-top: 40px;
            margin-bottom: 15px;
        }

        .recipe-card {
            width: 100%; /* S'adapte à la colonne centrée sans déborder */
            margin-bottom: 20px;
        }

        .recipe-image {
                width: 100%; 
                aspect-ratio: 1 / 1; 
                object-fit: cover;
                border-radius: 12px; /* Un peu plus arrondi pour le style Tasty */
                margin-bottom: 12px; /* Plus d'espace sous l'image */
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Optionnel : petite ombre pour faire ressortir */
            }
            
        .recipe-name {
                font-size: 1.1rem; /* Texte plus grand pour accompagner la grande image */
                font-weight: 700;
                line-height: 1.2;
                color: #000000 !important;
            }
                

        /* --- DESIGN PAGE RECETTE --- */
        .recipe-header-title {
            font-size: 3rem !important;
            font-weight: 800 !important;
            text-align: center;
            color: var(--k-text);
            margin-top: 2rem;
            text-transform: uppercase;
        }
        .recipe-meta-box {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 2rem;
            color: var(--k-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9rem;
        }
        .prep-container {
            background: white;
            padding: 40px;
            border-radius: 20px;
            margin-top: 20px;
            color: #1d1d1d !important; /* FORCE LE TEXTE EN NOIR */
        }
        .ing-title, .step-title {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 1.5rem !important;
            border-bottom: 3px solid var(--k-red);
            display: inline-block;
            padding-bottom: 5px;
            color: #1d1d1d;
        }
        /* Style des numéros d'étapes (Cercles rouges) */
        .step-number {
            background-color: var(--k-red);
            color: white;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
            flex-shrink: 0;
        }
                
        .step-row {
            display: flex;
            align-items: flex-start;
            margin-bottom: 20px;
            line-height: 1.6;
            color: #1d1d1d !important; /* FORCE LE TEXTE DES ÉTAPES EN NOIR */
        }


    </style>
    """, unsafe_allow_html=True)


    # --- 1. AFFICHAGE DÉTAILLÉ (Si une recette est cliquée) ---
    if st.session_state.get("selected_recipe") is not None:
        recipe = st.session_state["selected_recipe"]
        
        # --- BLOC DYNAMIQUE ICI ---
        # On définit le texte du bouton selon la provenance
        if st.session_state.get("recipe_source") == 2:
            label_retour = "← Retour à l'Assistant Gourmi"
        else:
            label_retour = "← Retour aux recommandations"

        if st.button(label_retour):
            # 1. On enlève la recette sélectionnée pour revenir à la liste
            st.session_state["selected_recipe"] = None  
            
            # 2. On change d'onglet si on vient de l'assistant, sinon on reste sur le 1
            if st.session_state.get("recipe_source") == 2:
                st.session_state["active_tab"] = 2
            else:
                st.session_state["active_tab"] = 1 
                
            # SURTOUT : Ne pas faire st.session_state["search_query"] = "" ici !
            st.rerun()

        # Design Header
        st.markdown(f'<div class="recipe-header-title">{recipe["titre"]}</div>', unsafe_allow_html=True)
        
        # Meta Infos
        st.markdown(f"""
            <div class="recipe-meta-box">
                <span>⏱️ {int(recipe['temps_total'])} MIN</span>
                <span>💰 {recipe['cout']}</span>
                <span>⚙️ {recipe['difficulte']}</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Image centrée avec sécurité anti-vide
        _, col_img, _ = st.columns([1, 4, 1])
        with col_img:
            # On récupère l'image. Si elle est vide ou None, on met l'URL par défaut
            img_path = recipe.get("image")
            if not img_path or img_path == "":
                img_path = "https://images.unsplash.com/photo-1556911220-e15b29be8c8f?q=80&w=1000&auto=format&fit=crop"
            
            st.image(img_path, use_container_width=True)

        # Corps de la recette : Ingrédients à gauche | Étapes à droite
        st.markdown('<div class="prep-container">', unsafe_allow_html=True)
        col_ing, col_steps = st.columns([1, 2], gap="large")

        with col_ing:
            st.markdown('<div class="ing-title">INGRÉDIENTS</div>', unsafe_allow_html=True)
            # On cherche les ingrédients dans les colonnes possibles
            ings = recipe.get("ingredients_clean_final", recipe.get("ingredients_clean_final_x", []))
            if isinstance(ings, list):
                for ing in ings:
                    # ICI : On remplace le st.markdown simple par celui avec la couleur forcée
                    st.markdown(f'<div style="color: #1d1d1d; margin-bottom: 5px;">• {ing.capitalize()}</div>', unsafe_allow_html=True)
            else:
                st.write("Détails des ingrédients non disponibles.")


        with col_steps:
            st.markdown('<div class="step-title">PRÉPARATION</div>', unsafe_allow_html=True)
            etapes_raw = recipe.get("etapes", recipe.get("etapes_x", "[]"))
            try:
                import ast
                etapes_list = ast.literal_eval(etapes_raw) if isinstance(etapes_raw, str) else etapes_raw
                for i, step in enumerate(etapes_list, start=1):
                    st.markdown(f"""
                        <div class="step-row">
                            <div class="step-number">{i}</div>
                            <div style="flex:1; padding-top:4px;">{step.strip()}</div>
                        </div>
                    """, unsafe_allow_html=True)
            except:
                st.write("Les étapes ne sont pas disponibles.")

            # --- AJOUT DU LIEN ICI ---
            url_origine = recipe.get("url", "#") # Remplace "url" par le nom exact de ta colonne lien si différent
            st.markdown(f"""
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px dashed #e1e1e1; text-align: center;">
                    <a href="{url_origine}" target="_blank" style="color: #c8102e; text-decoration: none; font-weight: 700; font-size: 1rem;">
                        Consulter la recette originale sur le site source →
                    </a>
                </div>
            """, unsafe_allow_html=True)
            # --------------------------

        st.markdown('</div>', unsafe_allow_html=True)


    # --- 2. AFFICHAGE RECOMMANDATION NORMAL (Si rien n'est sélectionné) ---
    else:
        # Styles CSS (Tes styles originaux + les nouveaux pour la fiche)
        st.markdown("""
        <style>
            .cluster-title { font-size: 1.7rem; font-weight: 800; color: #000000; margin-top: 40px; margin-bottom: 15px; }
            .recipe-card { width: 100%; margin-bottom: 20px; cursor: pointer; }
            .recipe-image { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .recipe-name { font-size: 1.1rem; font-weight: 700; line-height: 1.2; color: #000000 !important; }
            
            /* Styles Page Recette */
            .recipe-header-title { font-size: 3rem !important; font-weight: 800 !important; text-align: center; color: #1d1d1d; margin-top: 2rem; text-transform: uppercase; }
            .recipe-meta-box { display: flex; justify-content: center; gap: 30px; margin-bottom: 2rem; color: #666; font-weight: 600; text-transform: uppercase; font-size: 0.9rem; }
            .prep-container { background: white; padding: 40px; border-radius: 20px; margin-top: 20px; border: 1px solid #e1e1e1; }
            .ing-title, .step-title { font-size: 1.5rem !important; font-weight: 700 !important; margin-bottom: 1.5rem !important; border-bottom: 3px solid #c8102e; display: inline-block; padding-bottom: 5px; }
            .step-number { background-color: #c8102e; color: white; border-radius: 50%; width: 32px; height: 32px; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 15px; flex-shrink: 0; }
            .step-row { display: flex; align-items: flex-start; margin-bottom: 20px; line-height: 1.6; }
        </style>
        """, unsafe_allow_html=True)

        # PRÉPARATION DONNÉES
        df_chatbot['temps_total'] = pd.to_numeric(df_chatbot['temps_total'], errors='coerce').fillna(0)

        # BARRE RECHERCHE & FILTRES
        
        left, center, right = st.columns([1, 6, 1])
        with center:
            # On ajoute key="search_query" pour que le texte reste en mémoire lors du "Retour"
            current_search = st.session_state.get("search_query", "")

            # 2. On affiche l'input en utilisant une clé différente pour le widget
            search = st.text_input(
                "Recherche", 
                value=current_search, 
                placeholder="Rechercher une recette ou un ingrédient", 
                key="search_input_widget" 
            )
            # On met à jour manuellement le state pour la prochaine fois
            st.session_state["search_query"] = search

            f1, f2, f3, f4 = st.columns(4)
            with f1:
                sel_clusters = st.multiselect(
                    "Catégories", 
                    options=sorted(df_chatbot['cluster_name'].unique().tolist()), 
                    default=st.session_state.get("selected_cluster_filter", []), # Utilise default
                    key="cluster_widget"
                )
                st.session_state["selected_cluster_filter"] = sel_clusters
            with f2:
                sel_diff = st.selectbox("Difficulté", options=["Toutes"] + sorted([str(x) for x in df_chatbot['difficulte'].dropna().unique()]))
            with f3:
                sel_cout = st.selectbox("Budget", options=["Tous"] + sorted([str(x) for x in df_chatbot['cout'].dropna().unique()]))
            with f4:
                max_bdd = int(df_chatbot['temps_total'].max()) if not df_chatbot.empty else 120
                sel_temps = st.number_input("Temps max (min)", min_value=0, max_value=max_bdd, value=max_bdd, step=5)

        # LOGIQUE FILTRAGE
        # 1. On définit l'image par défaut (Option 3)
        IMG_PAR_DEFAUT = "https://images.unsplash.com/photo-1556911220-e15b29be8c8f?q=80&w=1000&auto=format&fit=crop"

        # 2. On garde tout le monde mais on remplace les vides par l'image par défaut
        df_display = df_chatbot.copy()
        df_display['image'] = df_display['image'].apply(lambda x: x if pd.notna(x) and x != "" else IMG_PAR_DEFAUT)
        if search:
           
            
            # 1. On nettoie la recherche : remplace virgules/points par espaces + minuscule
            clean_search = re.sub(r'[.,]', ' ', search.lower())
            
            # 2. On crée la liste des mots sans les "s" finaux (si mot > 3 lettres)
            # ex: "poulets" devient "poulet", "riz" reste "riz"
            search_words = [w[:-1] if w.endswith('s') and len(w) > 3 else w for w in clean_search.split()]

            def match_all_words(row):
                # On récupère le texte du titre et des ingrédients en gérant les cases vides (NaN)
                titre = str(row.get("titre", "")).lower()
                # On cherche dans "ingredients_str" ou "ingredients_clean_final" selon ce qui existe
                ingredients = str(row.get("ingredients_str", row.get("ingredients_clean_final", ""))).lower()
                
                text_to_search = titre + " " + ingredients
                
                # On vérifie si CHAQUE mot de la recherche est dans le texte
                # (On cherche la racine du mot pour ignorer le pluriel)
                return all(word in text_to_search for word in search_words)

            # On applique le filtre
            df_display = df_display[df_display.apply(match_all_words, axis=1)]
            
        if sel_clusters:
            df_display = df_display[df_display["cluster_name"].isin(sel_clusters)]
        if sel_diff != "Toutes":
            df_display = df_display[df_display["difficulte"] == sel_diff]
        if sel_cout != "Tous":
            df_display = df_display[df_display["cout"] == sel_cout]
        df_display = df_display[df_display["temps_total"] <= sel_temps]

        # AFFICHAGE
        clusters_presents = df_display["cluster_name"].unique()
        if len(clusters_presents) == 0:
            st.info("Désolé, aucune recette ne correspond.")
        else:
            filtre_actif = bool(search or sel_clusters or sel_diff != "Toutes" or sel_cout != "Tous")
            for idx_c, cluster_nom in enumerate(clusters_presents):
                cluster_df = df_display[df_display["cluster_name"] == cluster_nom].copy()
                l_t, c_t, r_t = st.columns([1, 6, 1])
                with c_t:
                    st.markdown(f'<div class="cluster-title">{cluster_nom}</div>', unsafe_allow_html=True)

                recipes_to_show = cluster_df if filtre_actif else cluster_df.head(4)
                recipes_to_show = recipes_to_show.reset_index(drop=True)


                for i in range(0, len(recipes_to_show), 4):
                    _, center_row, _ = st.columns([0.5, 11, 0.5])
                    with center_row:
                        cols = st.columns(4, gap="large")
                        batch = recipes_to_show.iloc[i : i + 4]
                        for idx_r, (_, row) in enumerate(batch.iterrows()):
                            
                            with cols[idx_r]:
                                
                                st.markdown(f"""
                                    <div class="recipe-card">
                                        <img class="recipe-image" src="{row['image']}">
                                        <div style="height: 110px; overflow: hidden; margin-bottom: 10px;">
                                            <div class="recipe-name" style="display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;">
                                                {row['titre']}
                                            </div>
                                            <div style="font-size:0.85rem; color:#666; margin-top:4px;">
                                                ⏱️ {int(row['temps_total'])} min | 💰 {row['cout']}
                                            </div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # 2. Création de la clé unique
                                clean_title = str(row['titre']).replace(" ", "_")[:20]
                                unique_key = f"btn_{idx_c}_{i}_{idx_r}_{clean_title}"
                                
                                # 3. Le bouton Streamlit s'affichera maintenant toujours à la même position verticale
                                if st.button("Voir la recette", key=unique_key, use_container_width=True):
                                    st.session_state["selected_recipe"] = row
                                    st.session_state["recipe_source"] = 1 
                                    st.rerun()



# ========================================================
# TAB 3 — ASSISTANT GOURMI (Version Centrée & Pro)
# ========================================================
elif st.session_state["active_tab"] == 2:
    import re
    from nlp_fonctions import agent_gourmi_master, culinary_chatbot_step2, reset_gourmi

    # --- STYLE CSS ---
    st.markdown("""
        <style>
        /* 1. CONTENEUR GLOBAL POUR L'ALIGNEMENT */
        [data-testid="stChatMessageContainer"] {
            display: flex !important;
            flex-direction: column !important;
            width: 100% !important;
        }

        /* 2. STYLE COMMUN (Largeur auto, pas d'avatar) */
        [data-testid="stChatMessage"] {
            width: fit-content !important; 
            max-width: 80% !important;
            border-radius: 15px !important;
            margin-bottom: 10px !important;
        }
        [data-testid="stChatMessageAvatarContainer"] {
            display: none !important;
        }

        /* 3. GOURMI (Assistant) : FORCE LE BLEU ET L'ALIGNEMENT GAUCHE */
        /* On cible l'attribut 'user' car Streamlit marque souvent l'IA comme 'assistant' */
        [data-testid="stChatMessage"][data-testimonial="assistant"], 
        [data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {
            background-color: #FAF7F7 !important; 
            align-self: flex-start !important;
            margin-right: auto !important;
            color: black !important; /* <--- ICI POUR LE NOIR */
        }
                
        /* 1. L'état normal : On cible le bouton ET le paragraphe à l'intérieur */
        [data-testid="stChatMessage"] .stButton > button, 
        [data-testid="stChatMessage"] .stButton > button p {
            background: #BABABA !important; 
            color: white !important;             /* Force le texte en blanc */
            border-color: white !important;
            transition: all 0.2s ease !important;
        }

        /* 2. L'effet au survol (HOVER) */
        [data-testid="stChatMessage"] .stButton > button:hover,
        [data-testid="stChatMessage"] .stButton > button:hover p {
            background: #960202 !important; 
            color: white !important;
        }

        /* 3. L'effet au clic (ACTIVE) */
        [data-testid="stChatMessage"] .stButton > button:active,
        [data-testid="stChatMessage"] .stButton > button:active p {
            background: #8E2E2E !important; 
            color: #f1f1f1 !important;
        }
                
        /* 4. UTILISATEUR (Toi) : FORCE LE ROUGE ET L'ALIGNEMENT DROITE */
        [data-testid="stChatMessage"][data-testimonial="user"],
        [data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
            background-color: #991212 !important; 
            align-self: flex-end !important;
            margin-left: auto !important;
            color: white !important; /* <--- ICI POUR LE BLANC */
        }
                
        /* Ciblage précis du texte de Gourmi (Noir) */
        [data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) p,
        [data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) li {
            color: black !important;
        }

        /* Ciblage précis du texte Utilisateur (Blanc) */
        [data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) p,
        [data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) li {
            color: white !important;
        }
                
        /* --- MÉTHODE RADICALE POUR SUPPRIMER LES AVATARS --- */
        
        /* 1. Cible le conteneur d'avatar et le force à disparaître sans laisser de trace */
        [data-testid="stChatMessageAvatarContainer"] {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* 2. Cible spécifiquement la colonne de gauche (l'avatar) dans la structure de chat */
        [data-testid="stChatMessage"] > div:first-child {
            display: none !important;
        }

        /* 3. On s'assure que la bulle de texte prend toute la place interne */
        [data-testid="stChatMessageContent"] {
            margin-left: 0 !important;
            padding: 0 !important;
        }

        /* 4. On réajuste les marges de la bulle pour qu'elle soit bien propre */
        [data-testid="stChatMessage"] {
            padding: 12px 18px !important;
        }
                   
        </style>
    """, unsafe_allow_html=True)

    # --- STRUCTURE CENTRÉE [1, 6, 1] ---
    left_gap, center_chat, right_gap = st.columns([1, 6, 1])



    # 1. Fonction pour encoder l'image du dossier 'logo'
    def get_base64_image(rel_path):
        # On construit le chemin complet (ex: logo/assistant.jpg)
        full_path = os.path.join(os.getcwd(), rel_path)
        if os.path.exists(full_path):
            with open(full_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        return None

    # 2. On récupère l'image spécifique
    img_data = get_base64_image("logo/assistant.png")

    if img_data:
        bg_style = f"url('data:image/jpeg;base64,{img_data}')"
    else:
        # Si l'image n'est pas trouvée, on met un fond rouge dégradé par défaut
        bg_style = "linear-gradient(45deg, #c8102e, #750202)"

    with center_chat:
        # --- LE BORDEREAU AVEC L'IMAGE DU DOSSIER LOGO ---
        st.markdown(
            f"""
            <div style="
                position: relative;
                width: calc(100% + 4rem);
                margin-left: -2rem;
                margin-top: -1.5rem;
                height: 250px;
                background-image: linear-gradient(rgba(0,0,0,0.2), rgba(0,0,0,0.2)), 
                                {bg_style};
                background-size: cover;
                background-position: center;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 2rem;
                border-bottom: 4px solid #c8102e;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            ">
                
            </div>
            """,
            unsafe_allow_html=True
        )
    
   

        # INITIALISATION DANS L'ONGLET 2
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "Bonjour ! Je suis **Gourmi-IA**, votre assistant culinaire personnel. \n\nJe peux vous suggérer des recettes, vous donner des conseils techniques ou vous aider à cuisiner avec ce qu'il reste dans votre frigo. **Que cuisine-t-on aujourd'hui ?**"
                }
            ]
        if "last_recs" not in st.session_state:
            st.session_state.last_recs = None

        
        # 3. AFFICHAGE DE L'HISTORIQUE
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                
                # On récupère 'clean_content' s'il existe, sinon on prend le 'content' brut
                
                texte_a_afficher = message.get("clean_content", message["content"])
                st.markdown(texte_a_afficher)

                # On affiche l'image seulement si 'img_url' existe ET n'est pas vide
                url_image = message.get("img_url")
                if url_image:
                    st.image(url_image, use_container_width=True)

                # --- LE BOUTON DE FICHE ---
                # On vérifie si les données de la recette sont stockées dans ce message
                if "recipe_data" in message:
                    # On crée un bouton unique pour chaque message grâce à l'index i
                    if st.button("Voir la recette", key=f"btn_chat_{i}"):
                        st.session_state["selected_recipe"] = message["recipe_data"]
                        st.session_state["recipe_source"] = 2
                        st.session_state["active_tab"] = 1
                        st.rerun()

        # 4. ZONE DE SAISIE
        if prompt := st.chat_input("Ex: 'Un plat rapide avec du saumon'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # On cherche si l'utilisateur a tapé un chiffre 1-5
            match_numero = re.search(r'\b([1-5])\b', prompt.strip())

            if match_numero and st.session_state.last_recs is not None:
                idx_choisi = int(match_numero.group(1)) - 1
                
                with st.spinner("Récupération de la recette..."):
                    recipe_row = st.session_state.last_recs.iloc[idx_choisi]
                    response = culinary_chatbot_step2(st.session_state.last_recs, prompt.strip())
                
                # Préparation du dictionnaire de message
                new_msg = {
                    "role": "assistant", 
                    "content": response,
                    "recipe_data": recipe_row # Présent peu importe l'image
                }

                # Extraction de l'image si présente
                if "🖼️ Image :" in response:
                    parts = response.split("🖼️ Image :")
                    new_msg["clean_content"] = parts[0].strip()
                    
                    # Nettoyage de l'URL
                    raw_url = parts[1].split("\n")[0].strip()
                    
                    # SÉCURITÉ : Si l'URL extraite est vide, on met celle par défaut
                    if not raw_url or raw_url == "":
                        new_msg["img_url"] = "https://images.unsplash.com/photo-1556911220-e15b29be8c8f?q=80&w=1000&auto=format&fit=crop"
                    else:
                        new_msg["img_url"] = raw_url
                
                st.session_state.messages.append(new_msg)
            
            else:
                # Recherche classique
                with st.spinner("Gourmi réfléchit..."):
                    recs, message_bot = agent_gourmi_master(
                        prompt, df, df_chatbot, tfidf_both, vec_both, VOCAB_INGREDIENTS
                    )
                st.session_state.last_recs = recs
                st.session_state.messages.append({"role": "assistant", "content": message_bot})
            
            st.rerun()

        # --- ÉTAPE 5 : LE BOUTON DISCRET (TOUT EN BAS) ---
        st.write("") 
        _, btn_col, _ = st.columns([2, 1, 2]) 
        with btn_col:
            if st.button("🗑️ Reset", use_container_width=True, help="Effacer la discussion"):
                reset_gourmi() 
                st.session_state.messages = []
                st.session_state.last_recs = None
                st.rerun()


# =========================
# TAB 4 — COMMUNAUTÉ
# =========================
elif st.session_state["active_tab"] == 3:
    import base64

    # 1. INITIALISATION DE LA MÉMOIRE COMMUNAUTÉ
    if "community_posts" not in st.session_state:
        st.session_state.community_posts = [
            {
                "user": "Marc L.",
                "image": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c",
                "caption": "Incroyable ce que l'IA a proposé avec mon reste de saumon ! 🍣",
                "likes": 24,
                "recipe": "Saumon laqué au soja et miel",
                "comments": [] 
            },
            {
                "user": "Sophie D.",
                "image": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38",
                "caption": "Ma pizza du dimanche soir, validée par les enfants. 🍕",
                "likes": 42,
                "recipe": "Pizza Margherita Express",
                "comments": [] 
            },

            {
                "user": "Merveil D.",
                "image": "https://images.unsplash.com/photo-1671572579845-52270341950f?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                "caption": "Mon wrap fait maison, validée par ma femme. ",
                "likes": 42,
                "recipe": "Wrap soleil",
                "comments": [] 
            }
        ]

    # 2. CSS ÉLÉGANT & PUR
    st.markdown("""
        <style>
        .community-header { text-align: center; padding: 30px 0; background: #ffffff; border-bottom: 1px solid #eee; margin-bottom: 25px; }
        .post-card { background: white; border: 1px solid #e1e1e1; border-radius: 15px; margin-bottom: 35px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
        .user-info { padding: 15px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #fcfcfc; }
        .user-avatar { width: 38px; height: 38px; border-radius: 50%; background: #c8102e; color: white; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.9rem; }
        .post-image { width: 100%; height: 450px; object-fit: cover; }
        .post-content { padding: 18px 22px; font-size: 1.05rem; line-height: 1.5; color: #1d1d1d; }
        .post-stats { padding: 0 22px 20px 22px; color: #888; font-size: 0.85rem; display: flex; gap: 18px; align-items: center; }
        .share-box { background: #fff; border-radius: 15px; padding: 20px; margin-bottom: 40px; }
        
        /* Style pour rendre l'expander visible et pro */
        .stExpander {
            background-color: white !important;
            border: 1px solid #e1e1e1 !important;
            border-radius: 10px !important;
            margin-bottom: 20px !important;
        }

        /* 1. Force le fond de l'expander et de la zone de partage en blanc */
        .stExpander, .share-box, [data-testid="stExpanderDetails"] {
            background-color: white !important;
            color: #1d1d1d !important;
        }

        /* 2. Force le titre de l'expander à être visible (Noir sur Blanc) */
        .stExpander summary {
            background-color: white !important;
            color: #1d1d1d !important;
        }


        /* Cible le rectangle de saisie */
        div[data-baseweb="input"], div[data-baseweb="textarea"] {
            background-color: #D6D6D6!important; /* Force le blanc */
        }

        /* Cible la couche intérieure que Streamlit utilise pour le gris */
        div[data-baseweb="input"] > div {
            background-color: #D6D6D6 !important;
        }
                
        div[data-baseweb="textarea"] > div {
            background-color: #D6D6D6 !important;
        }

        /* On ne force QUE la couleur du texte et du curseur sur la balise de saisie */
        input, textarea {
            color: #1d1d1d !important;
            caret-color: #1d1d1d !important; 
            -webkit-text-fill-color: #1d1d1d !important;
        }

        /* 4. NETTOYAGE RADICAL DE LA ZONE DE TÉLÉCHARGEMENT */

        /* On force le fond de la zone en blanc et on enlève les filtres sombres */
        [data-testid="stFileUploadDropzone"] {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 2px dashed #c8102e !important;
        }
        
        /* Cible le fond du rectangle de téléchargement */
        .stFileUploader section {
            background-color: #D6D6D6 !important;
        }

        /* Cible l'icône de téléchargement pour qu'elle ne soit pas blanche sur blanc */
        [data-testid="stFileUploadDropzone"] svg {
            fill: #c8102e !important;
        }
                
        .stFileUploader button {
        background-color: #c8102e !important; 
        color: #FFFFFF !important; 
        opacity: 1 !important; /* Enlève tout effet de transparence ou voile noir */
       }
 
                
       .stFileUploader button:hover {
        background: #e8e8e8 !important;
        border-color: #c8102e !important;
        color: #c8102e !important;
    }

        /* 5. Force les labels (titres au-dessus des champs) en noir */
        label p {
            color: #1d1d1d !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            font-size: 0.8rem !important;
        }
                
        </style>
    """, unsafe_allow_html=True)

    # 3. HEADER
    st.markdown("""
        <div class="community-header">
            <h1 style="color: #1d1d1d; font-weight: 900; font-size: 2.5rem; margin-bottom:5px;">L'ATELIER GOURMI</h1>
            <p style="color: #666; font-size: 1.1rem; letter-spacing: 0.05em;">VOS CRÉATIONS IA MISES EN LUMIÈRE</p>
        </div>
    """, unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 3.5, 1])

    with center_col:
        # --- FORMULAIRE DE PARTAGE ---
        with st.expander("✨ PARTAGER MA CRÉATION", expanded=False):
            st.markdown('<div class="share-box">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                u_name = st.text_input("Pseudo / Nom", placeholder="Chef...")
            with col2:
                r_name = st.text_input("Nom du plat", placeholder="Saumon croustillant...")
            
            u_caption = st.text_area("Racontez votre expérience...", placeholder="Grâce aux conseils de Gourmi-IA, j'ai réussi la cuisson parfaite !")
            u_file = st.file_uploader("Téléchargez la photo de votre plat", type=["jpg", "png", "jpeg"])
            
            if st.button("PUBLIER SUR L'ATELIER", use_container_width=True):
                if u_name and r_name and u_file:
                    # Conversion de l'image téléchargée en Base64 pour l'afficher
                    img_bytes = u_file.read()
                    encoded_img = base64.b64encode(img_bytes).decode()
                    img_data_url = f"data:image/jpeg;base64,{encoded_img}"
                    
                    # Création du post
                    new_post = {
                        "user": u_name,
                        "image": img_data_url, # C'est l'image de l'utilisateur !
                        "caption": u_caption,
                        "likes": 0,
                        "recipe": r_name,
                        "comments": []
                    }
                    st.session_state.community_posts.insert(0, new_post)
                    st.success("Publication réussie ! Votre plat inspire déjà la communauté.")
                    st.rerun()
                else:
                    st.error("N'oubliez pas votre nom, le titre du plat et la photo !")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- LISTE DES POSTS (CORRECTEUR RADICAL) ---
        for i, post in enumerate(st.session_state.community_posts):
            # ON COULE TOUT LE HTML SANS ESPACES AU DÉBUT DES LIGNES DANS LA CHAÎNE
            post_html = f"""<div class="post-card">
<div class="user-info">
<div class="user-avatar">{post['user'][0].upper()}</div>
<div>
<div style="font-weight: 800; color: #1d1d1d; font-size: 1rem;">{post['user']}</div>
<div style="font-size: 0.8rem; color: #c8102e; font-weight: 700; text-transform: uppercase;">RECETTE : {post['recipe']}</div>
</div>
</div>
<img src="{post['image']}" class="post-image">
<div class="post-content">{post['caption']}</div>
<div class="post-stats">
<span style="font-weight: 700; color: #1d1d1d;">❤️ {post['likes']} J'AIME</span>
</div>
</div>"""
            
                        
            # On affiche
            st.markdown(post_html, unsafe_allow_html=True)
            
            # Bouton de Like
            if st.button(f"Soutenir la création de {post['user']} ✨", key=f"like_btn_{i}", use_container_width=True):
                post['likes'] += 1
                st.rerun()


            
            # ========================================================
            # 3. AJOUTE LE BLOC COMMENTAIRES ICI (JUSTE EN DESSOUS)
            # ========================================================
            comments_list = post.get('comments', [])

            with st.expander(f"💬 Voir les commentaires ({len(comments_list)})"):
                # Affichage des anciens commentaires
                for comment in comments_list:
                    st.markdown(f"**{comment['author']}**: {comment['text']}")
                
                # Formulaire pour en ajouter un nouveau
                with st.form(key=f"comment_form_{i}"):
                    c_user = st.text_input("Ton nom", key=f"c_user_{i}")
                    c_text = st.text_area("Ton commentaire", key=f"c_text_{i}")
                    if st.form_submit_button("Envoyer"):
                        if c_user and c_text:
                            # Si 'comments' n'existe pas encore dans ce post précis, on le crée
                            if 'comments' not in post:
                                post['comments'] = []
                            post['comments'].append({"author": c_user, "text": c_text})
                            st.rerun()
  