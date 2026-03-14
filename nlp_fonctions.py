## Importation des différents Packages

import pandas as pd
import numpy as np
import re
import unicodedata
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import random
import pickle
from pathlib import Path
import json
from openai import OpenAI
import os
from dotenv import load_dotenv


# ============================================
# Installation et chargement du modèle spaCy français
# ============================================

# Initialisation pour que les fonctions du fichier voient cette variable
historique_gourmi = []

# Installation du modèle linguistique français (python -m spacy download fr_core_news_md) (à exécuter une seule fois dans le terminal de l'environnement)


nlp = spacy.load("fr_core_news_md")



# ============================================
# Fonction de normalisation des textes
# ============================================

def normalize_text(text):
    """
    Normalise un texte brut en vue d'un traitement NLP.

    Étapes :
    - passage en minuscules
    - suppression de la ponctuation et des caractères spéciaux
    - suppression des chiffres
    - normalisation des espaces
    """

    # Gestion des valeurs manquantes
    if text is None:
        return ""

    # Passage en minuscules
    text = text.lower()

    # Suppression de la ponctuation et des caractères spéciaux
    # On conserve uniquement les lettres (y compris accentuées) et les espaces
    text = re.sub(r"[^a-zàâäéèêëîïôöùûüç\s]", " ", text)

    #  Suppression des chiffres
    text = re.sub(r"\b\d+\b", " ", text)

    # Réduction des espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================
# Nettoyage des ingrédients par regex
# ============================================


def clean_ingredients_regex(text, ADJECTIFS_SIMPLES):
    """
    Nettoie les listes d'ingrédients afin de conserver principalement
    les termes porteurs de sens (ingrédients).

    Étapes :
    - normalisation typographique
    - suppression des quantités et unités de mesure
    - suppression des adjectifs simples peu discriminants
    - suppression du bruit typographique
    """

    # Gestion des valeurs manquantes
    if pd.isna(text):
        return ""

    # Passage en minuscules
    text = text.lower()

    # Normalisation des apostrophes typographiques
    text = text.replace("’", "'")

    # Suppression des quantités avec unités (ex : 480g, 200 ml, 2kg)
    text = re.sub(r"\b\d+\s*(?:g|kg|mg|ml|cl|l)\b", " ", text)

    # Suppression des chiffres collés aux mots (ex : 2cuillères → cuillères)
    text = re.sub(r"\b\d+(?=[a-zàâäéèêëîïôöùûüç])", " ", text)

    # Suppression des nombres isolés (ex : 6 oeufs → oeufs)
    text = re.sub(r"\b\d+\b", " ", text)

    # Suppression des unités de mesure restantes
    units = r"\b(g|kg|mg|ml|cl|l|càs|cas|càc|cac|cuillères?|cuillere|grammes?)\b"
    text = re.sub(units, " ", text)

    # Suppression des adjectifs simples peu informatifs
    pattern_adj = r"\b(" + "|".join(ADJECTIFS_SIMPLES) + r")\b"
    text = re.sub(pattern_adj, " ", text)

    # Suppression des caractères spéciaux et ponctuations restantes
    text = re.sub(r"[^a-zàâäéèêëîïôöùûüç\s]", " ", text)

    # Normalisation des espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text



# ============================================
# Définition des stopwords
# ============================================

# Stopwords fournis par spaCy (français)
stopwords_spacy = nlp.Defaults.stop_words

# Stopwords supplémentaires spécifiques au corpus
stopwords_extra = {"d", "l"}  # ex : "d'", "l'"

# ============================================
# Fonction de tokenisation
# ============================================

def tokenize_text(text):
    """
    Tokenise un texte à l'aide de spaCy et filtre les tokens non pertinents.

    Critères de conservation :
    - token non vide
    - longueur > 2 caractères
    - non présent dans les stopwords
    """

    # Gestion des textes vides ou manquants
    if not text:
        return []

    # Analyse linguistique avec spaCy
    doc = nlp(text)

    # Sélection des tokens pertinents
    tokens = [
        token.text.strip()
        for token in doc
        if token.text.strip()                         # token non vide
        and len(token.text.strip()) > 2               # longueur minimale
        and token.text.strip() not in stopwords_spacy
        and token.text.strip() not in stopwords_extra
    ]

    return tokens



# ============================================
# POS-tagging des ingrédients avec spaCy
# ============================================

def spacy_pos_tag(token_list, nlp):
    """
    Applique le POS-tagging spaCy à une liste de tokens.

    Retour :
    - liste de tuples (token, catégorie grammaticale)
    """

    # Reconstruction du texte à partir des tokens
    text = " ".join(token_list)

    # Analyse linguistique avec spaCy
    doc = nlp(text)

    # Extraction du token et de sa catégorie grammaticale
    pos_tags = [(token.text, token.pos_) for token in doc]

    return pos_tags




# ============================================
# Règles de correction du POS-tagging (domaine culinaire)
# ============================================

# Identification des ingrédients fréquents


def correct_pos_for_ingredients(
    token,
    pos,
    frequent_terms=None,
    forced_ingredients=None,
    suffixes=None
):
    """
    Corrige la catégorie grammaticale d'un token
    à l'aide de règles spécifiques au domaine culinaire.

    Paramètres :
    - token : forme textuelle du token
    - pos : POS original (spaCy)
    - frequent_terms : ensemble de termes fréquents (optionnel)
    - forced_ingredients : ensemble d'ingrédients forcés comme noms
    - suffixes : suffixes morphologiques caractéristiques
    """

    t = token.strip().lower()

    if forced_ingredients and t in forced_ingredients:
        return "NOUN"

    if frequent_terms and t in frequent_terms:
        return "NOUN"

    if suffixes and t.endswith(tuple(suffixes)):
        return "NOUN"

    return pos


# ============================================
# POS-tagging avec correction manuelle (spaCy)
# ============================================

def spacy_pos_tag(
    token_list,
    nlp,
    frequent_terms=None,
    forced_ingredients=None,
    suffixes=None
):
    """
    Applique le POS-tagging spaCy à une liste de tokens
    puis corrige les catégories grammaticales à l'aide
    de règles spécifiques au domaine culinaire.

    Retour :
    - liste de tuples (token, POS_corrigé)
    """

    text = " ".join(token_list)
    doc = nlp(text)

    corrected_pos = []

    for token in doc:
        corrected = correct_pos_for_ingredients(
            token=token.text,
            pos=token.pos_,
            frequent_terms=frequent_terms,
            forced_ingredients=forced_ingredients,
            suffixes=suffixes
        )
        corrected_pos.append((token.text, corrected))

    return corrected_pos


# ============================================
# Lemmatisation
# ============================================

def lemmatize_tokens(token_list):
    """
    Applique la lemmatisation spaCy à une liste de tokens.

    Retour :
    - liste de lemmes normalisés (minuscules)
    """

    # Gestion des cas vides ou incorrects
    if not token_list or not isinstance(token_list, list):
        return []

    # Reconstruction du texte pour l'analyse spaCy
    text = " ".join(token_list)
    doc = nlp(text)

    # Extraction des lemmes
    lemmes = [
        token.lemma_.lower().strip()
        for token in doc
        if not token.is_space
    ]

    return lemmes


# ============================================
# Extraction finale des ingrédients
# ============================================

def extract_ingredients_final_row(row, NON_ING, BRANDS, ADJ_EXCLUDE, INGREDIENTS_FORCES, ROOTS):
    """
    Extrait une liste finale d'ingrédients à partir d'une recette,
    en combinant POS-tagging corrigé et règles métier.
    """

    tokens = row["ingredients_tokens"]
    pos_list = [pos for _, pos in row["ingredients_pos"]]

    ingredients = []

    for token, pos in zip(tokens, pos_list):
        t = token.lower().strip()

        # Exclusions directes (contenants, marques, adjectifs)
        if t in NON_ING or t in BRANDS:
            continue

        # Exclure adjectifs (singulier + pluriel)
        if t in ADJ_EXCLUDE or re.sub(r"s$", "", t) in ADJ_EXCLUDE:
            continue

        # Participes passés / formes inutiles
        if re.search(r"(ées|és|ée|é)$", t):
            continue

        # Ingrédients essentiels forcés
        if t in INGREDIENTS_FORCES:
            ingredients.append(t)
            continue

        # Racines utiles (ex : tomat* → tomate)
        if any(t.startswith(root) for root in ROOTS):
            ingredients.append(t)
            continue

        # Sélection finale : noms uniquement
        if pos == "NOUN":
            ingredients.append(t)

    # Suppression des doublons en conservant l'ordre
    return list(dict.fromkeys(ingredients))



# ============================================
# Fonctions de nettoyage des ingrédients
# ============================================

def normaliser_singulier(mot, vocab):
    """
    Ramène un mot au singulier s'il existe dans le vocabulaire.
    """
    if mot.endswith("s") and mot[:-1] in vocab:
        return mot[:-1]
    return mot


def nettoyer_ingredient(mot, vocab, faux_ing, ingredients_always):
    """
    Nettoie un ingrédient individuel à l'aide de règles métier.
    """

    m = mot.lower().strip()

    # Exclusion forte
    if m in faux_ing:
        return None

    # Ingrédients protégés
    if m in ingredients_always:
        return m

    # Normalisation pluriel -> singulier si valide
    m = normaliser_singulier(m, vocab)

    return m


def nettoyer_liste_ingredients(liste, vocab, faux_ing, ingredients_always):
    """
    Nettoie une liste d'ingrédients :
    - exclusion des faux ingrédients
    - normalisation morphologique
    - suppression des doublons
    """

    cleaned = []

    for mot in liste:
        nm = nettoyer_ingredient(mot, vocab, faux_ing, ingredients_always)
        if nm:
            cleaned.append(nm)

    # Suppression des doublons en conservant l'ordre
    return list(dict.fromkeys(cleaned))



# ============================================
# Fonction de vectorisation TF-IDF des ingrédients
# ============================================

def tfidf_ingredients(
    df,
    ingredients_col="ingredients_clean_final",
    text_col=None,
    ngram_type="unigram",
    min_df=15,
    max_df=0.9
):
    """
    Construit une représentation TF-IDF des recettes.

    Deux modes possibles :
    - à partir d'une colonne contenant une liste d'ingrédients (ingredients_col)
    - à partir d'une colonne texte déjà construite (text_col)
    """

    # Choix du type de n-gram
    if ngram_type == "unigram":
        ngram_range = (1, 1)
    elif ngram_type == "bigram":
        ngram_range = (2, 2)
    elif ngram_type == "both":
        ngram_range = (1, 2)
    else:
        raise ValueError("ngram_type doit être 'unigram', 'bigram' ou 'both'")

    # ============================================
    # Construction du corpus textuel
    # ============================================

    # Cas 1 : une colonne texte est déjà disponible
    if text_col is not None:
        text_data = df[text_col]

    # Cas 2 : conversion d'une liste d'ingrédients en texte
    else:
        df["ingredients_str"] = df[ingredients_col].apply(
            lambda lst: " ".join(lst) if isinstance(lst, list) else ""
        )
        text_data = df["ingredients_str"]

    # ============================================
    # Vectorisation TF-IDF
    # ============================================

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )

    tfidf_matrix = vectorizer.fit_transform(text_data)


    return tfidf_matrix, vectorizer


# ============================================
# Analyse des ingrédients dominants (TF-IDF)
# ============================================

def top_ingredients(tfidf_df, recipe_index, top_k=10):
    """
    Retourne les top_k ingrédients les plus importants (TF-IDF)
    pour une recette donnée.
    """
    row = tfidf_df.loc[recipe_index]
    top = row.sort_values(ascending=False).head(top_k)
    return top[top > 0]  # suppression des valeurs nulles


# ============================================
# Nettoyage des titres pour le clustering (KMeans)
# ============================================


def clean_title(text, BRANDS,TITLE_STOPWORDS ):
    """
    Nettoie un titre de recette afin de réduire le bruit
    pour la classification non supervisée.
    """

    if pd.isna(text):
        return ""

    text = text.lower()

    # Suppression des marques
    pattern_brands = r"\b(" + "|".join(BRANDS) + r")\b"
    text = re.sub(pattern_brands, " ", text)

    # Suppression des mots génériques
    pattern_sw = r"\b(" + "|".join(TITLE_STOPWORDS) + r")\b"
    text = re.sub(pattern_sw, " ", text)

    # Suppression des chiffres
    text = re.sub(r"\b\d+\b", " ", text)

    # Suppression de la ponctuation et caractères spéciaux
    text = re.sub(r"[^a-zA-Zàâäéèêëîïôöùûüç\s]", " ", text)

    # Normalisation des espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text



# ============================================
# Analyse des mots dominants par cluster (KMeans)
# ============================================

def top_terms_cluster(centroids, feature_names, cluster_id, n=15):
    """
    Retourne les n termes les plus représentatifs
    d'un cluster à partir de son centroïde.

    Paramètres :
    - centroids : array des centroïdes KMeans
    - feature_names : vocabulaire TF-IDF
    - cluster_id : identifiant du cluster
    - n : nombre de termes à retourner
    """

    top_idx = centroids[cluster_id].argsort()[::-1][:n]
    return [feature_names[i] for i in top_idx]





# ============================================
# Nettoyage de la requête de l'utilisateur
# ============================================

def clean_user_ingredients(user_ingredients):
    """
    Nettoie et normalise la liste d'ingrédients fournie par l'utilisateur.
    """
    if not user_ingredients:
        return []

    return [
        ing.lower().strip()
        for ing in user_ingredients
        if isinstance(ing, str) and ing.strip()
    ]


# ============================================
# Vectorisation de la requête utilisateur
# ============================================

def vectorize_user_text(user_ingredients, vectorizer):
    """
    Vectorise la liste d'ingrédients utilisateur
    dans l'espace TF-IDF fourni.
    """
    if not user_ingredients:
        return None

    text = " ".join(user_ingredients)
    return vectorizer.transform([text])


# ============================
# Filtrage par ingrédients
# ============================

def count_common_ingredients(user_ingredients, recipe_ingredients):
    """
    Calcule le nombre d'ingrédients communs entre la liste fournie par l'utilisateur
    et les ingrédients d'une recette donnée. La comparaison est indépendante de
    l'ordre des ingrédients et retourne 0 si la recette ne contient pas de liste valide.
    """
    if not isinstance(recipe_ingredients, list):
        return 0

    return len(set(user_ingredients).intersection(recipe_ingredients))



def recommend_recipes(user_ingredients, tfidf_matrix, vectorizer, df, top_k=5, min_common_override=None):
    """
    Version corrigée : Priorise le respect de TOUS les ingrédients demandés.
    """
    if not user_ingredients:
        return pd.DataFrame()

    # --- MODIFICATION ICI ---
    # Si l'utilisateur donne des ingrédients, on veut qu'ils soient TOUS présents
    # Sauf si tu as forcé un chiffre spécifique via l'override
    if min_common_override is not None:
        min_common = min_common_override
    else:
        # On demande que TOUS les ingrédients de la requête soient présents
        min_common = len(user_ingredients) 
    # --------------------------

    # Vectorisation de la requête
    user_vector = vectorize_user_text(user_ingredients, vectorizer)
    
    # Calcul des similarités
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Filtrage par intersection réelle
    def count_common(recipe_ing):
        # On vérifie combien d'ingrédients de l'utilisateur sont dans la recette
        return len(set(user_ingredients) & set(recipe_ing))

    temp_df = df.copy()
    temp_df['score_tfidf'] = similarities
    temp_df['n_common'] = temp_df['ingredients_clean_final'].apply(count_common)
    
    # On filtre : la recette doit avoir AU MOINS le nombre d'ingrédients demandés
    results = temp_df[temp_df['n_common'] >= min_common]
    
    # Si après ce filtrage strict il n'y a rien, on peut éventuellement 
    # redescendre à min_common - 1 (optionnel), mais pour ton cas Wrap/Poulet, 
    # rester strict est la solution.
    
    return results.sort_values(by='score_tfidf', ascending=False).head(top_k)


# ============================================
# Détermination du cluster utilisateur
# ============================================

def get_user_cluster(user_ingredients, vectorizerk, kmeans):
    """
    Associe l'utilisateur à un cluster de recettes
    à partir de ses ingrédients.
    """
    # Nettoyage centralisé
    user_ingredients = clean_user_ingredients(user_ingredients)

    if not user_ingredients:
        return None

    # Vectorisation dans l'espace KMeans
    user_vec = vectorize_user_text(user_ingredients, vectorizerk)

    # Prédiction du cluster
    return kmeans.predict(user_vec)[0]



# ============================================
# Recommandation par catégorie / cluster
# ============================================

def normalize_text_strict(text):
    """
    Normalisation STRICTE pour comparaison texte :
    - minuscules
    - suppression des accents
    - suppression ponctuation
    - espaces normalisés
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Suppression des accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # Suppression caractères non alphabétiques
    text = re.sub(r"[^a-z\s]", " ", text)

    # Espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def recommend_by_cluster(user_text, df, top_k=5):
    """
    Recommandation par catégorie (cluster).
    - Insensible à la casse
    - Insensible aux accents
    - Basée sur cluster_name_norm (déjà calculé)
    """

    # Sécurité : Si le texte est vide ou non valide
    if not isinstance(user_text, str) or not user_text.strip():
        return pd.DataFrame() # Retourne un DataFrame vide p

    # Normalisation
    query_norm = normalize_text_strict(user_text)

    # 3. Filtrage
    results = df[df["cluster_name_norm"] == query_norm]

    # Retour sécurisé
    if results is not None and not results.empty:
        return results.head(top_k)
    
    return pd.DataFrame() # Toujours retourner un DataFrame vide si rien n'est trouvé



# ============================================
# Recommandation pondérée (similarité + cluster)
# ============================================

def recommend_weighted(
    user_ingredients,
    df,
    X_kmeans,
    vectorizerk,
    kmeans,
    alpha=0.75,
    top_k=5
):
    # ============================
    # 1) Sécurisation entrée
    # ============================
    if not user_ingredients:
        return pd.DataFrame()

    user_ingredients = clean_user_ingredients(user_ingredients)

    # ============================
    # Filtrage logique par ingrédients
    # ============================
    results = df.copy()

    results["n_common"] = results["ingredients_clean_final"].apply(
        lambda x: count_common_ingredients(user_ingredients, x)
    )

    min_common = max(2, int(np.ceil(len(user_ingredients) / 2)))
    results = results[results["n_common"] >= min_common]

    if results.empty:
        return pd.DataFrame()

    # ============================
    # Similarité TF-IDF
    # ============================
    user_vec = vectorizerk.transform([" ".join(user_ingredients)])

    results["cosine_similarity"] = cosine_similarity(
        user_vec,
        X_kmeans[results.index]
    ).ravel()

    # ============================
    # Bonus cluster
    # ============================
    user_cluster = kmeans.predict(user_vec)[0]

    results["cluster_score"] = (
        results["cluster"].values == user_cluster
    ).astype(int)

    # ============================
    # Score final pondéré
    # ============================
    results["final_score"] = (
        alpha * results["cosine_similarity"]
        + (1 - alpha) * results["cluster_score"]
    )

    return (
        results
        .sort_values(
            by=["final_score", "cosine_similarity"],
            ascending=[False, False]
        )
        .head(top_k)
    )


# ============================
# evaluation des systèmes de recommendations
# ============================

def evaluate_recommender(
    df,
    recommend_func,
    top_k=5,
    n_samples=1000,
    keep_ratio=0.6,
    random_state=123,
    **recommend_kwargs
):
    """
    Évaluation OFFLINE d’un système de recommandation non supervisé
    via simulation utilisateur (ranking Top-K).

    Métriques :
    - Recall@K (Hit Rate)
    - MRR (Mean Reciprocal Rank)
    """

    random.seed(random_state)

    hits = []     # pour Recall@K
    rranks = []   # pour MRR

    # ============================
    # Échantillonnage recettes
    # ============================
    sampled_df = df.sample(n=n_samples, random_state=random_state)

    # ============================
    #  Boucle de simulation
    # ============================
    for idx, row in sampled_df.iterrows():

        ingredients = row["ingredients_clean_final"]

        # Sécurité
        if not isinstance(ingredients, list) or len(ingredients) < 4:
            continue

        # ============================
        # Simulation utilisateur
        # ============================
        n_keep = max(2, int(len(ingredients) * keep_ratio))
        user_ingredients = random.sample(ingredients, n_keep)

        # ============================
        # Recommandation
        # ============================
        recs = recommend_func(
            user_ingredients=user_ingredients,
            top_k=top_k,
            **recommend_kwargs
        )

        
        if recs is None or recs.empty:
            hits.append(0)
            rranks.append(0)
            continue

        titles = list(recs["titre"])

        # ============================
        # Recall@K (Hit Rate)
        # ============================
        if row["titre"] in titles:
            hits.append(1)

            # ============================
            # MRR
            # ============================
            rank = titles.index(row["titre"]) + 1
            rranks.append(1 / rank)
        else:
            hits.append(0)
            rranks.append(0)

    # ============================
    # 7) Résultats finaux
    # ============================
    return {
        "Recall@K": np.mean(hits),
        "MRR": np.mean(rranks)
    }




# ============================================
# Extraction simple des ingrédients depuis une phrase utilisateur
# ============================================

def extract_ingredients_from_query(user_text, vocab_ingredients):
    """
    Extrait les ingrédients en utilisant normalize_text_strict pour 
    être insensible aux accents et à la casse.
    """
    if not isinstance(user_text, str):
        return []

    # Creation d'un dictionnaire de correspondance
    
    vocab_map = {normalize_text_strict(v): v for v in vocab_ingredients}

    # Normalisation sur le texte utilisateur complet
    
    clean_query = normalize_text_strict(user_text)

    # Decoupage en mots simples
    tokens = clean_query.split()

    found = []
    for tok in tokens:
 
        tok_norm = normaliser_singulier(tok, vocab_map.keys())

        if tok_norm in vocab_map:
 
            found.append(vocab_map[tok_norm])

    return list(dict.fromkeys(found))




# Formatage robuste des valeurs numériques (NaN safe)

def safe_display(value, suffix=""):
    """
    Affiche une valeur ou 'Non renseigné' si absente ou NaN.
    """
    if value is None or pd.isna(value):
        return "Non renseigné"
    try:
        return f"{int(value)}{suffix}"
    except:
        return str(value)




# ============================
# Analyse Sémantique (Vecteurs spaCy)
# ============================

def get_semantic_vector(text_list):
    """Calcule le vecteur moyen de sens pour une liste d'ingrédients via spaCy."""
    import numpy as np
    # On vérifie que c'est bien une liste (tes ingrédients clean sont stockés en listes)
    if not isinstance(text_list, list) or not text_list:
        return np.zeros(300)
   
    # On transforme la liste en chaîne pour spaCy
    doc = nlp(" ".join(text_list))
   
    # On retourne le vecteur numérique du document (moyenne des mots)
    return doc.vector if doc.vector_norm else np.zeros(300)


# ============================
# Mise en Forme de la Réponse Utilisateur
# ============================


def format_reponse_gourmi(final_recs, ingredients_query):
    """
    Transforme le DataFrame des résultats en un message élégant pour l'utilisateur.
    """
    if final_recs is None or final_recs.empty:
        return "❌ Désolé, je n'ai trouvé aucune recette correspondant à votre demande."

    # En-tête du message
    phrase_ing = ", ".join(ingredients_query)
    msg = f"**Assistant Gourmi** : Voici ma sélection pour **{phrase_ing}**\n"
    msg += "---" * 10 + "\n\n"

    # Boucle sur les recettes trouvées
    for i, (idx, row) in enumerate(final_recs.iterrows(), 1):
        # On récupère les infos en gérant les valeurs manquantes
        titre = row['titre'].upper()
        temps = int(row.get('temps_total', 0))
        cout = str(row.get('cout', 'N/C')).capitalize()
        diff = str(row.get('difficulte', 'N/C')).capitalize()
       
        # On ajoute une petite icône selon le score
        msg += f"{i}. **{titre}**\n"
        msg += f"   ⏱️ {temps} min | 💰 {cout} | ⚙️ {diff}\n\n"

    msg += "---" * 10 + "\n"
    msg += "👉 **Tapez un numéro** pour découvrir les secrets de la recette qui vous interesse ! \n\n"
    msg += "Besoin d'ajuster ? Donnez-moi vos préférences (ingrédients, temps...). Ou, pour explorer un tout autre horizon culinaire, **réinitialisez la discussion** et on repart de zéro !"
    return msg


# ============================
# Moteur de Recherche Hybride (TF-IDF + Sémantique + Filtres)
# ============================

def gourmi_brain_hybrid_v3(user_text, df_nlp, df_chatbot, X_both, vec_both, vocab_ingredients,
                          max_time_llm=None, budget_llm=None, ingredients_llm=None,
                          excluded_ingredients=None, dish_type=None, cluster_name=None, top_k=5):
   
    # On commence avec tout le monde
    candidates = df_nlp.copy()
    
    # --- ÉTAPE 1 : FILTRE DE STRUCTURE (L'ENTONNOIR) ---
    
    # Si on a un cluster (ex: Méditerranéen)
    if cluster_name:
        c_norm = normalize_text_strict(cluster_name)
        # On utilise ta colonne cluster_name_norm
        subset = candidates[candidates["cluster_name_norm"].str.contains(c_norm, na=False)]
        if not subset.empty:
            candidates = subset

    # Si on a un type de plat (ex: Pizza)
    if dish_type:
        d_norm = normalize_text_strict(dish_type)
        # On cherche dans le titre
        subset = candidates[candidates['titre'].apply(lambda x: d_norm in normalize_text_strict(str(x)))]
        if not subset.empty:
            candidates = subset

    # --- ÉTAPE 2 : RECHERCHE TF-IDF SUR LE RÉSULTAT ---
    ingredients_query = ingredients_llm if ingredients_llm else extract_ingredients_from_query(user_text, vocab_ingredients)

    if not ingredients_query and not dish_type and not cluster_name:
        return None, "❌ Je n'ai pas compris votre recherche."

    if ingredients_query:
        # On récupère les lignes de la matrice correspondant à nos candidats restants
        indices = candidates.index.tolist()
        
        # On relance recommend_recipes sur notre sélection
        res_recs = recommend_recipes(ingredients_query, X_both[indices], vec_both, candidates, top_k=50, min_common_override=1)
        if not res_recs.empty:
            candidates = res_recs
        else:
            candidates['score_tfidf'] = 0.01
    else:
        candidates['score_tfidf'] = 0.1

    # --- ÉTAPE 3 : MERGE ET FILTRES SANITAIRES ---
    candidates = candidates.merge(df_chatbot, on="titre", how="left", suffixes=('', '_drop'))
    candidates = candidates.loc[:, ~candidates.columns.str.contains('_drop')]

    # Filtre Temps
    candidates['temps_total'] = pd.to_numeric(candidates['temps_total'], errors='coerce').fillna(999)
    if max_time_llm:
        candidates = candidates[candidates['temps_total'] <= int(max_time_llm)]

    # Filtre Exclusion
    if excluded_ingredients:
        for ing in excluded_ingredients:
            candidates = candidates[~candidates['ingredients_clean_final'].astype(str).str.contains(ing.lower())]

    # --- ÉTAPE 4 : SCORE SÉMANTIQUE ---
    if not candidates.empty and ingredients_query:
        q_vec = get_semantic_vector(ingredients_query).reshape(1, -1)
        r_vecs = np.array([get_semantic_vector(ing) for ing in candidates['ingredients_clean_final']])
        candidates['semantic_score'] = cosine_similarity(q_vec, r_vecs).flatten()
        candidates['final_score'] = (candidates['score_tfidf'] * 0.4) + (candidates['semantic_score'] * 0.6)
    else:
        candidates['final_score'] = candidates.get('score_tfidf', 0.1)

    # --- ÉTAPE 5 : TRI FINAL ---
    if candidates.empty:
        return None, "⚠️ Aucune recette trouvée avec ces filtres."

    final_recs = candidates.sort_values('final_score', ascending=False).head(top_k)
    return final_recs, format_reponse_gourmi(final_recs, ingredients_query if ingredients_query else [user_text])

# ============================
# Configuration et Sécurité
# ============================

# Charge les variables d'environnement du fichier .env
load_dotenv()

# Récupère la clé et initialise le client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# ============================
# Agent Principal (Cerveau LLM & Gestion d'Intention)
# ============================


def agent_gourmi_master(user_query, df_nlp, df_chatbot, X_both, vec_both, vocab_ingredients):
    # --- ASTUCE DE COMPATIBILITÉ NOTEBOOK / STREAMLIT ---
    try:
        import streamlit as st
        # Si on est dans Streamlit, on utilise st.session_state
        if "historique_gourmi" not in st.session_state:
            st.session_state.historique_gourmi = []
        memoire = st.session_state.historique_gourmi
    except ImportError:
        # Si on est dans un Notebook, on utilise la variable globale
        global historique_gourmi
        memoire = historique_gourmi

    # 1. Mise à jour de l'historique (utilise 'memoire' qui pointe vers le bon endroit)
    memoire.append({"role": "user", "content": user_query})


    system_instruction = """
    Tu es Gourmi-IA, un expert culinaire. Tu dois extraire les intentions de l'utilisateur.
    Réponds EXCLUSIVEMENT en JSON.

    RÈGLES D'EXTRACTION DES INGRÉDIENTS :
    1. N'extrais QUE des noms d'aliments (chocolat, poulet, tomate).
    2. Ignore les mots de liaison, verbes ou adverbes (base, aimerait, possible, fait, mai).
    3. Si l'utilisateur dit 'sans sucre', n'ajoute pas 'sucre' dans les ingrédients, car le moteur va le chercher. Laisse la liste vide ou mets les ingrédients positifs uniquement.
    4. Si tu discute avec l'utilisateur et qu'il te demande de lui proposer des recette et e donne les ingrédients, utilise toujours le systeme de recommandation pour lui proposer des recettes ne les énumère jamais toi même

    RÈGLES DE FILTRAGE :
    1. 'ingredients' : Liste uniquement les aliments que l'utilisateur VEUT inclure.
    2. 'excluded_ingredients' : Si l'utilisateur dit 'sans X', 'pas de X', ou 'allergique à X', place 'X' dans cette liste.
    3. 'max_time' : Uniquement le nombre entier (ex: 30).
    4. 'budget' : 'bon marché' si l'utilisateur veut du pas cher.

    RÈGLES DE MÉMOIRE :
    - Si l'utilisateur change de critère (ex: 'plus rapide'), garde les ingrédients du message précédent en mémoire dans ta réponse JSON.
    - Conserve les ingrédients des messages précédents sauf si l'utilisateur change radicalement de sujet.

    Structure du JSON :
    {
    "intent": "chat" ou "search",
    "message_prive": "Ta réponse amicale",
    "filters": {
        "ingredients": [],
        "excluded_ingredients": [],
        "dish_type": "ex: wrap, pizza, quiche ou null",
        "cluster": "ex: italien, dessert ou null",
        "max_time": null,
        "budget": null
    }
    }
    """
   
    # On prépare les messages (système + les 6 derniers pour le contexte)
    messages_to_send = [{"role": "system", "content": system_instruction}] + memoire[-10:]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_to_send,
            response_format={ "type": "json_object" }
        )
        analysis = json.loads(response.choices[0].message.content)
    except Exception as e:
        return None, f"❌ Erreur de communication avec le cerveau : {e}"

    # 2. On enregistre la réponse de l'assistant dans l'historique
    historique_gourmi.append({"role": "assistant", "content": analysis.get('message_prive', '')})

    # 3. LOGIQUE DE SORTIE SÉCURISÉE
    if analysis.get('intent') == "chat":
        return None, analysis.get('message_prive', "Je suis à votre écoute !")
   
    # ... (Début de la fonction identique jusqu'à l'extraction du JSON)

    # Si c'est une recherche
    params = analysis.get('filters', {})
   
    # CORRECTIF ICI : On passe bien TOUS les paramètres, surtout excluded_ingredients
    recs, message = gourmi_brain_hybrid_v3(
        user_query,
        df_nlp,
        df_chatbot,
        X_both,
        vec_both,
        vocab_ingredients,
        max_time_llm=params.get('max_time'),
        budget_llm=params.get('budget'),
        ingredients_llm=params.get('ingredients', []),
        excluded_ingredients=params.get('excluded_ingredients', []), # <-- NE PAS OUBLIER ÇA,
        dish_type=params.get('dish_type'),  # NOUVEAU
        cluster_name=params.get('cluster')   # NOUVEAU
    )
   
    return recs, message




# ============================
# Gestion de la Mémoire de Session
# ============================

def reset_gourmi():
    """Efface l'historique de manière compatible Streamlit et Notebook."""
    try:
        import streamlit as st
        # Si on est dans Streamlit, on vide le session_state
        if "historique_gourmi" in st.session_state:
            st.session_state.historique_gourmi = []
        # On vide aussi les messages de l'interface si tu les utilises
        if "messages" in st.session_state:
            st.session_state.messages = []
           
    except ImportError:
        # Si on est dans un Notebook, on vide la variable globale
        global historique_gourmi
        historique_gourmi = []
       
    print("Mémoire de l'assistant effacée avec succès !")
    return "Mémoire effacée"


# ============================
# TEST — Étape 2 : détail d’une recette
# ============================


def culinary_chatbot_step2(recommendations, choice):
    # ============================
    # Sécurisation du choix
    # ============================
    try:
        idx = int(choice) - 1
        recipe = recommendations.iloc[idx]
    except Exception:
        return "❌ Choix invalide. Merci d’indiquer un numéro valide."

    # ============================
    # Titre
    # ============================
    response = f"🍽️ **{recipe.get('titre', 'Titre non renseigné')}**\n\n"

    # ============================
    # Ingrédients
    # ============================
    response += "🧾 Ingrédients :\n"

    # Sélection de la bonne colonne selon le mode
    if "ingredients_clean_final_y" in recipe:
        ingredients = recipe["ingredients_clean_final_y"]
    elif "ingredients_clean_final_x" in recipe:
        ingredients = recipe["ingredients_clean_final_x"]
    else:
        ingredients = recipe.get("ingredients_clean_final")

    # Affichage propre
    if isinstance(ingredients, list) and ingredients:
        for ing in ingredients:
            response += f"- {ing}\n"
    else:
        response += "- Non renseignés\n"

    # ============================
    # Étapes de préparation
    # ============================
    response += "\n👨‍🍳 Étapes de préparation :\n"

    # Sélection de la bonne colonne selon le mode
    if "etapes_y" in recipe:
        etapes_raw = recipe["etapes_y"]
    elif "etapes_x" in recipe:
        etapes_raw = recipe["etapes_x"]
    else:
        etapes_raw = recipe["etapes"]

    # Conversion string -> liste Python
    try:
        etapes_list = ast.literal_eval(etapes_raw)
        for i, step in enumerate(etapes_list, start=1):
            response += f"{i}. {step.strip()}\n"
    except:
        response += "- Étapes non disponibles.\n"

    # ============================
    # Temps (MODIFICATION ICI POUR LE 999)
    # ============================
    
    # Petite fonction interne pour intercepter le 999 uniquement pour l'affichage
    def clean_999(val, suffix=""):
        if pd.isna(val) or str(val) == "999" or str(val) == "999.0":
            return "Non renseigné"
        return safe_display(val, suffix)

    response += "\n⏱️ Temps :\n"
    response += f"- Préparation : {clean_999(recipe.get('temps_preparation'), ' min')}\n"
    response += f"- Cuisson : {clean_999(recipe.get('temps_cuisson'), ' min')}\n"
    response += f"- Total : {clean_999(recipe.get('temps_total'), ' min')}\n"

    # ============================
    # Infos complémentaires
    # ============================
    response += f"\n👥 Personnes : {safe_display(recipe.get('personnes'))}"
    response += f"\n⚙️ Difficulté : {safe_display(recipe.get('difficulte'))}"

    if recipe.get("image"):
        response += f"\n🖼️ Image : {recipe.get('image')}"

    if recipe.get("url"):
        response += f"\n🔗 Recette originale : {recipe.get('url')}"

    return response

# ============================
# Fonction de chargement des données pour l'application
# ============================

def load_models():
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"

    with open(MODELS_DIR / "vec_both.pkl", "rb") as f:
        vec_both = pickle.load(f)

    with open(MODELS_DIR / "tfidf_matrix.pkl", "rb") as f:
        tfidf_both = pickle.load(f)

    with open(MODELS_DIR / "df.pkl", "rb") as f:
        df = pickle.load(f)

    with open(MODELS_DIR / "df_chatbot.pkl", "rb") as f:
        df_chatbot = pickle.load(f)

    with open(MODELS_DIR / "vocab_ingredients.pkl", "rb") as f:
        VOCAB_INGREDIENTS = pickle.load(f)

    return df, df_chatbot, VOCAB_INGREDIENTS, tfidf_both, vec_both






















