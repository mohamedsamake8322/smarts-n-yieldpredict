class BilanPrevisionnel:
    def __init__(self, stock_initial, autres_donnees_avant, stock_final_souhaite, autres_donnees_pendant):
        self.stock_initial = stock_initial  # kg/ha ou mm selon le type de bilan
        self.autres_donnees_avant = autres_donnees_avant  # dict: {'type': valeur}
        self.stock_final_souhaite = stock_final_souhaite
        self.autres_donnees_pendant = autres_donnees_pendant  # dict: {'type': valeur}

    def besoins_culture(self, rendement_objectif, climat, sol, pratiques):
        """
        Estimation des besoins en fonction de plusieurs facteurs.
        Tous les paramètres sont des dicts ou scalaires selon le modèle.
        """
        besoin_eau = self._estimer_eau(rendement_objectif, climat, sol)
        besoin_nutriments = self._estimer_nutriments(rendement_objectif, sol, pratiques)
        return {
            'eau': besoin_eau,
            'nutriments': besoin_nutriments
        }

    def _estimer_eau(self, rendement, climat, sol):
        # Exemple simplifié : ETc estimée selon rendement et climat
        facteur_climatique = climat.get('ET0', 0) * climat.get('Kc', 1)
        facteur_sol = sol.get('capacite_retenue', 1)
        return rendement * facteur_climatique * facteur_sol

    def _estimer_nutriments(self, rendement, sol, pratiques):
        # Exemple simplifié : azote estimé selon rendement et pratiques
        besoin_N = rendement * 3.5  # kg N/ha par quintal
        correction_sol = sol.get('richesse_N', 0)
        correction_pratiques = pratiques.get('effluents', 0)
        return besoin_N - correction_sol - correction_pratiques

    def bilan(self):
        """
        Calcul du bilan global : Entrées - Sorties
        """
        entrees = self.stock_initial + sum(self.autres_donnees_avant.values())
        sorties = self.stock_final_souhaite + sum(self.autres_donnees_pendant.values())
        return entrees - sorties

#.🧮 Formule principale
def stock_final(Pf, Rf):
    """
    Évaluation du stock final : Pf + Rf
    Pf = besoins de la plante
    Rf = azote restant dans le sol après absorption
    """
    return Pf + Rf

#🌾 Besoins Pf selon type de culture
def besoins_plante(rendement_objectif, coefficient_b):
    """
    Besoins dépendants du niveau de production : Pf = rendement x b
    coefficient_b = kg N / unité de rendement (ex: quintal, tonne)
    """
    return rendement_objectif * coefficient_b

def besoins_forfaitaires(valeur_forfaitaire):
    """
    Besoins forfaitaires indépendants du rendement
    """
    return valeur_forfaitaire

#📊 Calcul du rendement objectif (sans historique)
def rendement_objectif(historique_rendements):
    """
    Calcul du rendement objectif : moyenne des 5 dernières campagnes
    en excluant la valeur max et min
    """
    if len(historique_rendements) < 5:
        raise ValueError("Minimum 5 campagnes nécessaires")
    rendements = sorted(historique_rendements)[1:-1]  # exclut min et max
    return sum(rendements) / len(rendements)

#🧠 Dictionnaire des coefficients b par culture
AZOTE_COEFFICIENTS = {
    "avoine": 2.2,
    "blé_dur": 2.2,
    "blé_tendre_améliorant": 3.0,
    "blé_tendre_mélangé": 3.0,  # ou 3.2 selon variétés
    "colza": 6.5,
    "lin_oléagineux": 3.5,
    "maïs_ensilage_<14": 8.0,
    "maïs_ensilage_14_18": 7.5,
    "maïs_ensilage_>18": 7.0,
    "maïs_grain_<30": 2.5,
    "maïs_grain_30_90": 2.3,
    "maïs_grain_>90": 2.1,
    "maïs_semence": 2.5,
    "orge_printemps": 2.2,
    "orge_hiver_brassicole": 2.2,
    "orge_hiver_non_brassicole": 2.5,
    "prairie_temporaire": 30.0  # forfaitaire
}
#📦 Fonction de calcul des besoins Pf
def calcul_pf(culture, rendement_objectif=None):
    """
    Calcule les besoins en azote Pf selon la culture et le rendement.
    Si la culture est forfaitaire, rendement_objectif est ignoré.
    """
    b = AZOTE_COEFFICIENTS.get(culture)
    if b is None:
        raise ValueError(f"Culture inconnue : {culture}")

    # Cas forfaitaire
    if culture == "prairie_temporaire":
        return b

    # Cas dépendant du rendement
    if rendement_objectif is None:
        raise ValueError(f"Rendement requis pour la culture : {culture}")

    return rendement_objectif * b

#🧮 1. Calcul du reliquat sortie hiver (maïs grain humide)
def reliquat_sortie_hiver(azote_total, rendement_maïs_humide):
    """
    Relevé du reliquat sortie hiver :
    Azote total - (100 × rendement) / (100 - 15)
    """
    correction = (100 * rendement_maïs_humide) / (100 - 15)
    return azote_total - correction

#🌾 2. Besoins forfaitaires Pf (Tableau 3)
BESOINS_FORFAITAIRES = {
    "maïs_fourrage": 160,
    "maïs_grain_humide": 180,
    "maïs_demi_sec": 200,
    "maïs_grain_sec": 220,
    "tournesol": 80,
    "betterave_graine": 160,
    "pomme_terre_demi_precoce": 220,
    "pomme_terre_precoce": 250
}

def pf_forfaitaire(culture):
    """
    Retourne le besoin forfaitaire Pf en kg N/ha
    """
    pf = BESOINS_FORFAITAIRES.get(culture)
    if pf is None:
        raise ValueError(f"Culture non reconnue : {culture}")
    return pf

#🧱 3. Azote restant dans le sol (Rf) selon texture
AZOTE_RESTANT_SOL = {
    "sableux": 10,
    "limoneux": 30,
    "argileux": 30
}

PROFONDEUR_RELIQUAT = {
    "sableux": "0-30 cm",
    "limoneux": "0-60 cm",
    "argileux": "0-60 cm"
}

def rf_sol(texture):
    """
    Retourne l'azote restant dans le sol Rf en kg N/ha
    """
    rf = AZOTE_RESTANT_SOL.get(texture)
    if rf is None:
        raise ValueError(f"Texture inconnue : {texture}")
    return rf

def profondeur_mesure(texture):
    """
    Retourne la profondeur de mesure recommandée
    """
    return PROFONDEUR_RELIQUAT.get(texture, "Non spécifiée")

#🧱 1. Table des valeurs Rf par texture et profondeur
RF_TABLE = {
    30: {"sableux": 5, "limoneux": 10, "argileux": 15},
    45: {"sableux": 8, "limoneux": 12, "argileux": 18},
    60: {"sableux": 10, "limoneux": 15, "argileux": 20},
    90: {"sableux": 12, "limoneux": 18, "argileux": 25}
}

#🧮 2. Fonction principale de calcul Rf
def calcul_rf(texture, profondeur):
    """
    Calcule Rf selon texture et profondeur.
    Gère les cas standards et les cas > 90 cm.
    """
    texture = texture.lower()

    if profondeur in RF_TABLE:
        return RF_TABLE[profondeur].get(texture, None)

    elif profondeur > 90:
        rf_90 = RF_TABLE[90].get(texture)
        if rf_90 is None:
            raise ValueError(f"Texture inconnue : {texture}")
        return rf_90 * (profondeur / 90)

    elif profondeur == 70 and texture == "limoneux":
        rf_30 = RF_TABLE[30]["limoneux"]
        rf_90 = RF_TABLE[90]["limoneux"]
        rf_60 = (rf_90 - rf_30) / 3
        return rf_30 + rf_60

    else:
        raise ValueError(f"Profondeur non standard : {profondeur} cm")

def calcul_pi_par_pesee(mveh=None, mvsh=None):
    """
    Calcule Pi (azote absorbé) selon la méthode par pesée.

    Paramètres :
    - mveh : Masse végétale fraîche à l'entrée de l'hiver (kg/m²)
    - mvsh : Masse végétale fraîche à la sortie de l'hiver (kg/m²)

    Retourne :
    - Pi : Azote absorbé estimé (kg N/ha)

    Règles :
    - NabsEH = mveh * 50
    - NabsSH = mvsh * 65
    - Si les deux sont disponibles :
        - Si NabsEH > NabsSH : Pi = NabsSH + (0.5 * (NabsEH - NabsSH) / 1.35)
        - Sinon : Pi = NabsSH
    - Si seul mvsh est fourni : Pi = NabsSH
    """
    if mvsh is None:
        raise ValueError("La masse végétale à la sortie de l'hiver (mvsh) est obligatoire.")

    nabs_sh = mvsh * 65

    if mveh is not None:
        nabs_eh = mveh * 50
        if nabs_eh > nabs_sh:
            pi = nabs_sh + (0.5 * (nabs_eh - nabs_sh) / 1.35)
        else:
            pi = nabs_sh
    else:
        pi = nabs_sh

    return round(pi, 2)
