# Analyse transcriptomique du stress hydrique chez le riz : description methodologique et interpretation des resultats

## Resume

Ce document presente une lecture academique du pipeline de classification supervisee applique a des donnees transcriptomiques de riz (`n = 18` echantillons, environ `p = 25 000` genes). L'objectif est de distinguer des echantillons `Control (CT)` et `Drought (D)` a l'aide de modeles de machine learning, tout en limitant les risques de fuite d'information et de surapprentissage.  
Les resultats mettent en evidence un signal discriminant, mais la robustesse reste limitee par la tres faible taille d'echantillon. Les performances doivent donc etre interpretees comme preliminaires et exploratoires.

## 1. Contexte scientifique et objectif

Le stress hydrique induit des reconfigurations transcriptomiques susceptibles d'etre exploitees pour la classification des etats physiologiques. Dans ce cadre, la question analysee est :

**Peut-on predire l'etat hydrique (controle vs stress) a partir du transcriptome foliaire ?**

L'etude est conduite dans un contexte **haute dimension - faible effectif** (`p >> n`), classiquement associe a :

- un risque eleve de surapprentissage,
- une variance importante des estimateurs de performance,
- une stabilite limitee des signatures genes candidates.

## 2. Donnees et pretraitement

Les donnees proviennent d'une table TPM (`TPM_table.txt`) initialement organisee en `genes x echantillons`, puis transposee en `echantillons x genes`.  
Les etiquettes de classes sont deduites des identifiants d'echantillons :

- `CT` : condition controle,
- `D` : condition stress hydrique.

La repartition observee est equilibree (9 vs 9).

Un split stratifie `train/test` est effectue **avant toute transformation** :

- train : 13 echantillons,
- test : 5 echantillons.

Les genes a variance nulle sont retires sur l'ensemble d'entrainement, puis le meme sous-ensemble de colonnes est applique au test.

## 3. Strategie methodologique

## 3.1 Selection de variables

La reduction dimensionnelle est effectuee via :

- `SelectFromModel`
- `LogisticRegression` penalisee `L1` (`liblinear`)
- plafond de variables retenues (en pratique 20 genes max)

Ce choix privilegie la parcimonie et diminue le risque de capter des correlations fortuites.

## 3.2 Modeles entraines

Deux familles de modeles sont comparees :

- **Random Forest** (modele non lineaire, interpretable via importance des variables),
- **MLP** (reseau de neurones), precede d'une standardisation et d'une reduction PCA.

## 3.3 Evaluation et robustesse

Plusieurs niveaux d'evaluation sont combines :

- score sur test hold-out (5 echantillons),
- cross-validation stratifiee 3-fold,
- repeated stratified 3-fold (10 repetitions),
- LOOCV (Leave-One-Out),
- test de permutation (comparaison au hasard).

L'approche est coherente avec les bonnes pratiques anti fuite d'information (selection et modelisation encapsulees dans les pipelines CV).

## 4. Resultats principaux

### Random Forest

- Test F1 : **0.8000**
- CV 3-fold F1 : **0.7667 +/- 0.2055**
- CV repetee F1 : **0.6617 +/- 0.2004**
- LOOCV F1 : **0.3077 +/- 0.4615**

### MLP

- Test F1 : **0.5714**
- CV 3-fold F1 : **0.6349 +/- 0.0449**
- LOOCV F1 : **0.5385 +/- 0.4985**

### Test contre le hasard (RF)

- score reel (CV) : **0.7667**
- score aleatoire moyen : **0.4747**
- gain : **+0.2920**

Ces resultats suggerent un signal biologique non nul, mais une incertitude substantielle sur la capacite de generalisation.

## 5. Interpretation des figures

## 5.1 Figure principale : `ml_results_final.png`

Cette figure assemble six vues complementaires :

- matrices de confusion RF et MLP,
- courbes ROC,
- top genes classes par importance RF,
- comparaison des distributions de scores CV/LOOCV,
- projection PCA des echantillons.

Point cle : les performances observees sur le test (taille tres faible) ne doivent pas etre considerees seules ; la confrontation aux scores LOOCV/CV est indispensable.

## 5.2 Courbe d'apprentissage : `learning_curve.png`

La courbe montre un ecart persistant entre performance train et validation, avec des intervalles larges sur validation. Ce profil est compatible avec un regime de donnees insuffisant et un risque d'overfitting.

## 5.3 Graphique de synthese : `performance_explicative.png`

Cette figure compare, pour RF et MLP, les metriques suivantes :

- Test F1,
- CV 3-fold F1,
- CV repetee F1,
- LOOCV F1.

Les barres d'erreur visualisent la variance estimee. La baisse des performances vers LOOCV, en particulier pour RF, appuie une interpretation prudente de la performance reelle.

## 6. Discussion

## 6.1 Points methodologiquement solides

- separation train/test realisee avant transformations,
- pipelines CV explicites (limitation de la fuite d'information),
- recours a plusieurs protocoles de validation,
- test statistique contre le hasard,
- selection de variables regularisee et contrainte en cardinalite.

## 6.2 Limites majeures

- effectif tres faible (`n = 18`),
- test hold-out tres petit (`n_test = 5`),
- forte variabilite inter-folds,
- stabilite partielle des genes selectionnes,
- absence de validation externe independante.

En consequence, les genes identifies doivent etre qualifies de **candidats exploratoires**, et non de biomarqueurs valides.

## 7. Conclusion

Le pipeline implante est rigoureux au plan procedurale et met en evidence un signal de discrimination entre conditions hydriques. Toutefois, la robustesse statistique demeure limitee par la taille de l'echantillon.  
Les performances sont defendables dans une perspective exploratoire, mais elles ne permettent pas, a ce stade, une revendication de generalisation forte.

## 8. Perspectives

Pour consolider scientifiquement les conclusions, il est recommande de :

- augmenter significativement la taille d'echantillon (idealement >= 100),
- effectuer une validation externe sur cohorte independante,
- tester explicitement la transferabilite inter-cultivars,
- confirmer experimentalement les genes candidats (par ex. qRT-PCR),
- completer par des approches de stabilite de selection (stability selection, bootstrap).

