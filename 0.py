import os
import fitz  # PyMuPDF

def compresser_pdf_robuste(chemin_entree, chemin_sortie):
    if not os.path.exists(chemin_entree):
        print(f"❌ Erreur : Le fichier n'existe pas :\n   {chemin_entree}")
        return

    print("⚡ Analyse et compression du PDF en cours...")
    
    try:
        # Ouverture du PDF (PyMuPDF répare automatiquement les tables d'objets cassées)
        doc = fitz.open(chemin_entree)
        
        # Sauvegarde optimisée avec compression des images et du texte
        doc.save(
            chemin_sortie, 
            garbage=4,             # Nettoie les objets dupliqués et inutilisés
            deflate=True,          # Compresse les flux de données (textes/vecteurs)
            clean=True             # Reconstruit et répare la structure interne du PDF
        )
        doc.close()

        # Calcul du gain d'espace
        taille_initiale = os.path.getsize(chemin_entree) / (1024 * 1024)
        taille_finale = os.path.getsize(chemin_sortie) / (1024 * 1024)
        
        print("\n✅ Opération réussie !")
        print(f"📉 Taille initiale : {taille_initiale:.2f} MB")
        print(f"📉 Nouvelle taille  : {taille_finale:.2f} MB")
        print(f"💾 Fichier réparé et compressé : {chemin_sortie}")

    except Exception as e:
        print(f"❌ Une erreur est survenue lors du traitement : {e}")

# Chemins de vos fichiers
chemin_fichier_origine = r"C:\Downloads\Yen.pdf"
chemin_fichier_compresse = r"C:\Downloads\Yeni-Maliye-02-03-2012-e2 (2)_compresse.pdf"

# Exécution
compresser_pdf_robuste(chemin_fichier_origine, chemin_fichier_compresse)
