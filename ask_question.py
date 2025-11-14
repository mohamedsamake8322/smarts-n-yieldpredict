from vector_store import VectorStore

# 📂 Charger la base vectorielle sauvegardée
store = VectorStore()
store.load_store("vector_store.pkl")

# 🧠 Vérification du nombre de chunks disponibles
total_chunks = len(store.documents)
print(f"\n📦 Base vectorielle chargée avec {total_chunks} chunks.")

if total_chunks == 0:
    print("⚠️ Aucun chunk en mémoire. Veuillez exécuter ingest_documents.py pour alimenter la base.")
    exit()

# 🔍 Question à poser
query = "Quelle est l’origine du palmier à huile et où est-il cultivé ?"
print(f"\n🔎 Recherche pour la question : {query}")

# 🔎 Recherche sémantique
results = store.search(query)

# 📊 Affichage des résultats
if not results:
    print("\n❌ Aucun résultat pertinent trouvé.")
else:
    print(f"\n✅ {len(results)} résultats pertinents trouvés :\n")
    for i, r in enumerate(results, 1):
        print(f"🔹 Résultat {i}")
        print(f"📄 Fichier : {r['filename']}")
        print(f"📈 Similarité : {r['similarity']:.4f}")
        print(f"🧠 Contenu : {r['content'][:300].strip()}...")
        print(f"🏷️ Métadonnées : {r['metadata']}")
        print("-" * 60)
