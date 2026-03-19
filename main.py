#!/usr/bin/env python3
"""
SMART AGRICULTURE - Script Principal
Orchestrateur pour lancer les différents modules du système
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Smart Agriculture System')
    parser.add_argument('action', choices=['train', 'index', 'app', 'all'],
                       help='Action à effectuer')
    parser.add_argument('--skip-training', action='store_true',
                       help='Sauter l\'entraînement si modèle existe')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port pour l\'application (défaut: 8501)')

    args = parser.parse_args()

    print("="*80)
    print("🌱 SMART AGRICULTURE - SYSTÈME COMPLET")
    print("="*80)

    if args.action == 'train':
        print("🤖 Lancement du module d'entraînement...")
        from train_model import main as train_main
        train_main()

    elif args.action == 'index':
        print("🔍 Lancement du module d'indexation...")
        from build_index import main as index_main
        index_main()

    elif args.action == 'app':
        print(f"🌐 Lancement de l'application (port {args.port})...")
        # Modifier le port dans l'environnement
        import os
        os.environ['STREAMLIT_SERVER_PORT'] = str(args.port)
        from start_app import main as app_main
        app_main()

    elif args.action == 'all':
        print("🚀 Lancement du pipeline complet...")

        # 1. Entraînement (optionnel)
        if not args.skip_training:
            print("\n" + "="*60)
            print("ÉTAPE 1: ENTRAÎNEMENT")
            print("="*60)
            try:
                from train_model import main as train_main
                train_main()
            except Exception as e:
                print(f"❌ Erreur entraînement: {e}")
                if "modèle existe" not in str(e).lower():
                    sys.exit(1)
        else:
            print("⏭️ Entraînement sauté (modèle existant)")

        # 2. Indexation
        print("\n" + "="*60)
        print("ÉTAPE 2: INDEXATION")
        print("="*60)
        try:
            from build_index import main as index_main
            index_main()
        except Exception as e:
            print(f"❌ Erreur indexation: {e}")
            sys.exit(1)

        # 3. Application
        print("\n" + "="*60)
        print("ÉTAPE 3: APPLICATION")
        print("="*60)
        try:
            import os
            os.environ['STREAMLIT_SERVER_PORT'] = str(args.port)
            from start_app import main as app_main
            app_main()
        except Exception as e:
            print(f"❌ Erreur application: {e}")
            sys.exit(1)

    print("\n🎉 Pipeline terminé!")

if __name__ == "__main__":
    main()