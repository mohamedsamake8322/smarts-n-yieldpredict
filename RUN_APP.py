#!/usr/bin/env python3
"""
QUICK EXECUTION SUMMARY - Smart Agriculture Application

Use this file as your reference for running the app.
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    🌾 SMART AGRICULTURE APPLICATION                        ║
║                  Complete Setup & Execution Instructions                   ║
╚════════════════════════════════════════════════════════════════════════════╝

📌 QUICK START (Recommended - One Command)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python quickstart.py

    This runs ALL steps automatically:
    ✓ Normalize BLIP2 files (109)
    ✓ Build FAISS index (1115)
    ✓ Test modules
    ✓ Launch Streamlit


📋 MANUAL EXECUTION (Step by Step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Activate environment
    Windows:
        .\\env311\\Scripts\\Activate.ps1
    
    Mac/Linux:
        source env311/bin/activate

Step 2: Install dependencies (if needed)
    pip install -r requirements.txt

Step 3: Normalize BLIP2 files (109 → standardized schema)
    python normalize_blip2.py
    
    └─ Output: BLIP2_normalized/ directory with 109 files

Step 4: Build FAISS index (1115 Plantwise entries)
    python build_moh_index.py
    
    └─ Output: moh_index.faiss + moh_metadata.json

Step 5: Test the modules
    python test_modules.py
    
    └─ Verify everything works

Step 6: Launch Streamlit application
    streamlit run 04_app_streamlit.py
    
    └─ Access: http://localhost:8501


🔗 GOOGLE COLAB EXECUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create a new Colab notebook and run these cells in order:

Cell 1: Setup and Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')
    import os
    os.chdir('/content/drive/MyDrive/smarts-n-yieldpredict')
    !pip install -q sentence-transformers faiss-cpu streamlit torch transformers

Cell 2: Verify Config
    from config import print_config, ensure_directories
    ensure_directories()
    print_config()

Cell 3: Normalize BLIP2
    !python normalize_blip2.py 2>&1 | tail -20

Cell 4: Build Index
    !python build_moh_index.py 2>&1 | tail -10

Cell 5: Test
    !python test_modules.py

Cell 6: Launch
    !streamlit run 04_app_streamlit.py --logger.level=error


📊 ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Visual Diagnosis Module (pages/2_Disease_Detection.py)
   - Upload plant image
   - See top-3 predictions with confidence scores
   - Confirm diagnosis
   - Get detailed recommendations

2. Agricultural Assistant Module (pages/3_Agricultural_Assistant.py)
   - Ask questions about pest/disease management
   - Search 1115 Plantwise knowledge entries
   - Get recommendations with sources


⚙️ KEY SCRIPTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

normalize_blip2.py
    └─ Standardizes 109 disease JSON files
    └─ Creates: BLIP2_normalized/
    └─ Time: ~5-10 seconds

build_moh_index.py
    └─ Creates searchable vector index
    └─ Processes: 1115 Plantwise entries
    └─ Creates: moh_index.faiss + moh_metadata.json
    └─ Time: ~1-2 minutes

quickstart.py
    └─ Runs ALL steps automatically
    └─ Time: ~2-3 minutes total


🎯 WORKFLOW FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Progressive Disease Detection:
    1. User uploads image
    2. App shows: disease name, causal agent, main symptoms
    3. App asks: "Does this match what you observe?"
    4. If YES → Show: management, prevention, treatment details
    5. If NO → Suggest: upload another image

Top-3 Predictions:
    - Shows 3 most likely diseases with confidence scores
    - Example: "Corn smut — 97%, Maize rust — 2%, Leaf blight — 1%"

Agricultural Assistant:
    - Ask: "How to control bean bruchid?"
    - Get: Relevant advice from 1115 knowledge entries
    - Browse: by topic (Prevention, Pest Control, etc.)


✅ FILE STRUCTURE CREATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

config.py                           ← Path management and Colab support
normalize_blip2.py                  ← Schema normalization
build_moh_index.py                  ← Vector index creation
quickstart.py                       ← Automated setup
test_modules.py                     ← Module testing
setup_colab.py                      ← Colab helper
EXECUTION_ORDER.md                  ← Detailed instructions
SETUP_AND_EXECUTION_GUIDE.md        ← Complete guide

modules/
    ├─ visual_diagnosis.py          ← Disease detection
    └─ agricultural_assistant.py    ← Knowledge Q&A

pages/
    ├─ 2_Disease_Detection.py       ← Progressive detection interface
    └─ 3_Agricultural_Assistant.py  ← Q&A interface


📞 TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: normalize_blip2.py fails
    → Check BLIP2/ directory exists
    → Verify JSON files are readable
    → Run again: python normalize_blip2.py

Issue: FAISS index not building
    → Ensure Moh/ directory has files
    → Check available disk space
    → Run again: python build_moh_index.py

Issue: Module import errors
    → Install dependencies: pip install -r requirements.txt
    → Verify modules/ has __init__.py
    → Run test: python test_modules.py

Issue: Streamlit not launching
    → Try: python -m streamlit run 04_app_streamlit.py
    → Check port 8501 is available
    → Verify Python 3.8+


🌐 ACCESSING THE APP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Local Machine:
    → http://localhost:8501

Google Colab:
    → Click the public URL shown in Colab output


⏱️ ESTIMATED TIME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

normalize_blip2.py:        ~5-10 seconds
build_moh_index.py:        ~1-2 minutes  (depends on CPU)
test_modules.py:           ~30 seconds
Total (with quickstart.py): ~2-3 minutes


╔════════════════════════════════════════════════════════════════════════════╗
║                     Ready to launch? Run:                                  ║
║                                                                            ║
║                        python quickstart.py                                ║
║                                                                            ║
║                     or follow manual steps above                           ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    # Offer to run quickstart
    response = input("\nWould you like to start quickstart.py now? [y/n]: ").strip().lower()
    if response == 'y':
        import subprocess
        import sys
        subprocess.run([sys.executable, 'quickstart.py'])
