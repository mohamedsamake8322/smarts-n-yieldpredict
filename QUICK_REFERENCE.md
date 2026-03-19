# 🚀 QUICK REFERENCE - Run Smart Agriculture App

## One-Command Launch (Recommended)
```bash
python quickstart.py
```
✅ Automatically runs all setup steps

---

## Manual Step-by-Step

```bash
# 1. Activate environment
.\env311\Scripts\Activate.ps1    # Windows
# source env311/bin/activate    # Mac/Linux

# 2. Install dependencies (if needed)
pip install -r requirements.txt

# 3. Normalize BLIP2 files (109)
python normalize_blip2.py

# 4. Build FAISS index (1115)
python build_moh_index.py

# 5. Test modules
python test_modules.py

# 6. Launch app
streamlit run 04_app_streamlit.py
```

Access: http://localhost:8501

---

## Google Colab - Script Unique Automatisé

**🚀 Option Ultra-Simple (Recommandé):**
```python
!python colab_setup_complete.py
```

**📖 Guide one-click:** Voir `COLAB_ONE_CLICK.md` pour instructions simplifiées

---

**Cell 3:**
```bash
!python normalize_blip2.py 2>&1 | tail -20
```

**Cell 4:**
```bash
!python build_moh_index.py 2>&1 | tail -10
```

**Cell 5:**
```bash
!python test_modules.py
```

**Cell 6:**
```bash
!streamlit run 04_app_streamlit.py --logger.level=error
```

---

## Files Executed in Order

| # | File | Purpose | Command |
|---|------|---------|---------|
| 1 | `normalize_blip2.py` | Standardize 109 disease JSONs | `python normalize_blip2.py` |
| 2 | `build_moh_index.py` | Create FAISS index for 1115 entries | `python build_moh_index.py` |
| 3 | `test_modules.py` | Verify modules work | `python test_modules.py` |
| 4 | `04_app_streamlit.py` | Launch web interface | `streamlit run 04_app_streamlit.py` |

---

## Output Directories

- `BLIP2_normalized/` - Normalized disease files (109)
- `moh_index.faiss` - Vector index for search
- `moh_metadata.json` - Search metadata
- `models/swin_base_production/` - Swin Transformer model files

---

## Model Setup (Swin + BLIP-2)

**For Local Environment:**
```bash
# Download model files from Google Drive and place in models/swin_base_production/
python setup_swin_model.py
```

**For Google Colab:**
Model files are automatically available at the trained paths.

**Models Used:**
- **Swin Transformer Base**: 99.54% Recall@1 on 109 plant diseases
- **BLIP-2 (2.7B)**: Natural language explanations
- **FAISS Index**: Fast similarity search for disease matching

## Features Available

✅ **Disease Detection** (Page 2)
- Upload plant image
- **Swin Transformer** classification (99.5% accuracy on 109 diseases)
- See top-3 predictions with confidence scores
- Progressive workflow with confirmation
- **BLIP-2 explanations** in natural language
- Detailed management recommendations from Plantwise

✅ **Agricultural Assistant** (Page 3)
- Ask questions about pest/disease management
- Search 1115 Plantwise knowledge entries
- Browse by topic
- Get answers with sources

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | `pip install -r requirements.txt` |
| FAISS index error | `python build_moh_index.py` |
| Swin model not found | `python setup_swin_model.py` |
| BLIP-2 memory error | Use smaller model or more RAM |
| Port 8501 in use | `streamlit run ... --server.port=8502` |
| Colab path error | Ensure project at `/MyDrive/smarts-n-yieldpredict/` |

---

## Detailed Guides

- **Full Setup Guide:** `SETUP_AND_EXECUTION_GUIDE.md`
- **Execution Order:** `EXECUTION_ORDER.md`
- **Module Info:** `RUN_APP.py`

---

**Last Updated:** March 16, 2026
