# 🌾 Smart Agriculture Application - Setup & Execution Guide

## Architecture Overview

The application is structured in three main components:

### 1. **Visual Diagnosis Module** (`modules/visual_diagnosis.py`)
- Classifies plant diseases using Swin Transformer
- Retrieves disease info from normalized BLIP2 JSON (109 files)
- Generates explanations with BLIP-2

### 2. **Agricultural Assistant Module** (`modules/agricultural_assistant.py`)
- Searches Plantwise knowledge base (1115 files)
- Uses FAISS for vector-based semantic search
- Provides actionable recommendations

### 3. **Streamlit Interface**
- `pages/2_Disease_Detection.py` - Progressive diagnosis workflow
- `pages/3_Agricultural_Assistant.py` - Q&A knowledge base


---

## 📋 Setup Instructions (Local Machine)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Normalize BLIP2 JSON Files

This creates a common schema for all 109 disease files, making parsing and integration easier.

```bash
python normalize_blip2.py
```

**Output:**
- Creates `BLIP2_normalized/` directory
- Contains 109 normalized JSON files with standardized fields

### 3. Build FAISS Index for Plantwise Knowledge Base

This creates a vector index for semantic search across 1115 agricultural knowledge files.

```bash
python build_moh_index.py
```

**Output:**
- `moh_index.faiss` - FAISS vector index
- `moh_metadata.json` - Metadata for search results

### 4. Test the Modules

Verify the setup works correctly:

```bash
python test_modules.py
```

**Expected output:**
```
✅ BLIP2 JSON normalized to common schema
✅ FAISS index built for Moh knowledge base
✅ Visual Diagnosis module created
✅ Agricultural Assistant module created
Ready for integration into Streamlit app!
```

### 5. Launch the Streamlit Application

```bash
streamlit run 04_app_streamlit.py
```

Then navigate to:
- **Disease Detection:** Page 2 - Progressive diagnosis workflow
- **Agricultural Assistant:** Page 3 - Q&A knowledge base


---

## 🔧 Google Colab Setup

### Step 1: Copy Project to Google Drive

Upload the entire project to Google Drive at:
```
/MyDrive/smarts-n-yieldpredict/
```

### Step 2: Create Colab Notebook

Create a new Colab notebook and run the following cells:

**Cell 1: Mount Drive and Install Packages**
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/smarts-n-yieldpredict')

# Install required packages
!pip install -q sentence-transformers faiss-cpu streamlit torch torchvision transformers pillow
```

**Cell 2: Check Configuration**
```python
from config import print_config
print_config()
```

**Cell 3: Normalize BLIP2 Files**
```python
import subprocess
result = subprocess.run(['python', 'normalize_blip2.py'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr)
```

**Cell 4: Build FAISS Index**
```python
result = subprocess.run(['python', 'build_moh_index.py'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr)
```

**Cell 5: Test Modules**
```python
result = subprocess.run(['python', 'test_modules.py'], capture_output=True, text=True)
print(result.stdout)
```

**Cell 6: Launch Streamlit**
```python
!streamlit run 04_app_streamlit.py --logger.level=error
```

Then click the public URL to access the app.


---

## 📦 File Structure

```
smarts-n-yieldpredict/
├── config.py                          # Configuration (paths, env detection)
├── normalize_blip2.py                 # BLIP2 normalization script
├── build_moh_index.py                 # FAISS index builder
├── test_modules.py                    # Module testing
├── setup_colab.py                     # Colab setup helper
├── 04_app_streamlit.py                # Main Streamlit app
├── BLIP2/                             # Original BLIP2 JSONs (109 files)
├── BLIP2_normalized/                  # Normalized BLIP2 JSONs
├── Moh/                               # Plantwise JSONs (1115 files)
├── models/                            # ML models directory
├── moh_index.faiss                    # FAISS vector index
├── moh_metadata.json                  # Search metadata
├── modules/
│   ├── __init__.py
│   ├── visual_diagnosis.py            # Disease detection module
│   └── agricultural_assistant.py      # Knowledge base assistant
└── pages/
    ├── 2_Disease_Detection.py         # Progressive diagnosis interface
    └── 3_Agricultural_Assistant.py    # Q&A interface
```


---

## 🚀 Execution Order (Step-by-Step)

### For Local Machine:

```bash
# 1. Activate Python environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Normalize BLIP2 files (109 files → standardized schema)
python normalize_blip2.py

# 4. Build FAISS index (1115 Plantwise files → searchable index)
python build_moh_index.py

# 5. Test the modules
python test_modules.py

# 6. Launch Streamlit application
streamlit run 04_app_streamlit.py
```

### For Google Colab:

```python
# Copy the 6 cells from the "Google Colab Setup" section above
# Run them in order in your Colab notebook
```


---

## 💡 Features & Workflow

### Disease Detection Tab (Progressive)

1. **Upload Image**
   - User uploads plant image
   - App analyzes with Swin Transformer

2. **View Predictions**
   - Shows top-3 disease predictions with confidence scores
   - E.g., "Corn smut — 97%", "Maize rust — 2%"

3. **Confirm Diagnosis**
   - App shows basic info: name, causal agent, symptoms
   - Asks: "Does this match what you see?"
   - User confirms or rejects

4. **Get Recommendations**
   - If confirmed: shows management, prevention, treatment details
   - If rejected: suggests uploading another image

### Agricultural Assistant Tab

1. **Ask Questions**
   - "How to control bean bruchid?"
   - "What are symptoms of maize rust?"
   - Assistant searches 1115 Plantwise entries

2. **Get Answers**
   - Returns relevant sources with explanations
   - Organized by Prevention, Monitoring, Direct Control

3. **Browse Topics**
   - Explore pest control, prevention, disease management
   - Browse entire knowledge base by topic


---

## 📚 Key Concepts

### Normalized BLIP2 Schema
Common fields across all 109 disease files:
- `name` - Disease/pest name
- `scientific_name` - Scientific nomenclature
- `causal_agent` - Pathogen type
- `hosts` - Affected crops
- `symptoms` - Symptom descriptions
- `description` - Full description
- `management` - Management strategies
- `prevention` - Prevention methods
- `sources` - References

### FAISS Vector Search
- **Model:** Sentence Transformers (all-MiniLM-L6-v2)
- **Database:** 1115 Plantwise knowledge entries
- **Search:** Semantic search for user queries
- **Results:** Top-k relevant entries with relevance scores


---

## 🔍 Troubleshooting

### BLIP2 files not normalizing
- Check that `BLIP2/` directory exists with JSON files
- Run: `python normalize_blip2.py` again

### FAISS index not building
- Ensure Moh directory has JSON files
- Check available disk space
- Run: `python build_moh_index.py` again

### Modules not loading in Streamlit
- Verify `modules/` directory exists with `__init__.py`
- Check that config paths are correct
- Run: `python test_modules.py` to diagnose

### Colab path issues
- Ensure project is at `/content/drive/MyDrive/smarts-n-yieldpredict/`
- Run `from config import print_config` to verify paths


---

## 📊 Performance Notes

- **Normalization:** ~5-10 seconds for 109 files
- **Index Building:** ~1-2 minutes for 1115 files
- **Search Query:** ~100-500ms per query
- **App Load:** First load ~5-10 seconds (caching models)


---

## 📝 Next Steps (Optional Enhancements)

1. **Load Real Models**
   - Replace mock Swin classification with real model
   - Integrate actual BLIP-2 for explanation generation

2. **Multilingual Support**
   - Add NLLB-200 for translation
   - Support multiple crop languages

3. **Voice Output**
   - Add text-to-speech for recommendations
   - Accessibility improvement

4. **Community Features**
   - Store user reports
   - Share solutions with farmers
   - Build knowledge from field observations


---

**Last Updated:** March 16, 2026
