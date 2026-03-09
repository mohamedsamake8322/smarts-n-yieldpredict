"""
PLATFORM CONFIGURATION - Quick Reference
Fast customization guide for the agricultural AI platform
"""

# ============================================================================
# 🔧 QUICK CUSTOMIZATION
# ============================================================================

# 1. FACEBOOK COMMUNITY
# File: pages/4_Farmers_Community.py
# Find and replace this URL with yours:
FACEBOOK_GROUP_URL = "https://www.facebook.com/share/1AkjYeh8ty/"

# 2. SUPPORT EMAIL
# File: app.py, pages/4_Farmers_Community.py
SUPPORT_EMAIL = "contact@plantdisease.ai"

# 3. PLATFORM NAME
# File: app.py
PLATFORM_NAME = "🌾 AI Plant Disease Diagnostic Platform"

# 4. AVAILABLE CROPS (for user profile)
# File: pages/6_User_Dashboard.py
AVAILABLE_CROPS = [
    "Maize", "Rice", "Cassava", "Banana", "Potato",
    "Tomato", "Onion", "Pepper", "Beans", "Groundnut",
    "Cotton", "Coffee", "Cocoa", "Sorghum", "Millet",
]

# 5. COUNTRIES FOR PROFILE
# File: pages/6_User_Dashboard.py
COUNTRIES = [
    "Senegal", "Mali", "Côte d'Ivoire", "Burkina Faso", "Niger",
    "Ghana", "Nigeria", "Cameroon", "Kenya", "Tanzania",
]

# ============================================================================
# 📊 PLATFORM METRICS (Home page display)
# ============================================================================

PLATFORM_STATS = {
    "total_users": 10000,
    "countries": 45,
    "diseases_detected": 200,
    "accuracy": 0.81,  # 81%
}

# ============================================================================
# 🎨 COLOR SCHEME
# ============================================================================

COLORS = {
    "primary": "#2ecc71",      # Green
    "secondary": "#3498db",    # Blue
    "danger": "#e74c3c",       # Red
    "warning": "#f39c12",      # Orange
}

# ============================================================================
# 🔐 AI MODEL SETTINGS
# ============================================================================

# Model path (from model_core.py)
MODEL_PATH = "models/phase2_swin_base"

# Unknown disease detection threshold
UNKNOWN_THRESHOLD = 0.55  # (0.3-0.9 range)

# Top similar images to show
TOP_K = 5

# Image size
IMAGE_SIZE = 224

# ============================================================================
# 💾 DATA STORAGE
# ============================================================================

# All data stored in:
# data/app_storage/
#   ├── history.json           # Diagnosis history
#   ├── feedback.json          # User feedback
#   ├── app.db                 # SQLite database
#   └── dataset_to_review/     # Misdiagnosed images

# ============================================================================
# 📚 DISEASE LIBRARY
# ============================================================================

# Update diseases in: data/disease_info.json
# Example structure:
"""
{
    "Leaf Blight": {
        "description": "...",
        "symptoms": "...",
        "treatment": "...",
        "prevention": "...",
        "severity": "high",
        "category": "fungal"
    }
}
"""

# ============================================================================
# 🚀 DEPLOYMENT QUICK COMMANDS
# ============================================================================

"""
# Start the application:
streamlit run app.py

# Force refresh (clear cache):
streamlit run app.py --logger.level=debug

# Deploy to Streamlit Cloud:
# 1. Push code to GitHub
# 2. Go to https://share.streamlit.io/
# 3. Choose your repo
# 4. Add secrets in deploy settings if needed

# Deploy to server:
pip install streamlit
streamlit run app.py --server.port=8501
"""

# ============================================================================
# ✅ FEATURE FLAGS (enable/disable features)
# ============================================================================

FEATURES = {
    "detect_disease": True,
    "history_tracking": True,
    "statistics_dashboard": True,
    "community_integration": True,
    "disease_library": True,
    "user_profiles": True,
    
    # Coming soon
    "built_in_forum": False,
    "expert_marketplace": False,
    "mobile_app": False,
}

# ============================================================================
# 📞 SUPPORT CONTACTS
# ============================================================================

CONTACTS = {
    "email": "support@plantdisease.ai",
    "facebook": "https://www.facebook.com/share/1AkjYeh8ty/",
    "phone": "+221-XXXXX",  # Optional
    "website": "https://plantdisease.ai/",  # Optional
}

# ============================================================================
# 📋 COMMON UPDATES CHECKLIST
# ============================================================================

"""
TASK: Add new disease to library
WHERE: data/disease_info.json
CODE:
{
    "New Disease Name": {
        "description": "Brief description",
        "symptoms": "List of symptoms",
        "treatment": "Treatment recommendations",
        "prevention": "Prevention methods",
        "severity": "high|medium|low",
        "category": "fungal|bacterial|viral|pest|physiological"
    }
}

TASK: Change Facebook group
WHERE: pages/4_Community.py (line ~40)
REPLACE: "https://www.facebook.com/share/1AkjYeh8ty/"
WITH: "your_facebook_group_url"

TASK: Add new country
WHERE: pages/6_Profile.py (line ~65)
ADD: "Your Country" to the selectbox list

TASK: Update model path
WHERE: model_core.py (line ~XX)
CHANGE: MODELS_PATH_PHASE2 = "your/model/path"

TASK: Adjust confidence threshold
WHERE: pages/1_Detect_Disease.py (line ~XX)
CHANGE: value=0.55 to desired threshold

TASK: Enable new feature
WHERE: pages/X_FeatureName.py
UNCOMMENT or REMOVE: st.info("Coming Soon")
"""

# ============================================================================
# 🔄 HOW DATA FLOWS
# ============================================================================

"""
USER JOURNEY:

1. USER UPLOADS IMAGE
   ├── Image sent to AI model
   │
2. MODEL PROCESSES
   ├── Disease detection
   ├── Confidence calculation
   ├── Similarity search
   │
3. RESULTS DISPLAYED
   ├── Disease name
   ├── Confidence score
   ├── Similar images
   ├── Grad-CAM heatmap
   │
4. USER FEEDBACK
   ├── ✔️ Correct → Saves to history.json
   ├── ❌ Incorrect → Saves to dataset_to_review/
   ├── ❓ Unsure → Logs for analysis
   │
5. DATA STORED
   ├── diagnostic_history table
   ├── Aggregated in statistics
   ├── Used for accuracy tracking

DATA FILES:
- history.json: {"image_name", "disease", "confidence", "date", "user_feedback"}
- app.db: SQLite with diagnostic_history, statistics, users tables
- dataset_to_review/: Misdiagnosed images for retraining
"""

# ============================================================================
# 🎯 KEY FILES TO KNOW
# ============================================================================

"""
CORE FILES:
├── app.py                      # 🏠 Home page (entry point)
├── model_core.py               # 🧠 AI model interface
├── utils/storage.py            # 💾 Data persistence (NEW!)
│
PAGES:
├── pages/1_Detect_Disease.py   # 🔍 AI diagnosis tool
├── pages/2_History.py          # 📜 Diagnosis history
├── pages/3_Statistics.py       # 📊 Platform analytics
├── pages/4_Community.py        # 👥 Community & Facebook
├── pages/5_Library.py          # 📚 Disease database
├── pages/6_User_Dashboard.py          # 👤 User management
│
DATA:
├── data/disease_info.json      # 🦠 Disease database
├── data/app_storage/
│   ├── history.json            # All diagnoses
│   ├── feedback.json           # User feedback
│   ├── app.db                  # SQLite main DB
│   └── dataset_to_review/      # Misdiagnosed images
│
DOCS:
├── PLATFORM_ARCHITECTURE.md    # 📘 Complete guide
├── RESTRUCTURING_SUMMARY.md    # ✅ What was done
└── QUICK_CUSTOMIZATION.md      # ⚡ This file

CONFIG FILES:
├── config.py                   # Existing config
├── requirements.txt            # Python packages
└── Dockerfile                  # If needed
"""

# ============================================================================
# 🐛 TROUBLESHOOTING
# ============================================================================

"""
ISSUE: "ModuleNotFoundError: No module named 'storage'"
FIX: pip install -r requirements.txt

ISSUE: Model fails to load
FIX: Check MODELS_PATH_PHASE2 in model_core.py
     Ensure model files exist in that path

ISSUE: No data is being saved
FIX: Check data/app_storage/ folder is created
     Run: from utils.storage import ensure_db; ensure_db()

ISSUE: Images won't upload
FIX: Check file size < 10MB
     Check format in SUPPORTED_IMAGE_FORMATS
     Try different image

ISSUE: Database locked error
FIX: Close other instances of streamlit app
     Or restart Python kernel

ISSUE: Slow performance
FIX: Image preprocessing is normal (first run slower)
     Clear browser cache
     Restart streamlit server
"""

# ============================================================================
# 📈 MONITORING DASHBOARD
# ============================================================================

"""
TO CHECK PLATFORM HEALTH:

1. Open Statistics page (📊)
   ├── Check total scans
   ├── Monitor accuracy rate
   ├── See trending diseases

2. Check History page (📜)
   ├── See if diagnoses are being saved
   ├── View feedback distribution
   ├── Check for any errors

3. Monitor files:
   ├── data/app_storage/history.json (size growing?)
   ├── data/app_storage/dataset_to_review/ (collecting misdiagnosed?)
   ├── data/app_storage/app.db (file size)

4. Check logs:
   ├── Browser console (F12)
   ├── Terminal where streamlit runs
"""

# ============================================================================
# 🚀 DEPLOYMENT OPTIONS
# ============================================================================

"""
OPTION 1: LOCAL (Laptop/Desktop)
- Run: streamlit run app.py
- Access: http://localhost:8501
- Data: Stored on your machine

OPTION 2: STREAMLIT CLOUD (Free)
- Push to GitHub
- Deploy via share.streamlit.io
- Limited compute, good for demo

OPTION 3: HEROKU
- pip install heroku
- heroku login
- git push heroku main

OPTION 4: DOCKER
- Build: docker build -t platform .
- Run: docker run -p 8501:8501 platform

OPTION 5: SERVER
- Install: apt-get install python3-streamlit
- Run: nohup streamlit run app.py &
- Monitor with: pm2 or supervisor
"""

# ============================================================================
# 📝 NEXT STEPS
# ============================================================================

"""
IMMEDIATE (Do now):
1. ✅ Test all pages work
2. ✅ Try upload, get feedback
3. ✅ Check data is saved
4. ✅ Visit Statistics page
5. ✅ Join Facebook community

SHORT-TERM (This week):
1. Update disease library with local diseases
2. Customize colors & branding
3. Add Facebook group link
4. Set up email support

MEDIUM-TERM (This month):
1. Collect more user feedback
2. Train custom model on local data
3. Set up regular backups
4. Plan mobile app
5. Consider Firebase migration

LONG-TERM (3-6 months):
1. Build in-app forum
2. Add expert marketplace
3. Create mobile app
4. Integrate with government
5. Plan fundraising/monetization
"""

print("✅ PLATFORM CONFIGURATION LOADED")
print("📖 For complete guide, see: PLATFORM_ARCHITECTURE.md")
print("⚡ Quick setup, see: RESTRUCTURING_SUMMARY.md")
