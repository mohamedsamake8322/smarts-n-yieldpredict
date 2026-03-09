"""
LIBRARY - Disease Knowledge Base & Educational Resources
Complete information about plant diseases, treatments, and prevention
"""

import streamlit as st
import pandas as pd
from utils.storage import load_disease_info

st.set_page_config(
    page_title="📚 Disease Library",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Disease Knowledge Base")
st.markdown("Comprehensive information about plant diseases, symptoms, treatments, and prevention strategies.")

st.markdown("---")

# Load disease database
disease_db = load_disease_info()

# ============================================================================
# SEARCH & FILTER
# ============================================================================
col_search, col_sort = st.columns([3, 1])

with col_search:
    search_query = st.text_input(
        "🔍 Search diseases...",
        placeholder="e.g., Leaf Blight, Powdery Mildew, Rust..."
    )

with col_sort:
    sort_by = st.selectbox(
        "Sort by:",
        ["Most Common", "A-Z", "Newest"]
    )

# Filter diseases
if search_query:
    filtered_diseases = {
        k: v for k, v in disease_db.items()
        if search_query.lower() in k.lower() or 
           search_query.lower() in (v.get('description', '')).lower()
    }
else:
    filtered_diseases = disease_db

# Sort
if sort_by == "A-Z":
    filtered_diseases = dict(sorted(filtered_diseases.items()))
elif sort_by == "Most Common":
    # None for now, would need usage stats
    pass

st.markdown("---")

# ============================================================================
# DISEASE CATEGORIES
# ============================================================================
st.subheader("🗂️ Browse by Category")

categories = {
    "🦠 Fungal Diseases": "fungal",
    "🦟 Bacterial Diseases": "bacterial",
    "🦠 Viral Diseases": "viral",
    "🐛 Pest Damages": "pest",
    "😷 Physiological": "physiological",
}

tabs = st.tabs([cat for cat in categories.keys()])

for tab, (category_name, category_key) in zip(tabs, categories.items()):
    with tab:
        # Filter diseases by category
        category_diseases = {
            k: v for k, v in filtered_diseases.items()
            if v.get('category', '').lower() == category_key.lower()
        } if filtered_diseases else {}
        
        if not category_diseases:
            # Show sample diseases for each category
            samples = {
                "fungal": ["Leaf Spot", "Powdery Mildew", "Leaf Blight"],
                "bacterial": ["Bacterial Wilt", "Bacterial Leaf Scorch", "Bacterial Spot"],
                "viral": ["Mosaic Virus", "Leaf Curl Virus", "Stripe Virus"],
                "pest": ["Aphids", "Spider Mites", "Whitefly"],
                "physiological": ["Nutrient Deficiency", "Water Stress", "Salt Damage"],
            }
            
            st.info(f"📚 Sample diseases in {category_name}")
            cols = st.columns(3)
            
            for i, disease in enumerate(samples.get(category_key, [])[:6]):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style='background: white; padding: 15px; border-radius: 10px; 
                                border-left: 4px solid #3498db;'>
                        <h4 style='margin-top: 0; color: #333;'>{disease}</h4>
                        <p style='color: #666; font-size: 0.9rem; margin-bottom: 10px;'>
                        Click to view details
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Display category diseases
            for disease_name, disease_info in category_diseases.items():
                with st.expander(f"🦠 {disease_name}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {disease_info.get('description', 'No description available')}")
                        
                        if disease_info.get('symptoms'):
                            st.markdown("**Symptoms:**")
                            st.write(disease_info['symptoms'])
                        
                        if disease_info.get('treatment'):
                            st.markdown("**Treatment:**")
                            st.write(disease_info['treatment'])
                        
                        if disease_info.get('prevention'):
                            st.markdown("**Prevention:**")
                            st.write(disease_info['prevention'])
                    
                    with col2:
                        if disease_info.get('severity'):
                            severity_map = {
                                'high': ('🔴 High', 'red'),
                                'medium': ('🟡 Medium', 'orange'),
                                'low': ('🟢 Low', 'green'),
                            }
                            severity_display, color = severity_map.get(disease_info['severity'], ('Unknown', 'gray'))
                            st.markdown(f"**Severity:** {severity_display}")
                        
                        if disease_info.get('host_plants'):
                            st.markdown("**Affects:**")
                            for plant in disease_info['host_plants'][:5]:
                                st.write(f"• {plant}")

st.markdown("---")

# ============================================================================
# COMPLETE DISEASE LIST
# ============================================================================
st.subheader("📋 Complete Disease Database")

if filtered_diseases:
    # Create dataframe for display
    diseases_list = []
    for name, info in filtered_diseases.items():
        diseases_list.append({
            'Disease': name,
            'Type': info.get('category', 'Unknown'),
            'Severity': info.get('severity', 'Unknown'),
            'Description': info.get('description', '')[:50] + '...',
        })
    
    df_diseases = pd.DataFrame(diseases_list)
    
    st.dataframe(
        df_diseases,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Disease": st.column_config.TextColumn(width="medium"),
            "Type": st.column_config.TextColumn(width="small"),
            "Severity": st.column_config.TextColumn(width="small"),
            "Description": st.column_config.TextColumn(width="large"),
        }
    )
    
    # Download button
    csv = df_diseases.to_csv(index=False)
    st.download_button(
        label="📥 Download Disease List",
        data=csv,
        file_name="disease_library.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.info("No diseases found matching your search.")

st.markdown("---")

# ============================================================================
# EDUCATIONAL RESOURCES
# ============================================================================
st.subheader("🎓 Learning Resources")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white;'>
        <h3 style='margin-top: 0;'>📖 Guides</h3>
        <p>Step-by-step guides for disease identification and treatment</p>
        <a href='#' style='color: white; text-decoration: none;'>View Guides →</a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 20px; border-radius: 10px; color: white;'>
        <h3 style='margin-top: 0;'>🎥 Videos</h3>
        <p>Expert videos on disease prevention and management</p>
        <a href='#' style='color: white; text-decoration: none;'>Watch Videos →</a>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 20px; border-radius: 10px; color: white;'>
        <h3 style='margin-top: 0;'>📊 Statistics</h3>
        <p>Data on disease prevalence and seasonal patterns</p>
        <a href='#' style='color: white; text-decoration: none;'>View Stats →</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# QUICK REFERENCE GUIDES
# ============================================================================
st.subheader("⚡ Quick Reference Guides")

col_guide1, col_guide2 = st.columns(2)

with col_guide1:
    with st.expander("🔍 How to Identify a Disease"):
        st.markdown("""
        1. **Observe the plant carefully**
           - Where is the damage located? (leaves, stems, roots)
           - What does it look like? (spots, patches, wilting)
        
        2. **Take note of environmental factors**
           - Temperature and humidity
           - Recent watering patterns
           - Presence of insects
        
        3. **Check multiple plants**
           - Is it spreading?
           - Are nearby plants affected?
        
        4. **Use the AI diagnosis tool**
           - Take a clear photo
           - Upload to get instant diagnosis
           - Confirm findings with consultants
        """)

with col_guide2:
    with st.expander("💊 Treatment Timeline"):
        st.markdown("""
        **Week 1:** Early Action
        - Isolate affected plants
        - Remove infected leaves
        - Improve air circulation
        
        **Week 2-3:** Treatment
        - Apply fungicide/pesticide
        - Follow recommended dosages
        - Repeat as instructed
        
        **Week 4+:** Recovery & Prevention
        - Monitor for regrowth
        - Maintain good sanitation
        - Implement preventive measures
        """)

st.markdown("---")

# ============================================================================
# KEY PREVENTION STRATEGIES
# ============================================================================
st.subheader("🛡️ Universal Prevention Strategies")

prev_col1, prev_col2, prev_col3 = st.columns(3)

with prev_col1:
    st.markdown("""
    ### 💧 Water Management
    - Water early morning
    - Avoid wet foliage
    - Use drip irrigation
    - Monitor soil moisture
    """)

with prev_col2:
    st.markdown("""
    ### 🌾 Cultural Practices
    - Crop rotation
    - Remove dead plants
    - Prune properly
    - Maintain sanitation
    """)

with prev_col3:
    st.markdown("""
    ### 🔬 Chemical Prevention
    - Use certified seeds
    - Apply fungicides preventively
    - Follow label instructions
    - Consult local experts
    """)

st.markdown("---")

st.markdown("""
### ❓ Still Have Questions?
- 💬 **Join Community** - Ask other farmers for advice
- 👨‍🌾 **Contact Expert** - Schedule consultation with pathologist
- 📸 **Use AI Tool** - Upload photo for instant diagnosis
- 📞 **Call Support** - Get help from our team
""")
