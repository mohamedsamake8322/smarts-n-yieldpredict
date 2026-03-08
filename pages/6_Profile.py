"""
PROFILE - User Profile & Settings Page
"""

import streamlit as st
from utils.storage import get_user_stats, create_or_update_user

st.set_page_config(
    page_title="👤 Profile",
    page_icon="👤",
    layout="wide",
)

st.title("👤 My Profile")
st.markdown("Manage your profile, view statistics, and customize settings.")

st.markdown("---")

# ============================================================================
# SESSION STATE FOR PROFILE
# ============================================================================
if "username" not in st.session_state:
    st.session_state.username = "User"
if "country" not in st.session_state:
    st.session_state.country = None

# ============================================================================
# PROFILE MANAGEMENT
# ============================================================================
tab_info, tab_stats, tab_settings, tab_feedback = st.tabs([
    "📋 Profile Info",
    "📊 My Statistics",
    "⚙️ Settings",
    "💬 Feedback"
])

# PROFILE INFO TAB
with tab_info:
    st.subheader("👤 Profile Information")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Profile form
        with st.form("profile_form"):
            name = st.text_input(
                "Full Name",
                value=st.session_state.username,
                help="Your full name"
            )
            
            email = st.text_input(
                "Email Address",
                placeholder="your@email.com"
            )
            
            country = st.selectbox(
                "Country",
                [
                    "Select...",
                    "Senegal",
                    "Mali",
                    "Côte d'Ivoire",
                    "Burkina Faso",
                    "Niger",
                    "Ghana",
                    "Nigeria",
                    "Cameroon",
                    "Kenya",
                    "Tanzania",
                    "Uganda",
                    "Other",
                ],
                index=0 if not st.session_state.country else None
            )
            
            region = st.text_input(
                "Region/State",
                placeholder="e.g., Kundiawa, Kaolack"
            )
            
            farm_size = st.number_input(
                "Farm Size (hectares)",
                min_value=0.1,
                max_value=10000.0,
                value=10.0
            )
            
            crops = st.multiselect(
                "Main Crops You Grow",
                [
                    "Maize",
                    "Rice",
                    "Cassava",
                    "Banana",
                    "Potato",
                    "Tomato",
                    "Onion",
                    "Pepper",
                    "Beans",
                    "Groundnut",
                    "Cotton",
                    "Coffee",
                    "Cocoa",
                    "Sorghum",
                    "Millet",
                ],
                help="Select all crops you cultivate"
            )
            
            experience = st.select_slider(
                "Farming Experience",
                options=["< 1 year", "1-5 years", "5-10 years", "10-20 years", "20+ years"],
                value="5-10 years"
            )
            
            if st.form_submit_button("💾 Save Profile", use_container_width=True, type="primary"):
                st.session_state.username = name
                st.session_state.country = country
                create_or_update_user(name, country)
                st.success("✅ Profile updated successfully!")
    
    with col2:
        st.markdown("""
        ### 🎯 Benefits of Complete Profile:
        - Better recommendations
        - Connect with local farmers
        - Get targeted tips
        - Community reputation
        """)
    
    with col3:
        # Verification badge
        st.markdown("""
        <div style='background: #e8f5e9; padding: 15px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 2rem; margin-bottom: 10px;'>✅</div>
            <div style='font-weight: bold; margin-bottom: 5px;'>Member Since</div>
            <div style='color: #666;'>March 8, 2026</div>
        </div>
        """, unsafe_allow_html=True)

# STATISTICS TAB
with tab_stats:
    st.subheader("📊 My Diagnosis Statistics")
    
    user_stats = get_user_stats(st.session_state.username)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Scans",
            user_stats.get("total_scans", 0),
            "Diagnoses made"
        )
    
    with col2:
        st.metric(
            "Contributions",
            user_stats.get("contributions", 0),
            "Feedback provided"
        )
    
    with col3:
        st.metric(
            "Community Points",
            0,  # Would need additional tracking
            "Earned by helping others"
        )
    
    with col4:
        st.metric(
            "Achievements",
            2,  # Would need achievement system
            "Badges earned"
        )
    
    st.markdown("---")
    
    # Activity chart
    st.markdown("### 📈 My Recent Activity")
    
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Sample data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    scans = [0 if i < 20 else 1 if i < 25 else 2 for i in range(len(dates))]
    
    df_activity = pd.DataFrame({
        'Date': dates,
        'Scans': scans
    })
    
    import plotly.express as px
    fig = px.line(df_activity, x='Date', y='Scans', markers=True,
                 title='My Diagnosis Activity (Last 30 Days)',
                 labels={'Date': 'Date', 'Scans': 'Number of Diagnoses'})
    st.plotly_chart(fig, use_container_width=True)

# SETTINGS TAB
with tab_settings:
    st.subheader("⚙️ Settings & Preferences")
    
    col_settings = st.columns([2, 1])
    
    with col_settings[0]:
        # Notification settings
        st.markdown("### 🔔 Notifications")
        
        st.toggle("Disease alerts", value=True, help="Get notified about new disease threats in your region")
        st.toggle("Community updates", value=True, help="Get notifications about new discussions")
        st.toggle("Expert recommendations", value=False, help="Get personalized farming tips")
        st.toggle("Email notifications", value=False, help="Receive daily or weekly email summaries")
        
        st.markdown("---")
        
        # Privacy settings
        st.markdown("### 🔒 Privacy & Data")
        
        st.toggle("Make profile public", value=True, help="Other farmers can see your profile")
        st.toggle("Show statistics", value=True, help="Share your diagnosis counts publicly")
        st.toggle("Allow analytics", value=True, help="Help improve the platform (anonymous)")
        
        st.markdown("---")
        
        # Diagnosis preferences
        st.markdown("### 📸 Diagnosis Preferences")
        
        confidence_threshold = st.slider(
            "Minimum confidence threshold for suggestions",
            0.0, 1.0, 0.55,
            help="Only show diagnoses with confidence above this level"
        )
        
        top_k = st.slider(
            "Number of similar images to show",
            1, 10, 5
        )
    
    with col_settings[1]:
        st.markdown("### 💾 Account Actions")
        
        if st.button("🔄 Reset History", use_container_width=True):
            if st.confirmation_dialog("Are you sure you want to reset your diagnosis history? This cannot be undone."):
                st.warning("History reset (feature coming soon)")
        
        if st.button("📥 Download My Data", use_container_width=True):
            st.info("Your data export is being prepared...")
    
    st.markdown("---")
    
    # Language
    st.markdown("### 🌍 Language & Region")
    
    language = st.selectbox(
        "Preferred Language",
        ["English", "Français", "Español", "Portuguese", "Swahili", "Arabic"]
    )
    
    timezone = st.selectbox(
        "Timezone",
        ["UTC", "UTC+1 (West Africa)", "UTC+2 (East Africa)", "UTC+3", "UTC+4"]
    )

# FEEDBACK TAB
with tab_feedback:
    st.subheader("💬 Send Feedback")
    st.markdown("Help us improve the platform by sharing your feedback.")
    
    with st.form("feedback_form"):
        feedback_type = st.selectbox(
            "Feedback Type",
            ["Bug Report", "Feature Request", "Suggestion", "Compliment", "Question"]
        )
        
        subject = st.text_input("Subject")
        
        message = st.text_area("Message", height=150, placeholder="Tell us what's on your mind...")
        
        email_contact = st.text_input("Email (optional, for response)")
        
        if st.form_submit_button("📧 Send Feedback", use_container_width=True, type="primary"):
            st.success("✅ Thank you for your feedback! We'll review it shortly.")

st.markdown("---")

# ============================================================================
# COMMUNITY PROFILE
# ============================================================================
st.subheader("👥 Community Profile")

col_comm1, col_comm2 = st.columns(2)

with col_comm1:
    st.markdown("""
    ### 🏆 Your Community Standing
    - **Rank**: Contributor (Next: Expert at 500 points)
    - **Reputation**: 145 points
    - **Helpful Answers**: 12
    - **Badges**: 
      - 🎯 First Scan
      - ⚡ Quick Learner
    """)

with col_comm2:
    st.markdown("""
    ### 🤝 Community Contributions
    - **Forum Posts**: 8
    - **Answers Given**: 5
    - **Images Shared**: 12
    - **Feedback Ratings**: 48
    """)

st.markdown("---")

st.markdown("""
### 🎓 Account Level Progress
Your current level: **Level 2 - Active Contributor**

Progress to Level 3 (Expert): 145/500 points (29% complete)

Ways to earn points:
- 💬 Post in forum: +10 points
- ✅ Get upvote on answer: +5 points
- 📸 Share diagnosis: +2 points
- ✔️ Provide feedback: +1 point
""")

# Logout button
st.markdown("---")

col_logout1, col_logout2 = st.columns([3, 1])

with col_logout2:
    if st.button("🚪 Logout", use_container_width=True):
        st.info("Logout feature coming soon")
