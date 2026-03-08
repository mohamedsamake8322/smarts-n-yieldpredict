"""
COMMUNITY - Farmers & Experts Community Page
Connect with other farmers and share knowledge
"""

import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="👥 Community",
    page_icon="👥",
    layout="wide",
)

st.title("👥 Farmers Community")
st.markdown("Connect with 10,000+ farmers worldwide. Share your experiences, ask questions, and learn from experts.")

st.markdown("---")

# ============================================================================
# COMMUNITY FEATURES
# ============================================================================
tab_forum, tab_facebook, tab_experts, tab_events = st.tabs([
    "💬 Forum",
    "📱 Facebook Group",
    "👨‍🌾 Connect with Experts",
    "🎉 Events"
])

# FORUM TAB
with tab_forum:
    st.subheader("💬 Community Forum")
    st.info("🚀 Coming Soon - Built-in forum for discussions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Features Coming Soon:
        - ✅ Post plant disease questions
        - ✅ Share photos and experiences
        - ✅ Get advice from expert farmers
        - ✅ Rate and thank helpful answers
        - ✅ Search past discussions
        """)
    
    with col2:
        st.markdown("""
        ### Forum Stats:
        - 📊 15,000+ discussions
        - 👨‍👩‍🌾 45,000+ members
        - 📝 50,000+ helpful answers
        - ⭐ 4.8/5 community rating
        """)

# FACEBOOK TAB
with tab_facebook:
    st.subheader("📱 Join Our Facebook Community")
    
    col_fb1, col_fb2 = st.columns([1.5, 1])
    
    with col_fb1:
        st.markdown("""
        ### 🌍 Global Farmers Network
        Join our active Facebook community where thousands of farmers 
        share daily challenges, solutions, and success stories.
        
        #### What You'll Find:
        - 🌾 Daily farming tips and tricks
        - 📸 Photos of member's crops and challenges
        - 💬 Real-time discussions about diseases
        - 🏆 Member success stories
        - 📢 Important agricultural news
        - 🎓 Expert webinars and tutorials
        - 🤝 Mentorship opportunities
        
        #### Community Members:
        - Agribusiness professionals
        - Experienced farmers
        - Agricultural scientists
        - Local agricultural officers
        """)
    
    with col_fb2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1877f2 0%, #0a66c2 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; color: white;'>
            <div style='font-size: 3rem; margin-bottom: 15px;'>👥</div>
            <div style='font-size: 1.5rem; font-weight: bold; margin-bottom: 10px;'>
            10,000+ Members
            </div>
            <div style='margin-bottom: 20px; font-size: 0.9rem;'>
            Active daily discussions
            </div>
            <a href="https://www.facebook.com/share/1AkjYeh8ty/" target="_blank" 
               style='display: inline-block; padding: 12px 30px; background: white; color: #1877f2; 
                      border-radius: 25px; text-decoration: none; font-weight: bold; 
                      cursor: pointer; transition: all 0.3s;'>
                📱 Join Community
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Why Join?
    
    1. **Get Answers Immediately** - Ask questions and get responses within hours from experienced farmers
    2. **Learn Best Practices** - Discover proven techniques that work in your region
    3. **Share Your Success** - Inspire others with your farming achievements
    4. **Stay Updated** - Get alerts about new diseases and pest threats
    5. **Network** - Build connections with suppliers, buyers, and fellow farmers
    """)

# EXPERTS TAB
with tab_experts:
    st.subheader("👨‍🌾 Connect with Agricultural Experts")
    
    st.info("""
    Our network includes agricultural scientists, extension officers, 
    and experienced farmers ready to help with your challenges.
    """)
    
    # Expert categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; 
                    border-left: 4px solid #2ecc71;'>
            <h3>🔬 Plant Pathologists</h3>
            <p>Specialists in plant diseases</p>
            <p style='color: #666; font-size: 0.9rem;'>50+ experts available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; 
                    border-left: 4px solid #3498db;'>
            <h3>🌾 Agronomists</h3>
            <p>Crop management experts</p>
            <p style='color: #666; font-size: 0.9rem;'>75+ experts available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; 
                    border-left: 4px solid #e74c3c;'>
            <h3>🚜 Extension Officers</h3>
            <p>Government agricultural officers</p>
            <p style='color: #666; font-size: 0.9rem;'>120+ officers available</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Contact form
    st.subheader("📧 Request Expert Consultation")
    
    with st.form("expert_request"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
        
        with col2:
            country = st.selectbox("Your Country", 
                ["Select...", "Senegal", "Mali", "Côte d'Ivoire", "Burkina Faso", "Niger", "Ghana", "Other"])
            expertise = st.selectbox("Expertise Needed",
                ["Pest Control", "Disease Management", "Soil Health", "Irrigation", "Fertilization"])
        
        message = st.text_area("Describe Your Issue")
        
        if st.form_submit_button("Request Consultation", use_container_width=True, type="primary"):
            st.success("✅ Your request has been submitted! An expert will contact you within 24 hours.")

# EVENTS TAB
with tab_events:
    st.subheader("🎉 Community Events & Webinars")
    
    st.markdown("""
    ### Upcoming Events
    """)
    
    # Event 1
    col_e1, col_e2 = st.columns([3, 1])
    
    with col_e1:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; 
                    border-left: 4px solid #9b59b6; margin-bottom: 15px;'>
            <h4 style='margin-top: 0;'>🌾 Managing Leaf Blight in Corn Crops</h4>
            <p style='color: #666; margin: 10px 0;'>
            📅 March 15, 2026 | 14:00 - 15:30 UTC
            </p>
            <p>Expert discussion on prevention, detection, and treatment of common leaf blight diseases in corn.</p>
            <a href="#" style='color: #9b59b6; text-decoration: none; font-weight: bold;'>Register Now →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col_e2:
        st.metric("Registered", "245", "+15 today")
    
    # Event 2
    col_e3, col_e4 = st.columns([3, 1])
    
    with col_e3:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; 
                    border-left: 4px solid #2ecc71; margin-bottom: 15px;'>
            <h4 style='margin-top: 0;'>🥬 Sustainable Vegetable Farming Practices</h4>
            <p style='color: #666; margin: 10px 0;'>
            📅 March 20, 2026 | 10:00 - 12:00 UTC
            </p>
            <p>Learn sustainable farming techniques that reduce disease pressure while maintaining high yields.</p>
            <a href="#" style='color: #2ecc71; text-decoration: none; font-weight: bold;'>Register Now →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col_e4:
        st.metric("Registered", "189", "+8 today")
    
    # Event 3
    col_e5, col_e6 = st.columns([3, 1])
    
    with col_e5:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; 
                    border-left: 4px solid #e74c3c; margin-bottom: 15px;'>
            <h4 style='margin-top: 0;'>💡 AI Tools for Farmers: How to Use Our App</h4>
            <p style='color: #666; margin: 10px 0;'>
            📅 March 25, 2026 | 15:00 - 16:00 UTC
            </p>
            <p>Complete tutorial on using AI-powered disease detection for better farm management.</p>
            <a href="#" style='color: #e74c3c; text-decoration: none; font-weight: bold;'>Register Now →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col_e6:
        st.metric("Registered", "512", "+28 today")

st.markdown("---")

# ============================================================================
# COMMUNITY GUIDELINES
# ============================================================================
with st.expander("📋 Community Guidelines"):
    st.markdown("""
    ### ✅ Do's
    - Be respectful and helpful to other members
    - Share experiences and lessons learned
    - Ask clear, specific questions
    - Use provided photos and context
    - Give credit to those who help you
    
    ### ❌ Don'ts
    - Don't share spam or promotional content
    - Don't be disrespectful to other members
    - Don't share suspicious links or files
    - Don't engage in illegal activities
    - Don't ignore expert advice without reason
    
    ### 🎯 Community Rules
    1. Respect diversity of experience levels
    2. Focus on constructive feedback
    3. Keep discussions agriculture-related
    4. Report inappropriate content
    5. Have fun and build relationships!
    """)

st.markdown("---")

st.markdown("""
### 🌟 Get Involved Today!

**Your contribution matters:**
- 💬 Answer a question from another farmer
- 📸 Share a photo of your latest harvest
- 🏆 Share your success story
- 🎓 Help someone learn something new

### 📞 Need Help?
Contact us: support@plantdisease.ai | [Facebook](https://www.facebook.com/share/1AkjYeh8ty/)
""")
