"""
Agricultural Assistant Page

Q&A interface powered by Plantwise knowledge base using FAISS vector search.
"""

import streamlit as st
from modules import AgriculturalAssistant
from config import ensure_directories

# Initialize
ensure_directories()
st.set_page_config(page_title="Agricultural Assistant", layout="wide")

st.title("🌱 Agricultural Assistant")
st.markdown("Get expert advice from our agricultural knowledge base powered by PlantwisePlus.")
st.markdown("---")

# Initialize agricultural assistant
@st.cache_resource
def load_assistant():
    """Load the assistant (cached for performance)."""
    return AgriculturalAssistant()

try:
    aa = load_assistant()
    assistant_ready = True
except Exception as e:
    st.error(f"Failed to load assistant: {e}")
    assistant_ready = False

if assistant_ready:
    # Create tabs for different query types
    tab1, tab2 = st.tabs(["Ask a Question", "Browse Topics"])
    
    with tab1:
        st.header("❓ Ask the Assistant")
        st.markdown("Ask questions about pest and disease management, prevention strategies, or crop care.")
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., How to control bean bruchid? or What are the symptoms of maize rust?",
            key="query_input"
        )
        
        # Number of sources slider
        col1, col2 = st.columns([3, 1])
        with col2:
            top_k = st.slider("Sources to check:", 1, 10, 3, key="sources_slider")
        
        # Submit query
        if st.button("🔍 Search", key="search_btn"):
            if query.strip():
                with st.spinner("Searching knowledge base..."):
                    response = aa.generate_response(query, top_k=top_k)
                    
                    # Display answer
                    st.success("Answer found!")
                    st.markdown("---")
                    
                    st.subheader("📝 Response")
                    st.write(response['answer'])
                    
                    # Display sources
                    if response['sources']:
                        st.markdown("---")
                        st.subheader("📚 Sources")
                        
                        for i, source in enumerate(response['sources'], 1):
                            with st.expander(f"{i}. {source['title']} (relevance: {source['score']:.2f})"):
                                # Get detailed info for this source
                                content = aa.get_detailed_info(source['filename'])
                                
                                if content:
                                    sections = content.get('sections', {}).get('Table', {})
                                    
                                    # Display available sections
                                    for section_name, section_data in sections.items():
                                        if isinstance(section_data, list) and section_data:
                                            st.write(f"**{section_name}:**")
                                            for item in section_data[1:]:  # Skip section title
                                                if item:
                                                    st.write(f"- {item}")
            else:
                st.warning("Please enter a question.")
    
    with tab2:
        st.header("🌍 Browse Knowledge Base")
        st.markdown("Explore different topics in the agricultural knowledge base.")
        
        # Common topics
        topics = {
            "Prevention": ["How to prevent", "prevention", "avoid"],
            "Pest Control": ["control", "management", "treat"],
            "Crop Diseases": ["disease", "symptom", "blight"],
            "Pest Management": ["pest", "insect", "control"],
            "Storage": ["storage", "store", "preservation"]
        }
        
        col1, col2 = st.columns(2)
        
        selected_topic = col1.selectbox(
            "Select a topic:",
            list(topics.keys()),
            key="topic_select"
        )
        
        search_query = col2.text_input(
            "Or search for a specific pest/disease:",
            placeholder="e.g., bean bruchid, maize rust",
            key="browse_input"
        )
        
        # Browse button
        if col1.button("🔍 Browse", key="browse_btn"):
            if search_query:
                query = search_query
            else:
                query = " ".join(topics[selected_topic])
            
            with st.spinner("Searching..."):
                results = aa.search(query, top_k=10)
                
                if results:
                    st.markdown("---")
                    st.subheader(f"Found {len(results)} relevant entries:")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"{i}. {result['title']} (relevance: {result['score']:.3f})"):
                            content = aa.get_detailed_info(result['filename'])
                            
                            if content:
                                st.markdown(f"**{content.get('title', 'Unknown')}**")
                                
                                sections = content.get('sections', {}).get('Table', {})
                                for section_name, section_data in sections.items():
                                    if isinstance(section_data, list) and len(section_data) > 1:
                                        st.markdown(f"*{section_name}:*")
                                        for item in section_data[1:3]:  # Show first 2 items
                                            if item:
                                                st.write(f"- {item}")
                                        if len(section_data) > 3:
                                            st.caption(f"... and {len(section_data) - 3} more items")
                else:
                    st.info("No results found. Try a different search term.")
    
    # ============= FAQ Section =============
    st.markdown("---")
    st.header("❓ Frequently Asked Questions")
    
    faq_items = {
        "How to prevent crop diseases?": "Use certified disease-free seeds, practice crop rotation, maintain proper plant spacing, and monitor regularly for early symptoms.",
        "What's the best timing for pesticide application?": "Apply pesticides in the early morning or late afternoon when beneficial insects are less active. Always follow label instructions.",
        "How to identify pest infestations early?": "Monitor plants regularly, look for discolored leaves, webbing, holes, or unusual growth patterns. Set up monitoring traps when appropriate.",
        "What are integrated pest management (IPM) practices?": "IPM combines cultural, biological, and chemical approaches. Use resistant varieties, encourage natural enemies, practice sanitation, and use pesticides as a last resort."
    }
    
    for question, answer in faq_items.items():
        with st.expander(f"❓ {question}"):
            st.write(answer)

else:
    st.error("❌ Agricultural Assistant is not available. Please ensure the FAISS index is built. Run `python build_moh_index.py` first.")
    
    st.markdown("""
    ### Setup Instructions:
    1. Ensure you have the Moh JSON files in the `Moh/` directory
    2. Run: `python build_moh_index.py`
    3. Refresh this page once the index is built
    """)
