"""
HISTORY - Diagnostic History Page
Shows all past diagnoses with visualization
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
from pathlib import Path

from utils.storage import load_history, load_disease_info

st.set_page_config(
    page_title="📜 History",
    page_icon="📜",
    layout="wide",
)

st.title("📜 Diagnosis History")
st.markdown("Track all of your plant disease diagnoses with confidence scores and feedback.")

st.markdown("---")

# Load data
history = load_history()
disease_db = load_disease_info()

if not history:
    st.info("📋 No diagnosis history yet. Start by analyzing your first plant image!")
else:
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Add utility columns
    df['date'] = pd.to_datetime(df['date'])
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')
    df['feedback_emoji'] = df['user_feedback'].apply(
        lambda x: '✅' if x == 'correct' else '❌' if x == 'incorrect' else '❓' if x == 'unsure' else '⏳'
    )
    
    # Sort by date (newest first)
    df = df.sort_values('date', ascending=False)
    
    # ============================================================================
    # SUMMARY STATS
    # ============================================================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Diagnoses", len(df))
    
    with col2:
        confirmed = len(df[df['user_feedback'] == 'correct'])
        st.metric("Confirmed Correct", confirmed)
    
    with col3:
        rejected = len(df[df['user_feedback'] == 'incorrect'])
        st.metric("Reported Incorrect", rejected)
    
    with col4:
        if confirmed + rejected > 0:
            accuracy = (confirmed / (confirmed + rejected)) * 100
            st.metric("Accuracy Rate", f"{accuracy:.1f}%")
        else:
            st.metric("Accuracy Rate", "N/A")
    
    st.markdown("---")
    
    # ============================================================================
    # FILTERS & SEARCH
    # ============================================================================
    col_search, col_filter = st.columns(2)
    
    with col_search:
        search_disease = st.selectbox(
            "Filter by disease:",
            ["All"] + sorted(df['disease'].unique().tolist()),
            key="search_disease"
        )
    
    with col_filter:
        feedback_filter = st.multiselect(
            "Filter by feedback:",
            ["✅ Correct", "❌ Incorrect", "❓ Unsure", "⏳ Pending"],
            default=["✅ Correct", "❌ Incorrect", "❓ Unsure", "⏳ Pending"],
            key="feedback_filter"
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_disease != "All":
        filtered_df = filtered_df[filtered_df['disease'] == search_disease]
    
    # Map feedback to display
    feedback_map = {
        "✅ Correct": "correct",
        "❌ Incorrect": "incorrect",
        "❓ Unsure": "unsure",
        "⏳ Pending": None
    }
    feedback_values = [feedback_map[f] for f in feedback_filter]
    filtered_df = filtered_df[filtered_df['user_feedback'].isin(feedback_values)]
    
    st.markdown("---")
    
    # ============================================================================
    # TABS: TABLE & CHARTS
    # ============================================================================
    tab_table, tab_charts, tab_details = st.tabs(["📋 Table", "📊 Charts", "🔍 Details"])
    
    # TABLE TAB
    with tab_table:
        st.subheader("All Diagnoses")
        
        # Display table
        display_df = filtered_df[['image_name', 'disease', 'confidence', 'date_str', 'user_feedback', 'feedback_emoji']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df = display_df.rename(columns={
            'image_name': 'Image',
            'disease': 'Disease',
            'confidence': 'Confidence',
            'date_str': 'Date & Time',
            'user_feedback': 'Feedback',
            'feedback_emoji': ''
        })
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "": st.column_config.TextColumn(width="small"),
            }
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"diagnosis_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # CHARTS TAB
    with tab_charts:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Diagnoses Over Time")
            df_time = filtered_df.groupby(filtered_df['date'].dt.date).size().reset_index(name='count')
            fig_time = px.line(df_time, x='date', y='count', markers=True,
                             labels={'date': 'Date', 'count': 'Number of Diagnoses'},
                             title="Diagnosis Trend")
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col_chart2:
            st.subheader("Feedback Distribution")
            feedback_counts = filtered_df['feedback_emoji'].value_counts()
            fig_feedback = px.pie(
                values=feedback_counts.values,
                names=['✅ Correct' if x == '✅' else '❌ Incorrect' if x == '❌' else '❓ Unsure' if x == '❓' else '⏳ Pending' for x in feedback_counts.index],
                title="User Feedback",
                color_discrete_sequence=['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
            )
            st.plotly_chart(fig_feedback, use_container_width=True)
        
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            st.subheader("Top 10 Detected Diseases")
            top_diseases = filtered_df['disease'].value_counts().head(10)
            fig_diseases = px.bar(
                x=top_diseases.values,
                y=top_diseases.index,
                orientation='h',
                labels={'x': 'Count', 'y': 'Disease'},
                title="Most Detected Diseases"
            )
            st.plotly_chart(fig_diseases, use_container_width=True)
        
        with col_chart4:
            st.subheader("Average Confidence by Disease")
            avg_conf = filtered_df.groupby('disease')['confidence'].mean().sort_values(ascending=False).head(10)
            fig_conf = px.bar(
                x=avg_conf.index,
                y=avg_conf.values,
                labels={'x': 'Disease', 'y': 'Average Confidence'},
                title="Confidence by Disease",
                color=avg_conf.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_conf, use_container_width=True)
    
    # DETAILS TAB
    with tab_details:
        st.subheader("Detailed Analysis")
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            avg_confidence = filtered_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with col_stat2:
            max_confidence = filtered_df['confidence'].max()
            st.metric("Highest Confidence", f"{max_confidence:.1%}")
        
        with col_stat3:
            min_confidence = filtered_df['confidence'].min()
            st.metric("Lowest Confidence", f"{min_confidence:.1%}")
        
        # By date analysis
        st.markdown("### 📅 Analysis by Date")
        
        df_by_date = filtered_df.groupby(filtered_df['date'].dt.date).agg({
            'disease': 'count',
            'confidence': 'mean',
            'user_feedback': lambda x: (x == 'correct').sum()
        }).reset_index()
        df_by_date.columns = ['Date', 'Total Scans', 'Avg Confidence', 'Confirmed Correct']
        df_by_date['Avg Confidence'] = df_by_date['Avg Confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            df_by_date,
            use_container_width=True,
            hide_index=True
        )
        
        # Disease details
        st.markdown("### 🦠 Disease Breakdown")
        
        disease_stats = filtered_df.groupby('disease').agg({
            'image_name': 'count',
            'confidence': 'mean',
            'user_feedback': lambda x: (x == 'correct').sum()
        }).reset_index()
        disease_stats.columns = ['Disease', 'Times Detected', 'Avg Confidence', 'Confirmed Correct']
        disease_stats['Avg Confidence'] = disease_stats['Avg Confidence'].apply(lambda x: f"{x:.1%}")
        disease_stats = disease_stats.sort_values('Times Detected', ascending=False)
        
        st.dataframe(
            disease_stats,
            use_container_width=True,
            hide_index=True
        )

st.markdown("---")

st.markdown("""
### 💡 How to Use History:
- **Track Progress** - Monitor your diagnosis patterns over time
- **Learn Trends** - Identify which diseases are most common
- **Improve Accuracy** - Feedback helps the AI learn
- **Export Data** - Download your history for record-keeping
- **Share Results** - Use data to inform farming decisions
""")
