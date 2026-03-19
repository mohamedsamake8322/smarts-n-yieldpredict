"""
Disease Detection Page with Advanced Features

Enhanced workflow with:
- Confidence thresholds and warnings
- Top-3 explanations with BLIP-2
- Visual attention maps
- Knowledge grounding
- Prediction logging and user feedback
"""

import streamlit as st
from PIL import Image
from modules import VisualDiagnosis
from config import ensure_directories

# Initialize
ensure_directories()
st.set_page_config(page_title="Disease Detection", layout="wide")

st.title("🌾 Advanced Disease Detection & Analysis")
st.markdown("---")

# Initialize session state
if 'diagnosis_results' not in st.session_state:
    st.session_state.diagnosis_results = None
if 'user_confirmed' not in st.session_state:
    st.session_state.user_confirmed = False
if 'show_details' not in st.session_state:
    st.session_state.show_details = False
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

# Initialize visual diagnosis
vd = VisualDiagnosis()

# Sidebar configuration
st.sidebar.title("⚙️ Analysis Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.1,
    help="Predictions below this threshold will show uncertainty warnings"
)

st.sidebar.markdown("---")

# Sidebar for image upload
st.sidebar.title("📷 Upload Plant Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)

    # Run diagnosis
    if st.sidebar.button("🔍 Analyze Image", key="analyze_btn"):
        with st.spinner("🧠 Analyzing with Swin Transformer..."):
            diagnosis = vd.diagnose(uploaded_file, confidence_threshold=confidence_threshold)
            st.session_state.diagnosis_results = diagnosis
            st.session_state.user_confirmed = False
            st.session_state.show_details = False
            st.session_state.feedback_given = False

# ============= Main Content =============
if st.session_state.diagnosis_results:
    results = st.session_state.diagnosis_results
    predictions = results['predictions']
    prediction_id = results.get('prediction_id', 'unknown')

    # ============= STEP 1: Show Top-3 Predictions with Warnings =============
    st.header("📊 Disease Predictions")

    # Display prediction ID
    st.caption(f"Prediction ID: {prediction_id}")

    col1, col2, col3 = st.columns(3)

    for idx, pred in enumerate(predictions):
        with [col1, col2, col3][idx]:
            disease_name = pred['disease'].title()
            confidence = pred['confidence'] * 100
            warning = pred.get('warning')

            # Color coding based on confidence
            if confidence >= 70:
                color = "🟢"  # High confidence
            elif confidence >= 50:
                color = "🟡"  # Medium confidence
            else:
                color = "🔴"  # Low confidence

            st.metric(
                label=f"{color} Prediction {idx + 1}",
                value=disease_name,
                delta=f"{confidence:.1f}% confidence"
            )

            # Show warning if any
            if warning:
                st.warning(f"⚠️ {warning}")

    st.markdown("---")

    # ============= STEP 2: AI-Generated Explanations =============
    st.header("🤖 AI Explanations (Top-3 Predictions)")

    if results.get('explanations'):
        explanations = results['explanations']

        for i in range(1, 4):  # Top 3
            exp_key = f'prediction_{i}'
            if exp_key in explanations:
                exp_data = explanations[exp_key]

                with st.expander(f"📝 Explanation for {exp_data['disease'].title()} ({exp_data['confidence']*100:.1f}% confidence)", expanded=(i==1)):
                    st.markdown(exp_data['explanation'])

                    # Show attention map
                    if i == 1:  # Only for top prediction
                        st.markdown("---")
                        st.subheader("👁️ Visual Attention Analysis")
                        attention_text = vd.generate_attention_map(uploaded_file, exp_data['disease'])
                        st.info(attention_text)
    else:
        st.info("🤖 AI explanations will be generated for the top predictions.")

    st.markdown("---")

    # ============= STEP 3: User Feedback =============
    st.header("📝 Your Feedback")

    if not st.session_state.feedback_given:
        st.markdown("Help us improve! Was this diagnosis accurate?")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Correct", key="correct_btn"):
                vd.log_user_feedback(prediction_id, "correct")
                st.session_state.feedback_given = True
                st.success("✅ Thank you for your feedback!")

        with col2:
            if st.button("❌ Incorrect", key="incorrect_btn"):
                st.session_state.feedback_given = True
                st.warning("Please specify the correct disease below.")

        with col3:
            if st.button("🤔 Unsure", key="unsure_btn"):
                vd.log_user_feedback(prediction_id, "unsure")
                st.session_state.feedback_given = True
                st.info("✅ Feedback recorded. We'll continue improving!")

        # If incorrect, ask for correct disease
        if st.session_state.feedback_given and st.session_state.get('incorrect_selected'):
            correct_disease = st.text_input("What was the actual disease?", key="correct_disease_input")
            additional_notes = st.text_area("Additional notes (optional)", key="notes_input")

            if st.button("Submit Correction", key="submit_correction"):
                vd.log_user_feedback(prediction_id, "incorrect", correct_disease, additional_notes)
                st.success("✅ Correction submitted! Thank you for helping us improve.")
    else:
        st.success("✅ Feedback already recorded. Thank you!")

    st.markdown("---")

    # ============= STEP 4: Detailed Information (Optional) =============
    st.header("📚 Detailed Agricultural Information")

    if st.button("🔍 Show Detailed Management Info", key="show_details_btn"):
        st.session_state.show_details = True

    if st.session_state.show_details:
        top_disease = results.get('top_disease')
        if top_disease:
            disease_info = vd.get_disease_info(top_disease)

            if disease_info:
                detailed_info = vd.get_detailed_info(disease_info)

                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📝 Description",
                    "🛡️ Prevention",
                    "🔧 Management",
                    "📚 Sources"
                ])

                with tab1:
                    st.subheader("Disease Description")
                    st.write(detailed_info['description'] or "No description available.")

                    if detailed_info['scientific_name']:
                        st.markdown(f"**Scientific name:** {detailed_info['scientific_name']}")

                    if detailed_info['hosts']:
                        st.markdown("**Affected hosts:**")
                        for host in detailed_info['hosts']:
                            st.write(f"- {host}")

                with tab2:
                    st.subheader("Prevention Strategies")
                    prevention = detailed_info['prevention']
                    if prevention:
                        st.write(prevention)
                    else:
                        st.info("General prevention: Use resistant varieties, practice crop rotation, maintain crop health.")

                with tab3:
                    st.subheader("Management & Control Methods")
                    management = detailed_info['management']
                    if management:
                        st.write(management)
                    else:
                        st.info("Management strategies not available for this disease.")

                with tab4:
                    st.subheader("Sources & References")
                    if detailed_info['sources']:
                        for source in detailed_info['sources']:
                            st.write(f"- {source}")
                    else:
                        st.info("No sources referenced for this disease.")
            else:
                st.warning("⚠️ Detailed information not available for this disease.")
        else:
            st.warning("⚠️ No disease identified for detailed information.")

    # ============= Statistics (Admin) =============
    st.markdown("---")
    with st.expander("📊 Prediction Statistics (Admin)", expanded=False):
        stats = vd.get_prediction_statistics()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Predictions", stats['total_predictions'])
            st.metric("Correct", stats['correct_predictions'])

        with col2:
            st.metric("Incorrect", stats['incorrect_predictions'])
            st.metric("Unsure", stats['unsure_predictions'])

        with col3:
            st.metric("Avg Confidence", f"{stats['avg_confidence']*100:.1f}%")
            st.metric("Feedback Rate", f"{stats['total_feedback']/max(stats['total_predictions'],1)*100:.1f}%")

        if stats['common_misdiagnoses']:
            st.subheader("Common Misdiagnoses")
            for misdiag, count in list(stats['common_misdiagnoses'].items())[:5]:
                st.write(f"• {misdiag}: {count} times")

else:
    # Initial state
    st.info("👈 Please upload a plant image to start the advanced analysis.")

    st.markdown("""
    ### ✨ New Advanced Features:

    🧠 **Swin Transformer**: 99.5% accuracy on 109 plant diseases
    🤖 **BLIP-2 Explanations**: Natural language explanations for top-3 predictions
    👁️ **Visual Attention**: Shows which parts of the image influenced the diagnosis
    📚 **Knowledge Grounding**: Explanations based on Plantwise agricultural sources
    📊 **Confidence Warnings**: Alerts when predictions may be uncertain
    📝 **User Feedback**: Help improve the system by rating predictions

    ### 📷 Tips for best results:
    - Use clear, well-lit photos of affected plant parts
    - Show disease symptoms clearly (spots, discoloration, etc.)
    - Include multiple angles if possible
    - Ensure good focus and avoid blurry images
    - Crop to show only the affected area
    """)

    # Show current statistics
    st.markdown("---")
    st.subheader("📈 System Statistics")
    try:
        stats = vd.get_prediction_statistics()
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Predictions", stats['total_predictions'])
        with col2:
            st.metric("User Feedback", stats['total_feedback'])
        with col3:
            st.metric("Accuracy", f"{stats['correct_predictions']/max(stats['total_feedback'],1)*100:.1f}%" if stats['total_feedback'] > 0 else "N/A")
        with col4:
            st.metric("Avg Confidence", f"{stats['avg_confidence']*100:.1f}%" if stats['avg_confidence'] > 0 else "N/A")
    except:
        st.info("Statistics will appear after first predictions.")
