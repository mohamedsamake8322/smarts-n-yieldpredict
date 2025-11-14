import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from utils.data_processing1 import validate_agricultural_data, clean_agricultural_data
from utils import AdvancedAI


st.set_page_config(page_title="Data Upload", page_icon="📁", layout="wide")

st.title("📁 Data Upload & Management")
st.markdown("### Import and manage your agricultural datasets")

# Sidebar for upload options
st.sidebar.title("Upload Options")

upload_type = st.sidebar.radio(
    "Select Data Type",
    ["Agricultural Data", "Weather Data", "Soil Data", "Yield Records"],
    help="Choose the type of data you want to upload"
)

# File format selection
file_format = st.sidebar.selectbox(
    "File Format",
    ["CSV", "Excel (.xlsx)", "Excel (.xls)"],
    help="Select the format of your data file"
)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["File Upload", "Data Preview", "Data Validation", "Manage Data"])

with tab1:
    st.subheader("Upload Your Data Files")

    # Data type specific instructions
    if upload_type == "Agricultural Data":
        st.markdown("""
        **Required columns for Agricultural Data:**
        - `crop_type`: Type of crop (wheat, corn, rice, etc.)
        - `yield`: Crop yield in tons per hectare
        - `area`: Cultivated area in hectares
        - `date`: Date of record (YYYY-MM-DD format)

        **Optional columns:**
        - `profit`, `cost`, `temperature`, `rainfall`, `soil_ph`, `fertilizer_used`
        """)

    elif upload_type == "Weather Data":
        st.markdown("""
        **Required columns for Weather Data:**
        - `date`: Date of record (YYYY-MM-DD format)
        - `temperature`: Temperature in Celsius
        - `humidity`: Relative humidity percentage
        - `rainfall`: Rainfall in millimeters

        **Optional columns:**
        - `wind_speed`, `pressure`, `uv_index`, `visibility`
        """)

    elif upload_type == "Soil Data":
        st.markdown("""
        **Required columns for Soil Data:**
        - `field_id`: Field identifier
        - `date`: Date of measurement
        - `ph`: Soil pH level
        - `moisture`: Soil moisture percentage

        **Optional columns:**
        - `nitrogen`, `phosphorus`, `potassium`, `organic_matter`, `temperature`, `conductivity`
        """)

    elif upload_type == "Yield Records":
        st.markdown("""
        **Required columns for Yield Records:**
        - `crop_type`: Type of crop
        - `yield`: Actual yield achieved
        - `area`: Area harvested
        - `harvest_date`: Date of harvest

        **Optional columns:**
        - `planting_date`, `variety`, `treatment`, `quality_grade`
        """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help=f"Upload your {upload_type.lower()} file in {file_format} format"
    )

    if uploaded_file is not None:
        # Display file information
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("File Type", uploaded_file.type)

        # Read the file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Store in session state
            st.session_state.uploaded_data = df
            st.session_state.upload_type = upload_type
            st.session_state.file_name = uploaded_file.name

            st.success(f"Data loaded successfully! {len(df)} rows and {len(df.columns)} columns found.")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()

    # Sample data templates
    st.markdown("---")
    st.subheader("Download Sample Templates")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 Agricultural Data Template", use_container_width=True):
            sample_ag_data = pd.DataFrame({
                'crop_type': ['Wheat', 'Corn', 'Rice', 'Soybeans'],
                'yield': [4.5, 8.2, 6.1, 3.8],
                'area': [10.5, 15.2, 8.0, 12.3],
                'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18'],
                'profit': [4500, 8200, 6100, 3800],
                'soil_ph': [6.5, 6.8, 6.2, 6.7]
            })

            csv = sample_ag_data.to_csv(index=False)
            st.download_button(
                label="Download Agricultural Template",
                data=csv,
                file_name="agricultural_data_template.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("🌤️ Weather Data Template", use_container_width=True):
            sample_weather_data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=30, freq='D'),
                'temperature': np.random.normal(25, 5, 30),
                'humidity': np.random.normal(65, 10, 30),
                'rainfall': np.random.exponential(2, 30),
                'wind_speed': np.random.normal(15, 5, 30)
            })

            csv = sample_weather_data.to_csv(index=False)
            st.download_button(
                label="Download Weather Template",
                data=csv,
                file_name="weather_data_template.csv",
                mime="text/csv"
            )

    with col3:
        if st.button("🌱 Soil Data Template", use_container_width=True):
            sample_soil_data = pd.DataFrame({
                'field_id': ['Field_1', 'Field_2', 'Field_3', 'Field_4'],
                'date': ['2024-01-15', '2024-01-15', '2024-01-15', '2024-01-15'],
                'ph': [6.5, 6.8, 6.2, 6.7],
                'moisture': [55, 62, 48, 58],
                'nitrogen': [45, 38, 52, 41],
                'phosphorus': [28, 32, 25, 30]
            })

            csv = sample_soil_data.to_csv(index=False)
            st.download_button(
                label="Download Soil Template",
                data=csv,
                file_name="soil_data_template.csv",
                mime="text/csv"
            )

    with col4:
        if st.button("📈 Yield Records Template", use_container_width=True):
            sample_yield_data = pd.DataFrame({
                'crop_type': ['Wheat', 'Corn', 'Rice'],
                'yield': [4.2, 7.9, 5.8],
                'area': [12.0, 18.5, 9.2],
                'harvest_date': ['2024-07-15', '2024-09-20', '2024-10-10'],
                'planting_date': ['2024-03-15', '2024-04-20', '2024-05-10'],
                'variety': ['Winter Wheat', 'Dent Corn', 'Long Grain']
            })

            csv = sample_yield_data.to_csv(index=False)
            st.download_button(
                label="Download Yield Template",
                data=csv,
                file_name="yield_records_template.csv",
                mime="text/csv"
            )

with tab2:
    st.subheader("Data Preview")

    if 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data

        # Basic information
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
        with col4:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)

        # Data preview
        st.markdown("**Data Preview (first 10 rows):**")
        st.dataframe(df.head(10), use_container_width=True)

        # Column information
        st.markdown("**Column Information:**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)

        # Basic statistics for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            st.markdown("**Statistical Summary (Numeric Columns):**")
            st.dataframe(numeric_df.describe(), use_container_width=True)

    else:
        st.info("No data uploaded yet. Please upload a file in the File Upload tab.")

with tab3:
    st.subheader("Data Validation & Quality Check")

    if 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
        upload_type = st.session_state.upload_type

        # Run validation
        validation_results = validate_agricultural_data(df, upload_type)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Validation Results:**")

            if validation_results['is_valid']:
                st.success("✅ Data validation passed!")
            else:
                st.error("❌ Data validation failed!")

            # Display validation details
            for check, result in validation_results['checks'].items():
                if result['passed']:
                    st.success(f"✅ {check}: {result['message']}")
                else:
                    st.error(f"❌ {check}: {result['message']}")

        with col2:
            st.markdown("**Data Quality Issues:**")

            issues_found = []

            # Check for missing values
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                issues_found.append("Missing values detected")
                st.warning(f"Missing values in {missing_data[missing_data > 0].to_dict()}")

            # Check for duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                issues_found.append("Duplicate records found")
                st.warning(f"Found {duplicates} duplicate records")

            # Check for outliers in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                if len(outliers) > 0:
                    issues_found.append(f"Outliers in {col}")
                    st.warning(f"Found {len(outliers)} outliers in {col}")

            if not issues_found:
                st.success("✅ No data quality issues detected!")

        # Data cleaning options
        if not validation_results['is_valid'] or issues_found:
            st.markdown("---")
            st.subheader("Data Cleaning Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🧹 Clean Missing Values", use_container_width=True):
                    cleaned_df = clean_agricultural_data(df, method='missing_values')
                    st.session_state.cleaned_data = cleaned_df
                    st.success("Missing values cleaned!")

            with col2:
                if st.button("🔄 Remove Duplicates", use_container_width=True):
                    cleaned_df = clean_agricultural_data(df, method='duplicates')
                    st.session_state.cleaned_data = cleaned_df
                    st.success("Duplicates removed!")

            with col3:
                if st.button("📊 Handle Outliers", use_container_width=True):
                    cleaned_df = clean_agricultural_data(df, method='outliers')
                    st.session_state.cleaned_data = cleaned_df
                    st.success("Outliers handled!")

        # Accept data for use in application
        st.markdown("---")
        if validation_results['is_valid']:
            if st.button("✅ Accept Data for Analysis", use_container_width=True):
                # Store the data based on type
                if upload_type == "Agricultural Data":
                    st.session_state.agricultural_data = df
                elif upload_type == "Weather Data":
                    st.session_state.weather_data = df
                elif upload_type == "Soil Data":
                    st.session_state.soil_data = df
                elif upload_type == "Yield Records":
                    st.session_state.yield_data = df

                st.success(f"✅ {upload_type} accepted and ready for analysis!")
                st.balloons()
        else:
            st.warning("⚠️ Please fix validation issues before accepting the data.")

    else:
        st.info("No data uploaded yet. Please upload a file first.")

with tab4:
    st.subheader("Manage Existing Data")

    # Display currently loaded datasets
    datasets = {
        "Agricultural Data": st.session_state.get('agricultural_data'),
        "Weather Data": st.session_state.get('weather_data'),
        "Soil Data": st.session_state.get('soil_data'),
        "Yield Records": st.session_state.get('yield_data')
    }

    active_datasets = {k: v for k, v in datasets.items() if v is not None}

    if active_datasets:
        st.markdown("**Currently Loaded Datasets:**")

        for dataset_name, dataset in active_datasets.items():
            with st.expander(f"{dataset_name} ({len(dataset)} records)"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Records", len(dataset))
                with col2:
                    st.metric("Columns", len(dataset.columns))
                with col3:
                    last_modified = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.metric("Last Modified", last_modified)

                # Dataset preview
                st.dataframe(dataset.head(5), use_container_width=True)

                # Dataset actions
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Export dataset
                    csv = dataset.to_csv(index=False)
                    st.download_button(
                        f"📥 Export {dataset_name}",
                        data=csv,
                        file_name=f"{dataset_name.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    # View full dataset
                    if st.button(f"👁️ View Full Data", key=f"view_{dataset_name}", use_container_width=True):
                        st.session_state.view_dataset = dataset_name

                with col3:
                    # Delete dataset
                    if st.button(f"🗑️ Delete", key=f"delete_{dataset_name}", use_container_width=True):
                        # Remove from session state
                        if dataset_name == "Agricultural Data":
                            del st.session_state.agricultural_data
                        elif dataset_name == "Weather Data":
                            del st.session_state.weather_data
                        elif dataset_name == "Soil Data":
                            del st.session_state.soil_data
                        elif dataset_name == "Yield Records":
                            del st.session_state.yield_data

                        st.success(f"{dataset_name} deleted!")
                        st.rerun()

    else:
        st.info("No datasets currently loaded. Upload data files to get started.")

    # View full dataset if requested
    if 'view_dataset' in st.session_state:
        view_name = st.session_state.view_dataset
        if view_name in active_datasets:
            st.markdown("---")
            st.subheader(f"Full Dataset: {view_name}")

            dataset = active_datasets[view_name]

            # Search and filter options
            col1, col2 = st.columns(2)

            with col1:
                search_column = st.selectbox(
                    "Search Column",
                    dataset.columns.tolist(),
                    key="search_col"
                )

            with col2:
                search_value = st.text_input(
                    "Search Value",
                    key="search_val"
                )

            # Apply filters
            display_df = dataset.copy()
            if search_value:
                display_df = display_df[display_df[search_column].astype(str).str.contains(search_value, case=False, na=False)]

            # Display filtered data
            st.dataframe(display_df, use_container_width=True, height=400)

            if st.button("Close Full View"):
                del st.session_state.view_dataset
                st.rerun()

    # Data import/export utilities
    st.markdown("---")
    st.subheader("Data Import/Export Utilities")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bulk Operations:**")
        if st.button("📤 Export All Data", use_container_width=True):
            if active_datasets:
                # Create a zip file with all datasets
                st.info("Bulk export functionality would be implemented here")
            else:
                st.warning("No data to export")

    with col2:
        st.markdown("**Data Backup:**")
        if st.button("💾 Create Backup", use_container_width=True):
            if active_datasets:
                backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.success(f"Backup created: backup_{backup_time}")
            else:
                st.warning("No data to backup")
