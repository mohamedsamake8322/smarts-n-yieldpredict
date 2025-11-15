#!/usr/bin/env python3
"""
Smart Fertilizer Application Launcher

This script provides multiple ways to run the Smart Fertilizer application:
1. Streamlit web interface (default)
2. FastAPI backend server
3. Command-line interface for batch processing
4. Development mode with hot reloading

Usage:
    python run_fertilizer.py                    # Run Streamlit app
    python run_fertilizer.py --mode api         # Run FastAPI server
    python run_fertilizer.py --mode cli         # Command-line interface
    python run_fertilizer.py --mode dev         # Development mode
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_streamlit_app():
    """Run the Streamlit web application"""
    print("🌾 Starting Smart Fertilizer Streamlit Application...")
    print("🌐 Access the application at: http://localhost:5000")
    print("🔄 Use Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py", 
            "--server.port", "5000",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Smart Fertilizer application stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit application: {e}")
        sys.exit(1)

def run_fastapi_server():
    """Run the FastAPI backend server"""
    print("🚀 Starting Smart Fertilizer FastAPI Server...")
    print("🌐 API documentation available at: http://localhost:8000/docs")
    print("🔄 Use Ctrl+C to stop the server")
    
    try:
        import uvicorn
        from api.main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 FastAPI server stopped.")
    except ImportError:
        print("❌ Error: uvicorn not available. Install with: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running FastAPI server: {e}")
        sys.exit(1)

def run_cli_interface():
    """Run command-line interface for batch processing"""
    print("💻 Smart Fertilizer Command-Line Interface")
    print("=" * 50)
    
    try:
        from core.smart_fertilizer_engine import SmartFertilizerEngine
        from api.models import SoilAnalysis, CropSelection
        from core.regional_context import RegionalContext
        
        # Initialize components
        engine = SmartFertilizerEngine()
        regional_context = RegionalContext()
        
        print("✅ Smart Fertilizer engine initialized successfully!")
        
        # Interactive CLI
        while True:
            print("\nAvailable commands:")
            print("1. Generate recommendation")
            print("2. List available regions")
            print("3. List available crops")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                generate_cli_recommendation(engine, regional_context)
            elif choice == "2":
                list_regions(regional_context)
            elif choice == "3":
                list_crops(engine)
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
                
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 CLI interface stopped.")

def generate_cli_recommendation(engine, regional_context):
    """Generate a fertilizer recommendation via CLI"""
    try:
        print("\n📋 Fertilizer Recommendation Generator")
        print("-" * 40)
        
        # Get basic inputs
        region = input("Enter region (e.g., nigeria, kenya, ghana): ").strip().lower()
        crop_type = input("Enter crop type (e.g., maize, rice, wheat): ").strip().lower()
        variety = input("Enter crop variety: ").strip()
        area = float(input("Enter farm area (hectares): "))
        target_yield = float(input("Enter target yield (tons/ha): "))
        
        # Get soil analysis inputs
        print("\n🧪 Soil Analysis Data:")
        ph = float(input("Soil pH (3.0-10.0): "))
        organic_matter = float(input("Organic matter (%): "))
        nitrogen = float(input("Available nitrogen (ppm): "))
        phosphorus = float(input("Available phosphorus (ppm): "))
        potassium = float(input("Available potassium (ppm): "))
        cec = float(input("CEC (cmol/kg): "))
        
        # Create objects
        soil_analysis = SoilAnalysis(
            ph=ph,
            organic_matter=organic_matter,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            cec=cec,
            texture="loamy"  # Default
        )
        
        crop_selection = CropSelection(
            crop_type=crop_type,
            variety=variety,
            planting_season="wet_season",
            growth_duration=120
        )
        
        # Get region data
        region_data = regional_context.get_region_data(region)
        
        # Generate recommendation
        print("\n⏳ Generating recommendation...")
        recommendation = engine.generate_recommendation(
            soil_analysis=soil_analysis,
            crop_selection=crop_selection,
            region_data=region_data,
            area_hectares=area,
            target_yield=target_yield
        )
        
        # Display results
        print("\n✅ Recommendation Generated!")
        print("=" * 50)
        print(f"Recommendation ID: {recommendation.recommendation_id}")
        print(f"Expected Yield: {recommendation.expected_yield:.1f} tons/ha")
        print(f"Total Cost: {recommendation.cost_analysis.currency} {recommendation.cost_analysis.total_cost:.2f}")
        print(f"ROI: {recommendation.roi_percentage:.1f}%")
        
        print("\n📊 Nutrient Requirements:")
        print(f"Nitrogen: {recommendation.nutrient_balance.total_n:.1f} kg/ha")
        print(f"Phosphorus: {recommendation.nutrient_balance.total_p:.1f} kg/ha")
        print(f"Potassium: {recommendation.nutrient_balance.total_k:.1f} kg/ha")
        
        print("\n💰 Recommended Fertilizers:")
        for fert in recommendation.recommended_fertilizers:
            print(f"- {fert.name} ({fert.n_content}-{fert.p_content}-{fert.k_content})")
        
        # Option to save
        save_option = input("\nSave recommendation to file? (y/n): ").strip().lower()
        if save_option == 'y':
            save_cli_recommendation(recommendation)
        
    except Exception as e:
        print(f"❌ Error generating recommendation: {e}")

def save_cli_recommendation(recommendation):
    """Save CLI recommendation to file"""
    try:
        from exports.export_utils import ExportUtilities
        
        export_utils = ExportUtilities()
        
        # Create exports directory if it doesn't exist
        os.makedirs("exports_cli", exist_ok=True)
        
        filename = f"exports_cli/recommendation_{recommendation.recommendation_id}"
        
        # Export as JSON
        json_data = export_utils.export_recommendation(recommendation.__dict__, "json")
        with open(f"{filename}.json", "w") as f:
            f.write(json_data)
        
        print(f"✅ Recommendation saved as {filename}.json")
        
    except Exception as e:
        print(f"❌ Error saving recommendation: {e}")

def list_regions(regional_context):
    """List available regions"""
    regions = regional_context.get_available_regions()
    print("\n🌍 Available Regions:")
    print("-" * 25)
    for region in regions:
        print(f"• {region['name']} ({region['key']})")
        print(f"  Climate: {region['climate_type']}")
        print(f"  Major crops: {', '.join(region['major_crops'][:3])}")
        print()

def list_crops(engine):
    """List available crops"""
    crops = engine.get_available_crops()
    print("\n🌱 Available Crops:")
    print("-" * 20)
    for crop in crops:
        print(f"• {crop.title()}")

def run_development_mode():
    """Run in development mode with hot reloading"""
    print("🔧 Starting Smart Fertilizer in Development Mode...")
    print("🔄 Hot reloading enabled")
    print("🌐 Streamlit app: http://localhost:5000")
    print("🚀 FastAPI docs: http://localhost:8000/docs")
    
    try:
        # Start FastAPI in background
        import subprocess
        import threading
        import time
        
        def start_fastapi():
            subprocess.run([
                sys.executable, "-c",
                "import uvicorn; from api.main import app; uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)"
            ])
        
        # Start FastAPI in a separate thread
        api_thread = threading.Thread(target=start_fastapi, daemon=True)
        api_thread.start()
        
        # Wait a moment for FastAPI to start
        time.sleep(3)
        
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py", 
            "--server.port", "5000",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Development mode stopped.")
    except Exception as e:
        print(f"❌ Error in development mode: {e}")
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        "streamlit", "fastapi", "uvicorn", "pandas", "numpy", 
        "plotly", "requests", "reportlab", "openpyxl"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Smart Fertilizer Application Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_fertilizer.py                    # Run Streamlit app
  python run_fertilizer.py --mode api         # Run FastAPI server
  python run_fertilizer.py --mode cli         # Command-line interface
  python run_fertilizer.py --mode dev         # Development mode
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["streamlit", "api", "cli", "dev"],
        default="streamlit",
        help="Application mode (default: streamlit)"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Smart Fertilizer v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies are available!")
        sys.exit(0)
    
    # Display banner
    print("🌾" * 50)
    print("     SMART FERTILIZER - AFRICAN AGRICULTURE")
    print("     Intelligent Fertilizer Recommendations")
    print("🌾" * 50)
    print()
    
    # Check dependencies before running
    if not check_dependencies():
        sys.exit(1)
    
    # Run based on mode
    if args.mode == "streamlit":
        run_streamlit_app()
    elif args.mode == "api":
        run_fastapi_server()
    elif args.mode == "cli":
        run_cli_interface()
    elif args.mode == "dev":
        run_development_mode()

if __name__ == "__main__":
    main()
