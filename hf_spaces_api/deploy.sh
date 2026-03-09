#!/bin/bash
# Deploy to Hugging Face Spaces

echo "🚀 Deploying Plant Disease Detection API to Hugging Face Spaces"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Not a git repository. Please run this from the hf_spaces_api directory"
    exit 1
fi

# Check if remote exists
if ! git remote get-url origin &>/dev/null; then
    echo "❌ No git remote configured"
    echo "Please create a Hugging Face Space first:"
    echo "1. Go to https://huggingface.co/spaces"
    echo "2. Create new Space: mohamedsamake8322/plant-disease-api"
    echo "3. Copy the git URL and run:"
    echo "   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME.git"
    exit 1
fi

# Add all files
git add .

# Commit
git commit -m "Deploy Plant Disease Detection API

- FastAPI backend with automatic OpenAPI docs
- Metric learning with FAISS for similar image search
- Batch processing support
- Health checks and monitoring
- Docker deployment ready"

# Push to Hugging Face
echo "📤 Pushing to Hugging Face Spaces..."
git push origin main

echo "✅ Deployment complete!"
echo ""
echo "🌐 Your API will be available at:"
echo "https://mohamedsamake8322-plant-disease-api.hf.space"
echo ""
echo "📚 API Documentation:"
echo "https://mohamedsamake8322-plant-disease-api.hf.space/docs"
echo ""
echo "🧪 Test the API:"
echo "python test_api.py https://mohamedsamake8322-plant-disease-api.hf.space"