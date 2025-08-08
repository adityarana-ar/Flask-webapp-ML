#!/bin/bash

# Deployment script for S&P 500 Forecasting System
# This script sets up the system on a VM server

echo "🚀 Deploying S&P 500 Forecasting System..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✅ Python 3 and pip3 are available"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Test the system
echo "🧪 Testing the system..."
python test_forecaster.py

if [ $? -eq 0 ]; then
    echo "✅ System test passed!"
    
    # Generate initial forecasts
    echo "📊 Generating initial S&P 500 forecasts..."
    echo "This may take 30-60 minutes for all 100 stocks..."
    python generate_forecasts.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Initial forecasts generated successfully!"
        
        # Start the Flask app
        echo "🌐 Starting Flask web server..."
        echo "The app will be available at http://localhost:5000"
        echo "S&P 500 forecasts will be available at http://localhost:5000/SP500"
        echo ""
        echo "Press Ctrl+C to stop the server"
        echo ""
        
        # Run the Flask app
        python main.py
        
    else
        echo "❌ Failed to generate initial forecasts"
        exit 1
    fi
    
else
    echo "❌ System test failed. Please check the error messages above."
    exit 1
fi
