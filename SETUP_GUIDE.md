# S&P 500 Forecasting System Setup Guide

This guide will help you deploy the S&P 500 Stock Forecasting System on a VM server.

## Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- Python 3.8+
- At least 4GB RAM (8GB recommended)
- At least 10GB free disk space
- Internet connection for data download

## Quick Deployment

### Option 1: Automated Deployment

```bash
# Clone the repository
git clone <your-repo-url>
cd ml-summer-project

# Run the automated deployment script
./deploy.sh
```

### Option 2: Manual Deployment

#### Step 1: Install Dependencies

```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Install additional system dependencies
sudo apt install build-essential python3-dev -y
```

#### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

#### Step 3: Test the System

```bash
# Run tests to ensure everything works
python test_forecaster.py
```

#### Step 4: Generate Initial Forecasts

```bash
# Generate forecasts for all S&P 500 stocks
python generate_forecasts.py
```

#### Step 5: Start the Application

```bash
# Start the Flask web server
python main.py
```

## Production Deployment

### Option 1: Systemd Service (Recommended)

1. **Copy the service file:**

```bash
sudo cp sp500-forecaster.service /etc/systemd/system/
```

2. **Update the service file with your actual paths:**

```bash
sudo nano /etc/systemd/system/sp500-forecaster.service
```

3. **Enable and start the service:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable sp500-forecaster
sudo systemctl start sp500-forecaster
```

4. **Check service status:**

```bash
sudo systemctl status sp500-forecaster
```

### Option 2: Screen Session

```bash
# Install screen
sudo apt install screen -y

# Create a new screen session
screen -S sp500-forecaster

# Activate virtual environment and start the app
source .venv/bin/activate
python main.py

# Detach from screen session (Ctrl+A, then D)
# Reattach later with: screen -r sp500-forecaster
```

### Option 3: Nohup

```bash
# Start the application in the background
nohup python main.py > app.log 2>&1 &

# Check if it's running
ps aux | grep python
```

## Monitoring and Maintenance

### Check Application Status

```bash
# Check if the app is running
curl http://localhost:5000

# Check forecast data directory
ls -la forecast_data/

# Check recent forecasts
ls -la forecast_data/predictions/ | head -10
```

### View Logs

```bash
# If using systemd
sudo journalctl -u sp500-forecaster -f

# If using nohup
tail -f app.log
```

### Manual Forecast Update

```bash
# Activate virtual environment
source .venv/bin/activate

# Manually trigger forecast update
python -c "from sp500_forecaster import forecaster; forecaster.update_all_forecasts()"
```

### Check Background Thread Status

```bash
# Check if background thread is running
python -c "from sp500_forecaster import forecaster; print('Background thread active:', forecaster.last_update)"
```

## Troubleshooting

### Common Issues

1. **Memory Issues**

   - Increase swap space: `sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`

2. **YFinance API Limits**

   - The system includes delays between requests to avoid rate limiting
   - If you encounter issues, increase delays in `sp500_forecaster.py`

3. **Model Training Failures**

   - Check available memory: `free -h`
   - Reduce batch size in `sp500_forecaster.py` if needed

4. **Web Interface Not Loading**
   - Check if Flask is running: `curl http://localhost:5000`
   - Check firewall settings: `sudo ufw status`

### Performance Optimization

1. **Reduce Memory Usage**

   - Edit `sp500_forecaster.py` and reduce `batch_size` from 15 to 8
   - Reduce `epochs` from 30 to 20

2. **Faster Processing**

   - Use fewer stocks by editing `SP500_TICKERS` list
   - Reduce training data period from "3y" to "2y"

3. **Disk Space Management**
   - Clean old model files: `rm forecast_data/models/*.h5`
   - Keep only recent forecasts: `find forecast_data/predictions/ -name "*.csv" -mtime +7 -delete`

## Security Considerations

1. **Firewall Configuration**

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 5000/tcp  # Flask app
sudo ufw enable
```

2. **HTTPS Setup (Optional)**

```bash
# Install nginx and certbot
sudo apt install nginx certbot python3-certbot-nginx -y

# Configure nginx reverse proxy
sudo nano /etc/nginx/sites-available/sp500-forecaster
```

3. **Regular Updates**

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

## Monitoring Scripts

### Create a monitoring script

```bash
cat > monitor.sh << 'EOF'
#!/bin/bash
echo "=== S&P 500 Forecaster Status ==="
echo "Time: $(date)"
echo ""

# Check if Flask is running
if curl -s http://localhost:5000 > /dev/null; then
    echo "✅ Flask app is running"
else
    echo "❌ Flask app is not responding"
fi

# Check forecast data
forecast_count=$(ls forecast_data/predictions/*.csv 2>/dev/null | wc -l)
echo "📊 Available forecasts: $forecast_count"

# Check last update
if [ -f forecast_data/last_update.json ]; then
    last_update=$(cat forecast_data/last_update.json | grep -o '"[^"]*"' | tail -1 | tr -d '"')
    echo "🕒 Last update: $last_update"
else
    echo "❌ No update history found"
fi

echo ""
echo "=== System Resources ==="
echo "Memory usage:"
free -h | grep -E "Mem|Swap"

echo ""
echo "Disk usage:"
df -h | grep -E "Filesystem|/$"
EOF

chmod +x monitor.sh
```

## Support

If you encounter issues:

1. Check the logs: `tail -f app.log`
2. Run the test script: `python test_forecaster.py`
3. Check system resources: `htop` or `top`
4. Verify internet connectivity: `ping google.com`

For additional help, check the main README.md file or create an issue in the repository.
