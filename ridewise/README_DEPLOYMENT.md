# 🚀 RideWise Streamlit Deployment Guide

## Deploy to Streamlit Cloud

### Prerequisites
1. A GitHub account
2. Your code pushed to a GitHub repository

### Step-by-Step Deployment

#### 1. Push Your Code to GitHub
```bash
cd c:\Users\hamsa\Downloads\Ridewise_Project_2025\ridewise
git init
git add .
git commit -m "Initial commit for RideWise Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ridewise.git
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Select your repository: `YOUR_USERNAME/ridewise`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **Wait for Deployment**
   - Streamlit will install dependencies from `requirements.txt`
   - Your app will be live at: `https://YOUR_USERNAME-ridewise.streamlit.app`

### Important Files for Deployment

- **streamlit_app.py** - Main application file
- **requirements.txt** - Python dependencies
- **.streamlit/config.toml** - Streamlit configuration
- **ridewise_pipeline.pkl** - ML model (optional, will be created if missing)

### Testing Locally Before Deployment

Run this command to test your app locally:
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Troubleshooting

**Issue: Model file not found**
- The app will automatically create a sample model if `ridewise_pipeline.pkl` is missing
- For production, train your model first with `train_and_save.py`

**Issue: Dependencies fail to install**
- Check that all package versions in `requirements.txt` are compatible
- Streamlit Cloud uses Python 3.9+ by default

**Issue: App crashes on startup**
- Check the logs in Streamlit Cloud dashboard
- Ensure all imports are available in `requirements.txt`

### Environment Variables (Optional)

If you need to add secrets or API keys:
1. Go to your app settings in Streamlit Cloud
2. Click "Secrets" in the left sidebar
3. Add your secrets in TOML format

### Custom Domain (Optional)

To use a custom domain:
1. Go to app settings
2. Click "Custom domain"
3. Follow the instructions to configure your DNS

---

## Alternative: Deploy with Docker

If you prefer Docker deployment:

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run:
```bash
docker build -t ridewise .
docker run -p 8501:8501 ridewise
```

---

## Support

For issues or questions:
- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Community: https://discuss.streamlit.io/

**Happy Deploying! 🎉**
