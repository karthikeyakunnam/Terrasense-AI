# 🚀 TerraSense AI Deployment Guide

This guide provides step-by-step instructions to deploy the **TerraSense AI** Soil Intelligence Dashboard to cloud platforms.

Since this is a **Streamlit** application using spatial libraries (`geopandas` and `rasterio`), it requires both Python packages and system-level C libraries (like `GDAL` and `GEOS`). We have configured both of these dependencies in the repository (`requirements.txt` and `packages.txt`).

---

## 选项 1: Streamlit Community Cloud (Highly Recommended)
Streamlit Community Cloud is the easiest, most optimized, and completely free platform to deploy and host Streamlit applications.

### ⚡ One-Click Deployment Link
Click the button or link below to start the deployment process:

👉 **[Deploy TerraSense AI to Streamlit Cloud](https://share.streamlit.io/deploy?repository=karthikeyakunnam/Terrasense-AI&branch=main&mainModule=dashboard/app.py)**

### 📝 Step-by-Step Instructions:
1. Click the deployment link above, or go to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Log in using your GitHub account associated with **karthikeyakunnam / karthikeyaunnam1364@gmail.com**.
3. If prompted, authorize Streamlit to access your repositories (this is required to access your private repository `Terrasense-AI`).
4. If you used the link above, the fields will be auto-filled. Otherwise, configure them manually:
   - **Repository:** `karthikeyakunnam/Terrasense-AI`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
5. Click **Deploy!**
6. Streamlit Cloud will boot up a container, install the system dependencies from `packages.txt`, install Python libraries from `requirements.txt`, and launch your app.
7. Once deployed, you will get a permanent public URL (e.g., `https://terrasense-ai.streamlit.app`).

---

## 选项 2: Hugging Face Spaces (Alternative Free Tier)
Hugging Face Spaces is another excellent, free platform that natively supports Streamlit.

### 📝 Step-by-Step Instructions:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and log in or create an account.
2. Click **Create new Space**.
3. Enter your space name (e.g., `terrasense-ai`).
4. Select **Streamlit** as the Space SDK.
5. Choose **Public** or **Private** visibility.
6. Click **Create Space**.
7. Connect your GitHub repository:
   - You can add your Hugging Face Space as a second git remote and push to it.
   - Alternatively, you can copy files into the Hugging Face web interface or use GitHub Actions to sync them.

---

## 选项 3: Render (Web Service)
If you want to host on Render as a standard web service:

### 📝 Step-by-Step Settings:
1. Sign in to [Render](https://render.com/).
2. Click **New +** > **Web Service**.
3. Connect your GitHub account and select the `karthikeyakunnam/Terrasense-AI` repository.
4. Configure the service:
   - **Runtime:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Deploy Web Service**.
