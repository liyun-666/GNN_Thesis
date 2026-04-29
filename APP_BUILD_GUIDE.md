# APP Build Guide

This project now includes:
- Desktop app (recommended): `desktop_app_v2.py`
- Mobile-style client under `mobile_app_pwa/`

## Desktop app (single app, no browser required)
Double-click:
- `start_desktop_app.bat`

Desktop app features:
- Account register/login
- Demo workspace (built-in thesis dataset + model)
- Custom DB workspace (user loads sqlite DB/table)
- Format validator (`u,i,b,t` required)
- History eviction policy per user (keep latest N, hard cap 2000)
- Inspector batch scoring + CSV/PNG export
- User behavior visualization (per-user per-item click/fav/cart/buy stats)

## 0) One-click startup (recommended)
Double-click in `D:\GNN_Thesis`:
- `start_all_app.bat`: start API + mobile web app + open browser
- `start_streamlit_app.bat`: start Streamlit full demo app
- `start_api.bat`: API only
- `start_mobile_app.bat`: mobile web frontend only

If you saw:
`ERROR: Error loading ASGI app. Could not import module "api_server".`
it means you ran `uvicorn` outside `D:\GNN_Thesis`.
Use one-click scripts or add:
`--app-dir D:\GNN_Thesis`

## Public sharing (single URL, no extra API config)
Now the mobile web app is hosted by `api_server` at `/web`.
So external users only need one URL:

- `https://<your-tunnel-domain>/web/index.html`

No manual API Base URL input is required in this mode.

## 1) Start backend API
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --app-dir D:\GNN_Thesis
```

## 2) Run the mobile PWA client
```bash
cd mobile_app_pwa
python -m http.server 5173
```
Open: `http://127.0.0.1:5173`

## 3) What the app can do
- Check API health
- Get top-K recommendation
- Toggle `include seen items` (for repurchase inspection)
- Submit click/favorite/cart/buy interaction
- Auto-refresh recommendation after interaction

## 4) Convert to Android/iOS package (optional)
Recommended path: Capacitor wrapper.

```bash
npm init -y
npm install @capacitor/core @capacitor/cli
npx cap init stgnn-mobile com.stgnn.mobile
```
Set webDir to `mobile_app_pwa` in `capacitor.config.ts`, then:
```bash
npm install @capacitor/android
npx cap add android
npx cap copy
npx cap open android
```
Build APK in Android Studio.

## 5) Data/Inspector outputs for thesis
Inspector export files are saved to:
- `artifacts/inspector_exports/*.csv`
- `artifacts/inspector_exports/*_summary.json`
- `artifacts/inspector_exports/*.png`

Use these directly in thesis tables and figures.

## Build installer (desktop)
1. Build exe:
```bash
python -m PyInstaller --noconfirm --clean --noconsole --onefile --name STGNN_Desktop --add-data "artifacts;artifacts" --add-data "paper_alignment.json;." --add-data "PAPER_PDF_LINKS.md;." --add-data "rec_system.db;." desktop_app_v2.py
```
2. Copy exe to installer payload:
```bash
copy /Y dist\STGNN_Desktop.exe installer_payload\STGNN_Desktop.exe
```
3. Compile setup:
```bash
"C:\Users\ASUS\AppData\Local\Programs\Inno Setup 6\ISCC.exe" D:\GNN_Thesis\STGNN_Desktop.iss
```
