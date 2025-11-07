"""
Data Advisor v18.4 - PDF met Grafieken en Visuals
Backend met ISO-8601 datums, AI analyse, grafieken in PDF
"""
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
import io
import logging
import os
from groq import Groq
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from io import BytesIO
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Seaborn styling
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq initialized")
except:
    groq_client = None

app = FastAPI(title="Data Advisor API v18.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== UTILITY FUNCTIONS ====================

def load_file(content: bytes, filename: str) -> pd.DataFrame:
    ext = filename.split('.')[-1].lower()
    try:
        if ext == 'csv':
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    return pd.read_csv(io.BytesIO(content), encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("CSV encoding error")
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported: {ext}")
    except Exception as e:
        raise ValueError(f"Load error: {str(e)[:100]}")

def safe_json_convert(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        try:
            pd.to_datetime(df[col], errors='coerce')
            if df[col].notna().sum() > len(df) * 0.5:
                return col
        except:
            continue
    return None

def build_groq_prompt(df: pd.DataFrame) -> str:
    """Behouden originele Groq prompt"""
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    missing_total = int(df.isnull().sum().sum())
    missing_pct = (missing_total / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0
    
    stats_lines = []
    if numeric:
        for col in numeric[:5]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                stats_lines.append(f"• {col}: μ={mean_val:.2f}, σ={std_val:.2f}, CV={cv:.1f}%")
    
    corr_lines = []
    if len(numeric) > 1:
        corr_matrix = df[numeric].corr().abs()
        correlations = []
        for i in range(len(numeric)):
            for j in range(i+1, len(numeric)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val) and corr_val > 0.3:
                    correlations.append((numeric[i], numeric[j], corr_val))
        correlations.sort(key=lambda x: x[2], reverse=True)
        for col1, col2, corr_val in correlations[:5]:
            strength = "zeer sterk" if corr_val > 0.8 else "sterk" if corr_val > 0.6 else "matig"
            corr_lines.append(f"• {col1} ↔ {col2}: r={corr_val:.2f} ({strength})")
    
    return f"""Je bent een senior data-analist. Analyseer deze dataset in het Nederlands.

📊 DATASET: {len(df):,} records × {len(df.columns)} kolommen
Compleetheid: {100-missing_pct:.1f}%

📈 STATISTIEKEN:
{chr(10).join(stats_lines) if stats_lines else 'Geen numerieke data'}

🔗 CORRELATIES:
{chr(10).join(corr_lines) if corr_lines else '• Geen sterke correlaties'}

PARAMETERS:
• Taal: Nederlands
• Lengte: 4 uitgebreide bullets (elk 2-3 zinnen)
• Stijl: Zakelijk, analytisch, direct bruikbaar voor besluitvorming
• Focus: Concrete, meetbare inzichten
• Vermijd: Vage uitspraken zoals "de data lijkt goed" of "meer onderzoek nodig"
• Na elke bullet een enter.
• Ik wil alle 4 de bullets dus: Kernbevinding, Datakwaliteit, Aanbeveling, Business Waarde

VERPLICHTE STRUCTUUR:

1. KERNBEVINDING (belangrijkste patroon/trend):
   → Noem de meest opvallende correlatie, trend of afwijking met exacte cijfers
   → Verklaar waarom dit belangrijk is voor het bedrijf
   → Voorbeeld: "Unit Price en Total Revenue correleren 0.93, wat betekent dat een prijsverhoging van 10% direct €X extra omzet kan opleveren"

2. DATAKWALITEIT (concrete issues + impact):
   → Als missing values: geef percentage + welke kolommen kritiek zijn
   → Als outliers: specificeer aantal + mogelijke oorzaak
   → Geef ALTIJD een praktisch verbeteradvies (bijv. "verwijder rijen met >20% missing")
   → Als data compleet is: benoem welke voorbereidingsstappen wél nodig zijn (normalisatie, encoding, etc.)

3. AANBEVELING (concrete vervolgstappen):
   → Koppel ELKE aanbeveling aan een analytisch doel:
     • Segmentatie-analyse → identificeer winstgevende klantsegmenten
     • Predictief model → voorspel churn/revenue/conversie
     • A/B testing → optimaliseer prijzen/campagnes
   → Geef prioriteit: "Start met X, daarna Y"
   → Voorbeeld: "1) Train een regressiemodel op price→revenue (verwacht R²>0.85). 2) Segmenteer klanten op basis van purchase frequency"

4. BUSINESSWAARDE (ROI-impact):
   → Formuleer één businessgerichte conclusie:
     • Winstoptimalisatie: "Prijsmodel kan omzet verhogen met X%"
     • Klantretentie: "Early churn detection kan €X per jaar besparen"
     • Procesverbetering: "Voorraadoptimalisatie reduceert kosten met X%"
   → Link data-inzichten aan KPI's (omzet, churn, conversie, kosten)
   → Wees specifiek: noem verwachte impact in percentages of bedragen waar mogelijk

BELANGRIJK:
• Begin DIRECT met punt 1 (geen inleidende tekst)
• Gebruik bullets (•) voor structuur
• Gebruik meetbare termen: percentages, correlaties, aantallen
• Baseer conclusies ALLEEN op zichtbare data
• Vermijd technisch jargon over statistiek
• Elke zin moet actionable zijn

Begin nu direct met de analyse:"""

def generate_smart_recommendations(df: pd.DataFrame) -> list:
    """Genereer dataset-specifieke aanbevelingen"""
    recommendations = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    missing_total = df.isnull().sum().sum()
    date_col = detect_date_column(df)
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if corr_matrix.iloc[i, j] > 0.7:
                    high_corr.append((numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j]))
        
        if high_corr:
            top_pair = max(high_corr, key=lambda x: x[2])
            recommendations.append(
                f"• CORRELATIE ANALYSE: {top_pair[0]} en {top_pair[1]} hebben een zeer sterke correlatie "
                f"({top_pair[2]:.2f}). Onderzoek of deze relatie causaal is en gebruik dit voor predictieve modellen."
            )
        else:
            recommendations.append(
                f"• CORRELATIE ANALYSE: Er zijn {len(numeric_cols)} numerieke variabelen zonder sterke correlaties. "
                "Voer feature engineering uit om nieuwe gecombineerde features te creëren."
            )
    
    if missing_total > 0:
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        worst_col = df.isnull().sum().idxmax()
        worst_pct = (df[worst_col].isnull().sum() / len(df) * 100)
        
        if worst_pct > 50:
            recommendations.append(
                f"• DATA CLEANING: Kolom '{worst_col}' heeft {worst_pct:.1f}% missing values. "
                f"Overweeg deze kolom te verwijderen of gebruik advanced imputation (KNN, MICE)."
            )
        else:
            recommendations.append(
                f"• DATA CLEANING: {len(cols_with_missing)} kolommen hebben missing values. "
                f"Gebruik median imputation voor numerieke data en mode voor categorische data."
            )
    else:
        recommendations.append(
            "• DATA KWALITEIT: ✅ Dataset heeft geen missing values - klaar voor directe analyse en modellering."
        )
    
    if len(numeric_cols) >= 2:
        target_candidates = []
        for col in numeric_cols:
            cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
            target_candidates.append((col, cv))
        
        best_target = max(target_candidates, key=lambda x: x[1])[0]
        feature_cols = [c for c in numeric_cols if c != best_target][:3]
        
        recommendations.append(
            f"• PREDICTIEF MODEL: Train een Random Forest regressiemodel met '{best_target}' als target "
            f"en features: {', '.join(feature_cols)}. Verwachte R² > 0.75 op basis van correlaties."
        )
    
    if date_col:
        recommendations.append(
            f"• TIME SERIES ANALYSE: Dataset bevat datumkolom '{date_col}'. "
            f"Voer trend- en seasonaliteitsanalyse uit, en train een forecasting model (ARIMA/ETS)."
        )
    
    if len(numeric_cols) >= 2:
        recommendations.append(
            f"• FEATURE ENGINEERING: Creëer ratio's tussen numerieke features "
            f"(bijv. {numeric_cols[0]}/{numeric_cols[1]}) om verborgen patronen te ontdekken."
        )
    
    return recommendations[:5]

def get_ai_insights(df: pd.DataFrame) -> str:
    """Haal AI insights op via Groq"""
    if not groq_client:
        return "AI analyse niet beschikbaar (Groq API key ontbreekt)"
    
    try:
        if len(df) > 1000:
            df = df.sample(n=1000, random_state=42)
        
        prompt = build_groq_prompt(df)
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Je bent een senior data-analist."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.8,
            max_tokens=800
        )
        
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"AI analyse fout: {str(e)[:100]}"

def create_distribution_plot(df: pd.DataFrame, numeric_cols: list) -> BytesIO:
    """📊 Creëer distributie plots voor numerieke kolommen"""
    n_cols = min(4, len(numeric_cols))
    if n_cols == 0:
        return None
    
    fig, axes = plt.subplots(1, n_cols, figsize=(12, 3))
    if n_cols == 1:
        axes = [axes]
    
    for idx, col in enumerate(numeric_cols[:n_cols]):
        data = df[col].dropna()
        axes[idx].hist(data, bins=30, color='#667eea', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Waarde', fontsize=8)
        axes[idx].set_ylabel('Frequentie', fontsize=8)
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def create_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> BytesIO:
    """🔥 Creëer correlatie heatmap"""
    if len(numeric_cols) < 2:
        return None
    
    # Neem max 10 kolommen voor leesbaarheid
    cols_to_plot = numeric_cols[:10]
    corr_matrix = df[cols_to_plot].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='RdYlGn', 
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Correlatie Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def create_missing_data_plot(df: pd.DataFrame) -> BytesIO:
    """📉 Visualiseer missing data"""
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    if len(missing_counts) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(8, max(3, len(missing_counts) * 0.3)))
    
    colors_list = ['#E53E3E' if x > len(df) * 0.5 else '#F6AD55' if x > len(df) * 0.2 else '#48BB78' 
                   for x in missing_counts.values]
    
    ax.barh(missing_counts.index, missing_counts.values, color=colors_list, edgecolor='black')
    ax.set_xlabel('Aantal Missing Values', fontsize=10, fontweight='bold')
    ax.set_title('Missing Data per Kolom', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def create_boxplot(df: pd.DataFrame, numeric_cols: list) -> BytesIO:
    """📦 Creëer boxplots voor outlier detectie"""
    n_cols = min(4, len(numeric_cols))
    if n_cols == 0:
        return None
    
    fig, axes = plt.subplots(1, n_cols, figsize=(12, 3))
    if n_cols == 1:
        axes = [axes]
    
    for idx, col in enumerate(numeric_cols[:n_cols]):
        data = df[col].dropna()
        axes[idx].boxplot(data, vert=True, patch_artist=True,
                         boxprops=dict(facecolor='#667eea', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))
        axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Waarde', fontsize=8)
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

# ==================== API ENDPOINTS ====================

@app.get("/")
def root():
    return {
        "status": "online",
        "version": "18.4",
        "features": ["upload", "insights", "forecasting", "ml-training", "pdf-with-charts"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "version": "18.4"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = load_file(content, file.filename)
        df = safe_json_convert(df)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_col = detect_date_column(df)
        
        info = {
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing": {col: int(df[col].isnull().sum()) for col in df.columns},
            "missing_total": int(df.isnull().sum().sum()),
            "preview": df.head(10).fillna("").to_dict('records'),
            "numeric_stats": {},
            "date_column": date_col,
            "has_timeseries": date_col is not None
        }
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                info["numeric_stats"][col] = {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "count": int(len(col_data))
                }
        
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().fillna(0)
            info["correlation"] = {
                "columns": numeric_cols,
                "matrix": corr.values.tolist()
            }
        
        return JSONResponse(content=info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)[:200])

@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    date_column: str = Form(...),
    target_column: str = Form(...),
    periods: int = Form(30),
    model_type: str = Form("auto")
):
    try:
        content = await file.read()
        df = load_file(content, file.filename)
        
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column, target_column])
        df = df.sort_values(date_column).set_index(date_column)
        
        series = df[target_column].astype(float)
        
        if len(series) < 10:
            raise ValueError("Need ≥10 data points")
        
        if model_type == "auto":
            model_type = "ets" if len(series) > 24 else "arima"
        
        if model_type == "ets":
            try:
                model = ExponentialSmoothing(
                    series,
                    seasonal_periods=min(12, len(series)//3) if len(series) > 24 else None,
                    trend='add',
                    seasonal='add' if len(series) > 24 else None
                ).fit()
                forecast_values = model.forecast(periods)
            except:
                model = ExponentialSmoothing(series, trend='add').fit()
                forecast_values = model.forecast(periods)
        else:
            try:
                model = ARIMA(series, order=(1, 1, 1)).fit()
                forecast_values = model.forecast(steps=periods)
            except:
                last_val = series.iloc[-1]
                forecast_index = pd.date_range(
                    start=series.index[-1], 
                    periods=periods+1, 
                    freq=pd.infer_freq(series.index) or 'D'
                )[1:]
                forecast_values = pd.Series([last_val] * periods, index=forecast_index)
        
        freq = pd.infer_freq(series.index) or 'D'
        forecast_dates = pd.date_range(
            start=series.index[-1], 
            periods=periods + 1, 
            freq=freq
        )[1:]
        
        std_error = series.std()
        lower_bound = forecast_values - 1.96 * std_error
        upper_bound = forecast_values + 1.96 * std_error
        
        result = {
            "model": model_type.upper(),
            "target": target_column,
            "periods": periods,
            "historical": {
                "dates": [d.isoformat() for d in series.index],
                "values": series.tolist()
            },
            "forecast": {
                "dates": [d.isoformat() for d in forecast_dates],
                "values": forecast_values.tolist(),
                "lower_bound": lower_bound.tolist(),
                "upper_bound": upper_bound.tolist()
            },
            "metrics": {
                "mean_historical": float(series.mean()),
                "std_historical": float(series.std()),
                "mean_forecast": float(forecast_values.mean()),
                "trend": "up" if forecast_values.iloc[-1] > series.iloc[-1] else "down",
                "change_pct": float(((forecast_values.iloc[-1] / series.iloc[-1]) - 1) * 100)
            }
        }
        
        logger.info(f"✅ Forecast: {len(series)} hist + {periods} forecast")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"❌ Forecast error: {e}")
        raise HTTPException(status_code=400, detail=str(e)[:200])

@app.post("/insights")
async def insights(file: UploadFile = File(...)):
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq unavailable")
    
    try:
        content = await file.read()
        df = load_file(content, file.filename)
        df = safe_json_convert(df)
        
        insights_text = get_ai_insights(df)
        
        return JSONResponse(content={
            "insights": insights_text,
            "model": "Groq (llama-3.3-70b)",
            "rows_analyzed": min(len(df), 1000)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target: str = Form(...),
    model_type: str = Form("regression"),
    test_size: float = Form(0.2),
    n_estimators: int = Form(100),
    max_depth: Optional[int] = Form(None)
):
    try:
        content = await file.read()
        df = load_file(content, file.filename)
        df = safe_json_convert(df)
        
        if target not in df.columns:
            raise ValueError(f"Column '{target}' not found")
        
        df = df.dropna(subset=[target])
        X = df.drop(columns=[target])
        y = df[target]
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols].copy()
        
        if len(X.columns) == 0:
            raise ValueError("No numeric features")
        
        X = X.fillna(X.median())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if model_type == "regression":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            result = {
                "model": "Random Forest Regressor",
                "metrics": {
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "R2": float(r2_score(y_test, y_pred)),
                    "MAE": float(mean_absolute_error(y_test, y_pred))
                },
                "feature_importance": [
                    {"feature": col, "importance": float(imp)}
                    for col, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:10]
                ]
            }
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            result = {
                "model": "Random Forest Classifier",
                "metrics": {"Accuracy": float(accuracy_score(y_test, y_pred))}
            }
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)[:200])

@app.post("/generate-pdf")
async def generate_pdf(file: UploadFile = File(...)):
    """🔥 PDF met AI analyse + GRAFIEKEN"""
    try:
        logger.info("📄 Generating PDF with charts...")
        content = await file.read()
        df = load_file(content, file.filename)
        
        # Convert datetime
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        original_size = len(df)
        sample_size = min(len(df), 500)
        df_sample = df.sample(n=sample_size, random_state=42) if len(df) > 500 else df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Haal AI insights op
        ai_insights = get_ai_insights(df_sample)
        
        # Genereer aanbevelingen
        smart_recommendations = generate_smart_recommendations(df)
        
        # Genereer grafieken
        dist_plot = create_distribution_plot(df_sample, numeric_cols)
        corr_heatmap = create_correlation_heatmap(df_sample, numeric_cols)
        missing_plot = create_missing_data_plot(df)
        box_plot = create_boxplot(df_sample, numeric_cols)
        
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer, 
            pagesize=landscape(A4),
            rightMargin=40, leftMargin=40,
            topMargin=50, bottomMargin=40
        )
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=26,
            textColor=colors.HexColor('#1A202C'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2D3748'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#4A5568'),
            alignment=TA_JUSTIFY,
            spaceAfter=10,
            leading=14
        )
        
        story = []
        
        # === COVER PAGE ===
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("📊 Data Advisor", title_style))
        story.append(Paragraph("Professional Data Analysis Report + AI Insights + Visuals", 
            ParagraphStyle('Subtitle', parent=body_style, fontSize=14, alignment=TA_CENTER, textColor=colors.grey)
        ))
        story.append(Spacer(1, 0.5*inch))
        
        # Info table
        info_data = [
            ['Bestandsnaam:', file.filename],
            ['Analyse datum:', pd.Timestamp.now().strftime('%d %B %Y, %H:%M')],
            ['Totaal records:', f"{original_size:,}"],
            ['Geanalyseerd:', f"{sample_size:,} records"],
            ['Kolommen:', f"{len(df.columns)}"],
            ['Compleetheid:', f"{100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%"]
        ]
        
        info_table = Table(info_data, colWidths=[2.5*inch, 5*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#4A5568')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (1, 0), (1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(info_table)
        story.append(PageBreak())
        
        # === AI ANALYSE ===
        story.append(Paragraph("1. AI Data Analyse (Groq llama-3.3-70b)", heading_style))
        
        insight_paragraphs = ai_insights.split('\n\n')
        for para in insight_paragraphs:
            if para.strip():
                safe_para = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(safe_para, body_style))
                story.append(Spacer(1, 0.1*inch))
        
        story.append(PageBreak())
        
        # === DISTRIBUTIE GRAFIEKEN ===
        if dist_plot:
            story.append(Paragraph("2. Distributie Analyse", heading_style))
            story.append(Paragraph(
                "De onderstaande histogrammen tonen de verdeling van de belangrijkste numerieke variabelen. "
                "Let op scheefheid, outliers en normaliteit van de data.",
                body_style
            ))
            story.append(Spacer(1, 0.1*inch))
            
            img = Image(dist_plot, width=9*inch, height=2.5*inch)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
        
        # === CORRELATIE HEATMAP ===
        if corr_heatmap:
            story.append(Paragraph("3. Correlatie Analyse", heading_style))
            story.append(Paragraph(
                "De heatmap toont de correlatie tussen numerieke variabelen. "
                "Donkergroen = sterke positieve correlatie, rood = negatieve correlatie.",
                body_style
            ))
            story.append(Spacer(1, 0.1*inch))
            
            img = Image(corr_heatmap, width=6*inch, height=5*inch)
            story.append(img)
            story.append(PageBreak())
        
        # === MISSING DATA ===
        if missing_plot:
            story.append(Paragraph("4. Data Quality: Missing Values", heading_style))
            story.append(Paragraph(
                "Deze grafiek toont het aantal ontbrekende waarden per kolom. "
                "Rood = kritisch (>50%), oranje = matig (20-50%), groen = acceptabel (<20%).",
                body_style
            ))
            story.append(Spacer(1, 0.1*inch))
            
            img = Image(missing_plot, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
        
        # === BOXPLOTS ===
        if box_plot:
            story.append(Paragraph("5. Outlier Detectie (Boxplots)", heading_style))
            story.append(Paragraph(
                "Boxplots helpen bij het identificeren van uitschieters. "
                "Punten buiten de 'whiskers' zijn potentiële outliers die verder onderzoek vereisen.",
                body_style
            ))
            story.append(Spacer(1, 0.1*inch))
            
            img = Image(box_plot, width=9*inch, height=2.5*inch)
            story.append(img)
            story.append(PageBreak())
        
        # === STATISTIEKEN TABEL ===
        story.append(Paragraph("6. Numerieke Statistieken", heading_style))
        
        if numeric_cols:
            stats_data = [['Kolom', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Missing']]
            
            for col in numeric_cols[:15]:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    stats_data.append([
                        col[:25],
                        f"{col_data.mean():.2f}",
                        f"{col_data.median():.2f}",
                        f"{col_data.std():.2f}",
                        f"{col_data.min():.2f}",
                        f"{col_data.max():.2f}",
                        str(df[col].isnull().sum())
                    ])
            
            stats_table = Table(stats_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 0.8*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A5568')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
            ]))
            story.append(stats_table)
        
        story.append(PageBreak())
        
        # === AANBEVELINGEN ===
        story.append(Paragraph("7. Dataset-Specifieke Aanbevelingen", heading_style))
        story.append(Paragraph(
            "Deze aanbevelingen zijn gegenereerd op basis van de structuur en patronen in jouw specifieke dataset:",
            body_style
        ))
        story.append(Spacer(1, 0.1*inch))
        
        for rec in smart_recommendations:
            safe_rec = rec.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe_rec, body_style))
            story.append(Spacer(1, 0.05*inch))
        
        # === FOOTER ===
        story.append(Spacer(1, 1*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(
            f"Data Advisor v18.4 | Gegenereerd op {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | "
            f"Powered by Groq AI, FastAPI, scikit-learn, matplotlib & ReportLab",
            footer_style
        ))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        
        logger.info("✅ PDF with charts generated successfully")
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=data_advisor_visual_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ PDF error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF failed: {str(e)[:200]}")

if __name__ == "__main__":
    import os

    # Print startinfo
    print("\n" + "="*70)
    print("📊 DATA ADVISOR API v18.4 - PDF WITH CHARTS & VISUALS")
    print("="*70)
    print("🌐 Server:     http://localhost:8000")
    print("🤖 AI:         Groq (llama-3.3-70b)")
    print("📊 Grafieken:  matplotlib + seaborn")
    print("📄 PDF:        AI + statistieken + 4 grafieken")
    print("="*70 + "\n")

    # Start lokaal met uvicorn, Render gebruikt Gunicorn
    if os.environ.get("RENDER") == "true":
        from app import app  # Render start dit automatisch
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

