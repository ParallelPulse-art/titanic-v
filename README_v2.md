# 🚢 Titanic ML Dashboard v2

A production-grade Streamlit dashboard for Titanic survival prediction with **14 features**, **3 ML models**, and a rich dark nautical UI.

## ✨ Improvements over v1

| Feature | v1 (Decision Tree) | v2 (This App) |
|---|---|---|
| Features used | 3 (class, age, fare) | **14** (+ title, family, cabin, embarkation, etc.) |
| Models | 1 | **3** (Random Forest, Gradient Boosting, Decision Tree) |
| Accuracy | ~80% | **~84.7%** |
| AUC-ROC | — | **0.898** |
| Dataset | 891 rows | **1,309 rows** (full Titanic manifest) |
| Sidebar inputs | 3 | **11** (all passenger attributes) |
| Visualizations | 5 basic charts | **12+ interactive charts** |
| UI | Default Streamlit | **Custom dark nautical theme** |
| Feature engineering | None | Title extraction, family size, cabin deck, age/fare bins |

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python train_model_v2.py   # generates model_v2.pkl
streamlit run titanic_app.py
```

## 📁 Files

| File | Purpose |
|---|---|
| `titanic_app.py` | Main Streamlit dashboard |
| `train_model_v2.py` | Model training script |
| `model_v2.pkl` | Pre-trained models (RF, GB, DT) + encoders |
| `titanic.csv` | Full 1,309-passenger dataset |
| `requirements.txt` | Python dependencies |

## 🌐 Deploy to Streamlit Cloud

1. Push all files to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file as `titanic_app.py`
4. Click **Deploy** — free public URL in ~2 minutes

## 📊 Features Used

- Passenger Class, Sex, Age, Title
- Siblings/Spouses, Parents/Children aboard
- Fare, Port of Embarkation
- Family Size, Traveling Alone
- Cabin deck, Has cabin info
- Derived: Age group, Fare group
