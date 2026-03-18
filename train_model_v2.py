"""
Run this ONCE to generate model_v2.pkl before launching the app.
Usage: python train_model_v2.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

df = pd.read_csv("titanic.csv")

# ── Feature Engineering ──
data = df.copy()
data['title'] = data['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
data['title'] = data['title'].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
data['title'] = data['title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})

data['family_size'] = data['sibsp'] + data['parch'] + 1
data['is_alone']    = (data['family_size'] == 1).astype(int)
data['has_cabin']   = data['cabin'].notna().astype(int)
data['cabin_letter']= data['cabin'].str[0].fillna('U')

data['age'] = data.groupby(['pclass','sex','title'])['age'].transform(lambda x: x.fillna(x.median()))
data['age'] = data['age'].fillna(data['age'].median())
data['fare'] = data.groupby('pclass')['fare'].transform(lambda x: x.fillna(x.median()))
data['embarked'] = data['embarked'].fillna('S')

data['age_bin']  = pd.cut(data['age'],  bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
data['fare_bin'] = pd.qcut(data['fare'], q=4, labels=[0,1,2,3]).astype(int)

le_sex   = LabelEncoder()
le_emb   = LabelEncoder()
le_title = LabelEncoder()
le_cabin = LabelEncoder()

data['sex_enc']      = le_sex.fit_transform(data['sex'])
data['embarked_enc'] = le_emb.fit_transform(data['embarked'])
data['title_enc']    = le_title.fit_transform(data['title'])
data['cabin_enc']    = le_cabin.fit_transform(data['cabin_letter'])

FEATURES = ['pclass','sex_enc','age','sibsp','parch','fare','embarked_enc',
            'title_enc','family_size','is_alone','has_cabin','cabin_enc',
            'age_bin','fare_bin']

FEATURE_LABELS = ['Passenger Class','Sex','Age','Siblings/Spouses','Parents/Children',
                  'Fare','Embarkation','Title','Family Size','Traveling Alone',
                  'Has Cabin Info','Cabin Deck','Age Group','Fare Group']

X = data[FEATURES]
y = data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=4, random_state=42)
gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
dt = DecisionTreeClassifier(max_depth=6, min_samples_split=4, random_state=42)

for name, m in [('Random Forest', rf), ('Gradient Boosting', gb), ('Decision Tree', dt)]:
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:,1])
    print(f"✅ {name}: Accuracy={acc:.4f}  AUC={auc:.4f}")

joblib.dump({
    'rf': rf, 'gb': gb, 'dt': dt,
    'le_sex': le_sex, 'le_emb': le_emb,
    'le_title': le_title, 'le_cabin': le_cabin,
    'features': FEATURES,
    'feature_labels': FEATURE_LABELS,
    'age_median': data['age'].median(),
    'fare_median': data['fare'].median(),
    'title_classes': le_title.classes_.tolist(),
    'sex_classes': le_sex.classes_.tolist(),
    'emb_classes': le_emb.classes_.tolist(),
}, 'model_v2.pkl')

print("\n✅ model_v2.pkl saved successfully!")
EOF
