# 🚀 telecom-customer-churn-prediction

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-production-green.svg)

## 📋 Description

**Problème résolu :** L'entreprise de télécommunications(Telco) perd 25% de ses clients chaque année alors que coût d'acquisition d'un nouveau client (CAC) est 5x plus élevé que la rétention.

**Solution apportée :** Identifier les clients à risque 3 mois avant leur départ pour lancer des actions de rétention.

---

## 🎬 Démo Live

🔗 **[Voir la démo interactive](votre-lien-demo)**

![Demo GIF](assets/demo.gif)

_Capture d'écran montrant les fonctionnalités principales_

---

## ⚡ Installation Rapide

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/telecom-customer-churn-prediction.git

# 2. Installer les dépendances
cd telecom-customer-churn-prediction
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

⏱️ **Prêt en moins de 3 minutes !**

---

## 🔧 Usage

### Via l'interface web :

1. Uploadez votre fichier CSV de clients.
2. Visualisez les prédictions de churn en temps réel.
3. Exportez la liste des clients à risque.

### En tant que librairie Python :

```python
from src.models import CustomerChurnPredictor

# Charger le modèle pré-entraîné
model = CustomerChurnPredictor.load('models/best_model.pkl')

# Obtenir une prédiction pour un nouveau client
prediction = model.predict(new_customer_data)
print(f"Probabilité de churn: {prediction:.2%}")
```

---

## 📊 Performances & Résultats

### 📈 Métriques du Modèle :

- **Accuracy :** >= 80.0%
- **AUC :** 85.0%

### 💰 Impact Business Estimé :

- **Réduction du churn :** Potentiel de -15% en ciblant les clients à risque.
- **ROI :** +125K€/an pour une entreprise de 1000 clients.

---

## 🛠️ Technologies Utilisées

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=Python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)

---

## 🚀 Prochaines Étapes

- [ ] Développer une **API REST** pour l'intégration.
- [ ] Mettre en place un pipeline **MLOps** avec monitoring.

---

## 👤 Contact & Collaboration

**Développé par :** Habibou

**Intéressé par ce projet ?**

- 💼 **Recruteurs :** Je suis ouvert aux opportunités. Contactez-moi sur [LinkedIn](linkedin.com/in/mouhamadou-habibou-ba-3595b317a) ou par [email](habiblearning089@gmail.com).
- 🤝 **Développeurs :** Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une Pull Request.

---

⭐ **Ce projet vous a plu ?** N'hésitez pas à laisser une étoile pour soutenir mon travail !
