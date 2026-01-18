# NutriMed – Flask Backend (AI Diet & Exercise API)

This repository contains the backend service for **NutriMed**, an AI-powered diet and exercise recommendation system developed as part of an academic AI/ML project.

The backend is built using **Flask** and exposes REST APIs that process user health inputs and return personalized diet, calorie, and exercise recommendations using a trained Machine Learning model.

---

## 🔹 Features
- RESTful API built with Flask
- Machine Learning model (Decision Tree) for diet prediction
- BMI, BMR, and TDEE calculations
- Personalized calorie and macronutrient recommendations
- Rule-based + ML-based exercise suggestions
- Report generation endpoint
- CORS-enabled for frontend integration

---

## 🔹 Tech Stack
- Python
- Flask & Flask-CORS
- Scikit-learn
- Pandas & NumPy
- Gunicorn (production server)

---

## 🔹 API Endpoints

### `GET /`
Health check endpoint  
Returns API status message.

### `POST /predict`
Accepts user health data and returns a personalized diet & exercise plan.

**Sample Request (JSON):**
```json
{
  "gender": "Male",
  "age": 25,
  "height_cm": 170,
  "weight_kg": 65,
  "activity": "Medium",
  "goal": "Maintain"
}
