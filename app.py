from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import io
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "diet_exercise_model.pkl")
EXTENDED_MODEL_PATH = os.path.join(BASE_DIR, "extended_diet_exercise_model.pkl")


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place your model there.")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

model = data.get('model')
gender_map = data.get('gender_map', {'Male': 0, 'Female': 1})
activity_map = data.get('activity_map', {'Low': 0, 'Medium': 1, 'High': 2})
goal_map = data.get('goal_map', {'Lose Weight': 0, 'Maintain': 1, 'Gain Weight': 2})

extended_model = None
extended_encoders = None
if os.path.exists(EXTENDED_MODEL_PATH):
    try:
        with open(EXTENDED_MODEL_PATH, "rb") as ef:
            ext = pickle.load(ef)
            extended_model = ext.get("model")
            extended_encoders = ext.get("exercise_encoders")
    except Exception:
        extended_model = None
        extended_encoders = None

def bmi_calc(weight, height_cm):
    return weight / ((height_cm / 100.0) ** 2)

def bmr_calc(gender, weight, height, age):
    if gender == 'Male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

activity_factor_map = {'Low': 1.2, 'Medium': 1.55, 'High': 1.725}

def calorie_goal_ranges(tdee, goal):
    if goal == 'Lose Weight':
        low = max(900, int(tdee - 600))
        high = max(900, int(tdee - 300))
    elif goal == 'Gain Weight':
        low = int(tdee + 250)
        high = int(tdee + 500)
    else:
        low = int(tdee - 100)
        high = int(tdee + 100)
    return low, high

def macro_split(goal):
    if goal == 'Lose Weight':
        return (0.30, 0.40, 0.30)
    if goal == 'Gain Weight':
        return (0.25, 0.50, 0.25)
    return (0.25, 0.45, 0.30)

def exercise_recommender(age, bmi, weight, goal, activity):
    suggestions = []

    if bmi < 18.5:
        suggestions = [
            "Resistance training - full body (3x/week)",
            "Compound lifts (progressive overload)",
            "Core strengthening (planks, 3x/week)",
            "Light cardio 10-15 min post-workout",
            "High-calorie recovery snack (protein + carbs)"
        ]
    elif bmi < 25:
        suggestions = [
            "Strength training - 2x/week (full-body)",
            "Cardio - 30 min moderate (3x/week)",
            "Mobility & stretching - daily 10 min",
            "Core stability exercises",
            "Optional HIIT 1x/week"
        ]
    elif bmi < 30:
        suggestions = [
            "Brisk walking - 30-45 min daily",
            "Bodyweight circuit - 3x/week (circuits 20-30 min)",
            "Low-impact cardio (cycling/swim)",
            "Resistance bands - full body 2-3x/week",
            "Mobility & stretching"
        ]
    else:
        suggestions = [
            "Low-impact cardio - walking/cycling 30 min daily",
            "Chair/low-impact strength exercises - 3x/week",
            "Resistance bands - light intensity",
            "Flexibility & balance drills daily",
            "Consult physician / supervised program recommended"
        ]

    if age >= 60:
        suggestions = [s.replace("HIIT", "low-impact intervals") for s in suggestions]
        suggestions = [s for s in suggestions if "Compound lifts" not in s and "progressive overload" not in s]
        if "Flexibility & balance drills daily" not in suggestions:
            suggestions[-1] = "Flexibility & balance drills daily"
    elif age < 18:
        suggestions = [s.replace("Compound lifts", "Supervised compound lifts") for s in suggestions]

    if goal == "Gain Weight":
        suggestions[0] = "Progressive resistance training - 3x/week (focus on strength)"
        suggestions[-1] = "Increase caloric intake + protein timing (post-workout snack)"
    elif goal == "Lose Weight":
        suggestions[0] = "Interval cardio or brisk walking 30-45 min (daily)"

    if activity == "High":
        suggestions = [s for s in suggestions if "HIIT" not in s]
        if "Mobility & stretching - daily 10 min" not in suggestions:
            suggestions.insert(1, "Mobility & stretching - daily 10 min")

    return suggestions[:5]

def meal_recommender(bmi, goal):
    if goal == "Gain Weight":
        return [
            "Breakfast: Omelette + oats + peanut butter (calorie-dense)",
            "Lunch: Chicken/quinoa + vegetables + olive oil",
            "Snack: Greek yogurt + nuts (high-calorie snack)",
            "Dinner: Salmon + sweet potato + greens"
        ]
    elif goal == "Lose Weight":
        if bmi >= 30:
            return [
                "Breakfast: Veg omelette + spinach (low-calorie, protein-rich)",
                "Lunch: Grilled chicken salad (light dressing)",
                "Snack: Apple + handful of almonds (small portion)",
                "Dinner: Steamed fish + non-starchy veg (small carb)"
            ]
        else:
            return [
                "Breakfast: Oats with berries (controlled portion)",
                "Lunch: Salad + lean protein + wholegrain side",
                "Snack: Cottage cheese or boiled egg",
                "Dinner: Grilled paneer/tofu + steamed vegetables"
            ]
    else:
        return [
            "Breakfast: Wholegrain toast + eggs + fruit",
            "Lunch: Balanced plate (protein + complex carbs + veg)",
            "Snack: Nuts + yogurt",
            "Dinner: Lean protein + complex carbs + salad"
        ]

def compute_plan(payload):
    required = ['gender', 'age', 'height_cm', 'weight_kg', 'activity', 'goal']
    missing = [k for k in required if k not in payload]
    if missing:
        return {"error": f"missing fields: {', '.join(missing)}"}, 400

    try:
        gender = str(payload['gender'])
        age = int(payload['age'])
        height = float(payload['height_cm'])
        weight = float(payload['weight_kg'])
        activity = str(payload['activity'])
        goal = str(payload['goal'])
    except Exception as e:
        return {"error": "invalid field types", "details": str(e)}, 400

    bmi = round(bmi_calc(weight, height), 2)

    row = [[
        gender_map.get(gender, 0),
        age,
        height,
        weight,
        bmi,
        activity_map.get(activity, 0),
        goal_map.get(goal, 1)
    ]]

    try:
        diet_pred = model.predict(row)[0]
    except Exception as e:
        return {"error": "model prediction failed", "details": str(e)}, 500

    try:
        bmr = bmr_calc(gender, weight, height, age)
        tdee = bmr * activity_factor_map.get(activity, 1.2)
    except Exception as e:
        return {"error": "failed to compute metabolic values", "details": str(e)}, 500

    cal_low, cal_high = calorie_goal_ranges(tdee, goal)
    p_ratio, c_ratio, f_ratio = macro_split(goal)
    avg_cal = (cal_low + cal_high) / 2.0
    protein_g = int((avg_cal * p_ratio) / 4)
    carbs_g = int((avg_cal * c_ratio) / 4)
    fats_g = int((avg_cal * f_ratio) / 9)

    meal_templates = meal_recommender(bmi, goal)

    exercises = None
    if extended_model is not None:
        try:
            pred = extended_model.predict(row)
            if extended_encoders:
                decoded = []
                for i, key in enumerate(sorted(extended_encoders.keys())):
                    le = extended_encoders[key]
                    val = pred[0][i] if hasattr(pred[0], "__len__") else pred[0]
                    try:
                        decoded.append(le.inverse_transform([int(val)])[0])
                    except Exception:
                        decoded.append(str(val))
                exercises = decoded[:5]
            else:
                exercises = [str(x) for x in pred[0][:5]]
        except Exception:
            exercises = exercise_recommender(age, bmi, weight, goal, activity)
    else:
        exercises = exercise_recommender(age, bmi, weight, goal, activity)

    result = {
        "bmi": bmi,
        "diet_style": diet_pred,
        "calorie_range": {"low": cal_low, "high": cal_high},
        "macros_g": {"protein_g": protein_g, "carbs_g": carbs_g, "fats_g": fats_g},
        "meal_suggestions": meal_templates,
        "exercise_suggestions": exercises,
        "exercise_note": "These are guideline suggestions. Adjust intensity and consult a professional if you have medical conditions.",
        "raw": {
            "bmr": int(bmr),
            "tdee": int(tdee)
        }
    }

    return result, 200

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Diet & Exercise API running. Use POST /predict with JSON."}), 200

@app.route("/predict", methods=["POST"])
def predict():
    payload = None
    if request.is_json:
        payload = request.get_json()
    else:
        if request.form:
            payload = request.form.to_dict()
        else:
            try:
                raw = request.get_data(as_text=True)
                if raw:
                    import json
                    payload = json.loads(raw)
            except Exception:
                payload = None

    if not payload:
        return jsonify({"error": "Request must contain JSON (Content-Type: application/json) or form data."}), 415

    result, status = compute_plan(payload)
    return jsonify(result), status

@app.route("/report", methods=["POST"])
def report():
    if request.is_json:
        payload = request.get_json()
    elif request.form:
        payload = request.form.to_dict()
    else:
        try:
            raw = request.get_data(as_text=True)
            if raw:
                import json
                payload = json.loads(raw)
            else:
                payload = None
        except Exception:
            payload = None

    if not payload:
        return jsonify({"error": "Please POST JSON or form data with the same payload as /predict."}), 415

    data, status = compute_plan(payload)
    if status != 200:
        return jsonify(data), status

    report_lines = []
    report_lines.append("SMART DIET & EXERCISE PLAN")
    report_lines.append("--------------------------")
    report_lines.append(f"Gender: {payload.get('gender')}")
    report_lines.append(f"Age: {payload.get('age')}")
    report_lines.append(f"Height: {payload.get('height_cm')} cm")
    report_lines.append(f"Weight: {payload.get('weight_kg')} kg")
    report_lines.append(f"BMI: {data['bmi']}")
    report_lines.append(f"Suggested diet style: {data['diet_style']}")
    report_lines.append(f"Calorie range: {data['calorie_range']['low']} - {data['calorie_range']['high']} kcal")
    report_lines.append("")
    report_lines.append("Meal suggestions:")
    for m in data['meal_suggestions']:
        report_lines.append(f"- {m}")
    report_lines.append("")
    report_lines.append("Exercise suggestions:")
    for ex in data['exercise_suggestions']:
        report_lines.append(f"- {ex}")
    report_text = "\n".join(report_lines)

    return send_file(
        io.BytesIO(report_text.encode('utf-8')),
        mimetype='text/plain',
        as_attachment=True,
        download_name='diet_exercise_report.txt'
    )

if __name__ == "__main__":
    app.run()

