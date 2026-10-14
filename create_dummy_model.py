import pickle
import os

# Prescription Knowledge Dictionary
PRESCRIPTION_DATA = {
    "fever": {
        "disease": "Fever / Viral Infection",
        "medicines": ["Paracetamol 650 mg", "Electral Oral Rehydration Salts"],
        "dosage": [
            "Paracetamol: 1 tablet every 6-8 hours after meals (if fever > 100°F)",
            "Electral: 1 sachet in 1 liter clean water, sip throughout day"
        ],
        "timings": ["Morning & Night – After food", "Throughout the day as needed"],
        "home_remedies": [
            "Drink plenty of warm fluids (soups, herbal tea, warm water)",
            "Take cold compress on forehead if body temperature spikes",
            "Adequate bed rest for 48 hours in a well-ventilated room"
        ],
        "precautions": [
            "Avoid cold drinks, ice, and unpasteurized dairy",
            "Do not exceed 3 grams of Paracetamol in 24 hours",
            "Consult physician if fever persists beyond 3 days or is accompanied by rash"
        ]
    },
    "cold": {
        "disease": "Common Cold / Cough / Upper Respiratory Infection",
        "medicines": ["Cetirizine 10 mg", "Dextromethorphan Cough Syrup 10 ml"],
        "dosage": [
            "Cetirizine: 1 tablet at night",
            "Cough Syrup: 2 teaspoons (10 ml) twice daily"
        ],
        "timings": ["Night – After dinner", "Morning & Evening – After meals"],
        "home_remedies": [
            "Steam inhalation with eucalyptus or menthol twice daily",
            "Warm saline water gargle 3 times a day for throat irritation",
            "Ginger and honey tea"
        ],
        "precautions": [
            "Avoid chilled water, ice creams, and dusty environments",
            "Wear a protective mask if going outside",
            "Seek urgent medical care if you experience shortness of breath"
        ]
    },
    "cough": {
        "disease": "Bronchial Cough / Throat Irritation",
        "medicines": ["Ambroxol Hydrochloride Cough Syrup 10 ml", "Lozenges (Strepsils / Cofsils)"],
        "dosage": [
            "Cough Syrup: 10 ml twice daily after food",
            "Lozenges: Dissolve 1 lozenge slowly in mouth every 4 hours"
        ],
        "timings": ["Morning & Night – After food"],
        "home_remedies": [
            "Warm turmeric milk before sleeping",
            "Warm water gargle with rock salt",
            "Steam inhalation twice a day"
        ],
        "precautions": [
            "Avoid oily, deep-fried, and sour food",
            "Avoid exposure to air conditioning directly on chest/face"
        ]
    },
    "diabetes": {
        "disease": "Type 2 Diabetes Mellitus Management",
        "medicines": ["Metformin Hydrochloride 500 mg"],
        "dosage": ["1 tablet daily with or immediately after meals"],
        "timings": ["Morning – With breakfast"],
        "home_remedies": [
            "Brisk walking for at least 30-45 minutes daily",
            "Incorporate high-fiber foods: oats, leafy vegetables, sprouts, fenugreek",
            "Stay hydrated with cinnamon-infused or plain water"
        ],
        "precautions": [
            "Strictly avoid refined sugars, sweetened beverages, and bakery goods",
            "Monitor fasting and post-prandial blood glucose levels regularly",
            "Never skip meals while taking anti-diabetic medication"
        ]
    },
    "heart": {
        "disease": "Cardiovascular / Hypertension Support",
        "medicines": ["Aspirin 75 mg (Cardioprotective)", "Atorvastatin 10 mg"],
        "dosage": [
            "Aspirin 75 mg: 1 tablet once daily",
            "Atorvastatin 10 mg: 1 tablet once daily at bedtime"
        ],
        "timings": ["Morning – After food (Aspirin)", "Night – Bedtime (Atorvastatin)"],
        "home_remedies": [
            "Light daily aerobic exercise / brisk walking as approved by your cardiologist",
            "Practice deep breathing (Pranayama) and stress management",
            "Low-sodium, heart-healthy Mediterranean diet with nuts and omega-3"
        ],
        "precautions": [
            "Strictly avoid oily, fried, deep-fried snacks, and high-cholesterol foods",
            "Limit table salt intake to under 1 teaspoon daily",
            "Seek IMMEDIATE emergency care (Call 112) if experiencing chest heaviness or left arm pain"
        ]
    },
    "skin": {
        "disease": "Skin Allergy / Dermatitis / Fungal Infection",
        "medicines": ["Cetirizine 10 mg", "Hydrocortisone / Antifungal Topical Ointment 1%"],
        "dosage": [
            "Cetirizine: 1 tablet once daily",
            "Ointment: Apply thin layer to affected area twice daily"
        ],
        "timings": ["Night – After dinner", "Morning & Evening after washing affected area"],
        "home_remedies": [
            "Keep skin completely clean, dry, and aerated",
            "Apply aloe vera gel or calamine lotion to soothe itchiness",
            "Wear loose, breathable 100% cotton clothing"
        ],
        "precautions": [
            "Do not scratch or puncture the affected area to prevent secondary bacterial infection",
            "Avoid harsh synthetic soaps and fragrant body lotions",
            "Consult a dermatologist if rash spreads or develops pustules"
        ]
    }
}

os.makedirs("models", exist_ok=True)

with open("models/dummy_model.pkl", "wb") as f:
    pickle.dump(PRESCRIPTION_DATA, f)

print("Prescription knowledge base model created successfully at models/dummy_model.pkl")
