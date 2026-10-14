"""
Machine Learning Training Pipeline for Smart Healthcare Disease Diagnosis
Trains a multi-class Random Forest Classifier on Disease-Symptom correlations
and packages pharmacology & Health Cafe dietary guidance.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Complete list of 50 clinical symptoms
SYMPTOMS_LIST = [
    "high_fever", "mild_fever", "chills", "shivering", "sweating", "fatigue", "lethargy",
    "headache", "migraine", "dizziness", "cough", "phlegm", "throat_irritation",
    "breathlessness", "chest_pain", "continuous_sneezing", "runny_nose", "congestion",
    "sinus_pressure", "loss_of_smell", "stomach_pain", "abdominal_pain", "acidity",
    "vomiting", "nausea", "indigestion", "diarrhoea", "constipation", "loss_of_appetite",
    "burning_micturition", "dark_urine", "yellowish_skin", "yellowing_of_eyes",
    "skin_rash", "itching", "redness_of_eyes", "joint_pain", "back_pain", "muscle_pain",
    "swelling_joints", "weight_loss", "weight_gain", "excessive_hunger", "increased_thirst",
    "frequent_urination", "blurred_vision", "anxiety", "restlessness", "insomnia", "fast_heart_rate"
]

# Disease profiles with characteristic symptoms (primary and secondary)
DISEASE_PROFILES = {
    "Viral Fever": {
        "primary": ["high_fever", "chills", "fatigue", "headache", "sweating", "muscle_pain"],
        "secondary": ["loss_of_appetite", "mild_fever", "cough"]
    },
    "Common Cold & Flu": {
        "primary": ["continuous_sneezing", "runny_nose", "cough", "throat_irritation", "congestion"],
        "secondary": ["mild_fever", "headache", "chills", "sinus_pressure", "loss_of_smell"]
    },
    "Bronchitis & Chronic Cough": {
        "primary": ["cough", "phlegm", "breathlessness", "chest_pain", "throat_irritation"],
        "secondary": ["fatigue", "mild_fever", "headache"]
    },
    "Type 2 Diabetes": {
        "primary": ["increased_thirst", "frequent_urination", "excessive_hunger", "fatigue", "weight_loss"],
        "secondary": ["blurred_vision", "lethargy", "itching"]
    },
    "Hypertension & Cardiovascular Strain": {
        "primary": ["fast_heart_rate", "chest_pain", "dizziness", "headache", "breathlessness"],
        "secondary": ["fatigue", "anxiety", "sweating", "insomnia"]
    },
    "Gastritis & Acid Reflux (GERD)": {
        "primary": ["acidity", "indigestion", "stomach_pain", "nausea", "chest_pain"],
        "secondary": ["loss_of_appetite", "vomiting", "headache"]
    },
    "Gastroenteritis & Food Poisoning": {
        "primary": ["vomiting", "diarrhoea", "abdominal_pain", "nausea", "stomach_pain"],
        "secondary": ["chills", "mild_fever", "fatigue", "loss_of_appetite"]
    },
    "Skin Allergy & Dermatitis": {
        "primary": ["skin_rash", "itching", "redness_of_eyes"],
        "secondary": ["continuous_sneezing", "fatigue"]
    },
    "Migraine & Tension Headache": {
        "primary": ["migraine", "headache", "nausea", "dizziness", "blurred_vision"],
        "secondary": ["fatigue", "anxiety", "insomnia"]
    },
    "Arthritis & Joint Inflammation": {
        "primary": ["joint_pain", "swelling_joints", "muscle_pain", "back_pain"],
        "secondary": ["fatigue", "lethargy"]
    },
    "Asthma & Respiratory Distress": {
        "primary": ["breathlessness", "cough", "chest_pain", "fatigue"],
        "secondary": ["fast_heart_rate", "anxiety", "restlessness"]
    },
    "Typhoid Fever": {
        "primary": ["high_fever", "chills", "vomiting", "abdominal_pain", "diarrhoea", "fatigue"],
        "secondary": ["loss_of_appetite", "headache", "muscle_pain"]
    },
    "Malaria": {
        "primary": ["high_fever", "chills", "shivering", "sweating", "headache", "nausea"],
        "secondary": ["vomiting", "diarrhoea", "muscle_pain"]
    },
    "Jaundice & Hepatic Stress": {
        "primary": ["yellowish_skin", "yellowing_of_eyes", "dark_urine", "abdominal_pain", "loss_of_appetite"],
        "secondary": ["nausea", "vomiting", "fatigue", "mild_fever"]
    },
    "Urinary Tract Infection (UTI)": {
        "primary": ["burning_micturition", "frequent_urination", "abdominal_pain", "mild_fever"],
        "secondary": ["dark_urine", "nausea", "fatigue"]
    },
    "Anxiety & Chronic Fatigue": {
        "primary": ["anxiety", "restlessness", "insomnia", "fatigue", "fast_heart_rate"],
        "secondary": ["dizziness", "headache", "weight_loss"]
    }
}

# Rich clinical, pharmacology, and Smart Health Cafe diet dataset
CLINICAL_KNOWLEDGE = {
    "Viral Fever": {
        "specialist": "General Physician",
        "severity": "Moderate",
        "medicines": ["Paracetamol 650 mg", "Electral ORS Solution", "Vitamin C 500 mg"],
        "dosage": [
            "Paracetamol: 1 tab every 6-8 hrs if temp > 100°F (Max 3/day)",
            "Electral: Mix 1 sachet in 1L boiled, cooled water; sip throughout day",
            "Vitamin C: 1 chewable tablet once daily after breakfast"
        ],
        "timings": ["Morning & Night after meals", "Continuous throughout day", "Morning after breakfast"],
        "home_remedies": [
            "Cold water sponge on forehead and armpits if fever exceeds 101°F",
            "Drink hot ginger, holy basil (tulsi) and black pepper tea",
            "Complete physical rest in an airy, quiet room"
        ],
        "health_cafe_diet": {
            "title": "Immunity & Hydration Cafe Plan",
            "breakfast": "Light vegetable oats porridge or steamed idlis with mild mint chutney",
            "lunch": "Moong dal khichdi with a dash of clarified butter (ghee) and curd",
            "dinner": "Clear chicken soup or tomato-lentil broth with soft boiled vegetables",
            "beverages": "Coconut water, warm lemon-honey water, and pomegranate juice",
            "strictly_avoid": "Chilled beverages, oily fried snacks, unpasteurized milk, spicy curries"
        },
        "precautions": [
            "Monitor body temperature every 4 hours with a digital thermometer",
            "Do not consume NSAIDs like Aspirin without physician consultation",
            "Consult hospital immediately if fever persists over 3 days or causes breathing difficulty"
        ]
    },
    "Common Cold & Flu": {
        "specialist": "ENT Specialist / Physician",
        "severity": "Mild",
        "medicines": ["Cetirizine 10 mg", "Ambroxol + Levosalbutamol Syrup", "Saline Nasal Drops"],
        "dosage": [
            "Cetirizine: 1 tablet at bedtime",
            "Cough Syrup: 10 ml twice daily after meals",
            "Nasal Drops: 2 drops in each nostril 3 times daily"
        ],
        "timings": ["Night after dinner", "Morning & Night after food", "Morning, Afternoon, Night"],
        "home_remedies": [
            "Steam inhalation with 2 drops of eucalyptus oil twice daily",
            "Warm saline water gargling 3 times a day for throat soothe",
            "Drink warm turmeric milk with crushed black pepper before bed"
        ],
        "health_cafe_diet": {
            "title": "Anti-Inflammatory Decongestant Plan",
            "breakfast": "Warm spiced oatmeal with crushed almonds and a cup of ginger tea",
            "lunch": "Vegetable clear soup with boiled brown rice and steamed spinach",
            "dinner": "Garlic-infused vegetable broth with whole wheat toast",
            "beverages": "Herbal green tea, lukewarm water with a squeeze of fresh lemon",
            "strictly_avoid": "Ice cream, cold refrigerated sodas, sour foods, deep-fried snacks"
        },
        "precautions": [
            "Wear a surgical mask when in public or around family members",
            "Wash hands thoroughly after blowing nose or coughing",
            "Keep warm and avoid direct cold air conditioning exposure"
        ]
    },
    "Bronchitis & Chronic Cough": {
        "specialist": "Pulmonologist / Chest Physician",
        "severity": "Moderate to High",
        "medicines": ["Ambroxol Hydrochloride 30 mg", "Montelukast + Levocetirizine", "Steam Inhalant Capsules"],
        "dosage": [
            "Ambroxol: 1 tab three times daily after food",
            "Montelukast+Levo: 1 tab at bedtime",
            "Inhalant: Dissolve capsule in steaming water twice daily"
        ],
        "timings": ["Morning, Afternoon, Night after meals", "Night at bedtime", "Morning & Evening"],
        "home_remedies": [
            "Honey with freshly ground black pepper 1 teaspoon twice a day",
            "Chest warmth compress with warm towel",
            "Elevate head with an extra pillow while sleeping"
        ],
        "health_cafe_diet": {
            "title": "Respiratory Soothing Nutrition Plan",
            "breakfast": "Warm chia seed pudding with warm almond milk and honey",
            "lunch": "Soft cooked lentils with cumin, garlic and soft millet roti",
            "dinner": "Hot vegetable stew with sweet potatoes and carrots",
            "beverages": "Licorice (mulethi) tea, warm ginger water",
            "strictly_avoid": "Dairy products with thick cream, icy drinks, smoke, dust exposure"
        },
        "precautions": [
            "Avoid exposure to smoke, paint fumes, and outdoor smog",
            "Seek emergency care if phlegm contains blood or breathing becomes strained"
        ]
    },
    "Type 2 Diabetes": {
        "specialist": "Endocrinologist / Diabetologist",
        "severity": "Chronic / Requires Continuous Care",
        "medicines": ["Metformin 500 mg Extended Release", "Alpha Lipoic Acid + B-Complex"],
        "dosage": [
            "Metformin: 1 tablet with principal meal",
            "B-Complex: 1 capsule in the morning"
        ],
        "timings": ["Morning with breakfast", "Morning after breakfast"],
        "home_remedies": [
            "Soak 1 teaspoon fenugreek seeds overnight in water and drink on empty stomach",
            "Brisk walking for 35-45 minutes daily after breakfast or dinner",
            "Regular yoga poses like Mandukasana and Kapalbhati"
        ],
        "health_cafe_diet": {
            "title": "Low Glycemic Index Metabolic Cafe Plan",
            "breakfast": "Sprouted moong beans salad with cucumber, lemon and roasted flaxseeds",
            "lunch": "2 Multigrain rotis (barley+chickpea) with bitter gourd (karela) sabzi and dal",
            "dinner": "Grilled tofu or paneer with high-fiber stir-fried zucchini and broccoli",
            "beverages": "Cinnamon water, bitter gourd juice, unsweetened green tea",
            "strictly_avoid": "White sugar, fruit juices, bakery products, white rice, sweetened soda"
        },
        "precautions": [
            "Check fasting and 2-hour post-meal blood sugar levels 3 times a week",
            "Carry glucose tablets or hard candy in case of hypoglycemia (dizziness/sweating)",
            "Inspect feet daily for any cuts, numbness, or blisters"
        ]
    },
    "Hypertension & Cardiovascular Strain": {
        "specialist": "Cardiologist",
        "severity": "High / Critical",
        "medicines": ["Amlodipine 5 mg or Telmisartan 40 mg (Rx by Doctor)", "Aspirin 75 mg"],
        "dosage": [
            "Telmisartan / Amlodipine: 1 tablet once daily morning",
            "Aspirin 75 mg: 1 tablet after lunch"
        ],
        "timings": ["Morning 8:00 AM after light food", "Afternoon after lunch"],
        "home_remedies": [
            "Practice Anulom Vilom (alternate nostril breathing) for 15 minutes daily",
            "Regular mild aerobic walk 30 minutes 5 days a week",
            "Reduce salt to under half teaspoon daily"
        ],
        "health_cafe_diet": {
            "title": "DASH Heart-Protective Cafe Plan",
            "breakfast": "Steel-cut oatmeal topped with crushed walnuts, chia seeds, and sliced banana",
            "lunch": "Steamed quinoa or brown rice with garlic spinach (palak) and boiled beans",
            "dinner": "Lentil soup with baked vegetables (bottle gourd, zucchini, beans)",
            "beverages": "Beetroot juice (rich in natural nitrates), hibiscus tea",
            "strictly_avoid": "Pickles, processed meats, papad, salted chips, butter, saturated fats"
        },
        "precautions": [
            "CALL EMERGENCY 112 IMMEDIATELY if you experience radiating chest or left arm pain",
            "Monitor blood pressure daily at the same time and record readings",
            "Never stop anti-hypertensive medication abruptly"
        ]
    },
    "Gastritis & Acid Reflux (GERD)": {
        "specialist": "Gastroenterologist",
        "severity": "Mild to Moderate",
        "medicines": ["Pantoprazole 40 mg", "Sucralfate + Oxetacaine Suspension 10 ml"],
        "dosage": [
            "Pantoprazole: 1 tablet 30 minutes before breakfast",
            "Antacid: 2 teaspoons 1 hour after meals and bedtime"
        ],
        "timings": ["Morning on empty stomach", "After meals and at bedtime"],
        "home_remedies": [
            "Sip chilled skimmed milk or freshly prepared coconut water",
            "Chew half a teaspoon of fennel seeds (saunf) after every meal",
            "Elevate head of bed by 6 inches to prevent nighttime acid reflux"
        ],
        "health_cafe_diet": {
            "title": "Alkaline & Gut Soothing Nutrition Plan",
            "breakfast": "Ripe papaya slices with soaked chia seeds and a cup of chamomile tea",
            "lunch": "Curd rice with pomegranate seeds and lightly salted boiled pumpkin",
            "dinner": "Soft khichdi or vegetable dahlia with zucchini and carrots",
            "beverages": "Aloe vera juice, fresh coconut water, barley water",
            "strictly_avoid": "Red chili, vinegar, tomatoes, coffee, tea, chocolate, carbonated soda"
        },
        "precautions": [
            "Do not lie down within 2 hours of having a meal",
            "Eat small, frequent meals rather than large heavy feasts",
            "Avoid tight-fitting waist belts or abdominal garments"
        ]
    },
    "Gastroenteritis & Food Poisoning": {
        "specialist": "General Physician",
        "severity": "Moderate",
        "medicines": ["ORS Rehydration Solution", "Probiotic Capsules (Lactobacillus)", "Ondansetron 4 mg"],
        "dosage": [
            "ORS: 200 ml after every loose stool",
            "Probiotic: 1 capsule twice daily",
            "Ondansetron: 1 tablet for persistent vomiting"
        ],
        "timings": ["Continuous as needed", "Morning & Night with water", "As needed before food"],
        "home_remedies": [
            "Drink salted rice starch water (kanji)",
            "Sip pomegranate juice to restore electrolytes",
            "Total gut rest from solid food for the first 6 hours"
        ],
        "health_cafe_diet": {
            "title": "BRAT Gut Recovery Plan",
            "breakfast": "1 Mashed banana with a slice of dry toast",
            "lunch": "Soft white boiled rice with clear moong dal and a pinch of rock salt",
            "dinner": "Lentil broth with applesauce and saltine crackers",
            "beverages": "Electrolyte ORS water, tender coconut water",
            "strictly_avoid": "Raw salads, dairy milk, oily food, spicy gravies, citrus fruits"
        },
        "precautions": [
            "Watch for dehydration signs: dry mouth, dark urine, sunken eyes",
            "Visit emergency clinic if unable to keep liquids down for over 12 hours"
        ]
    },
    "Skin Allergy & Dermatitis": {
        "specialist": "Dermatologist",
        "severity": "Mild to Moderate",
        "medicines": ["Levocetirizine 5 mg", "Calamine Lotion Topical", "Hydrocortisone Cream 1%"],
        "dosage": [
            "Levocetirizine: 1 tab at bedtime",
            "Calamine: Apply to affected itching skin 3 times daily",
            "Hydrocortisone: Apply thin film to localized rash twice daily"
        ],
        "timings": ["Night after dinner", "Morning, Afternoon, Evening", "Morning & Night"],
        "home_remedies": [
            "Take oatmeal bath with lukewarm water to soothe intense itching",
            "Apply pure virgin coconut oil or chilled aloe vera gel",
            "Wear loose, washed 100% pure cotton garments"
        ],
        "health_cafe_diet": {
            "title": "Anti-Histamine Cleanse Diet",
            "breakfast": "Apple and blueberry smoothie with almond milk and pumpkin seeds",
            "lunch": "Steamed sweet potato with cucumber salad and green gram sprouts",
            "dinner": "Lentil soup with zucchini, bottle gourd, and steamed brown rice",
            "beverages": "Watermelon juice, filtered water with mint and cucumber slices",
            "strictly_avoid": "Shellfish, peanuts, fermented foods, aged cheese, artificial food colorings"
        },
        "precautions": [
            "Do NOT scratch the skin to prevent bacterial infection and scarring",
            "Use only fragrance-free hypoallergenic bathing soaps",
            "Avoid direct exposure to synthetic detergents and cleaning chemicals"
        ]
    },
    "Migraine & Tension Headache": {
        "specialist": "Neurologist",
        "severity": "Moderate to High",
        "medicines": ["Naproxen 500 mg or Paracetamol 650 mg", "Domperidone 10 mg (if nausea)"],
        "dosage": [
            "Pain Reliever: 1 tab at onset of migraine aura",
            "Domperidone: 1 tab 15 mins before pain reliever"
        ],
        "timings": ["At onset of symptoms", "Before pain medication"],
        "home_remedies": [
            "Rest in a pitch-dark, quiet, well-ventilated room",
            "Apply an ice pack to back of neck and temple area for 15 minutes",
            "Massage temples gently with diluted peppermint or lavender oil"
        ],
        "health_cafe_diet": {
            "title": "Magnesium-Rich Migraine Shield Diet",
            "breakfast": "Chia and flaxseed porridge with sliced bananas and soaked almonds",
            "lunch": "Spinach dal with brown rice, cucumber raita and boiled legumes",
            "dinner": "Baked sweet potato and pumpkin soup with whole grain bread",
            "beverages": "Electrolyte water, ginger tea (natural anti-nausea)",
            "strictly_avoid": "Aged cheese, monosodium glutamate (MSG), cured meats, chocolate, wine"
        },
        "precautions": [
            "Maintain a consistent sleep-wake schedule (7-8 hours nightly)",
            "Do not skip meals, which triggers hypoglycemic headaches",
            "Wear polarized sunglasses when stepping into intense sunlight"
        ]
    },
    "Arthritis & Joint Inflammation": {
        "specialist": "Rheumatologist / Orthopedic",
        "severity": "Chronic / Moderate",
        "medicines": ["Aceclofenac + Paracetamol", "Glucosamine + Chondroitin Supplement", "Topical Pain Relief Gel"],
        "dosage": [
            "Pain Reliever: 1 tablet after meals (max 2 daily during flare-up)",
            "Glucosamine: 1 tablet daily with lunch",
            "Gel: Apply gently without rubbing on affected joints 3 times daily"
        ],
        "timings": ["Morning & Night after meals", "Afternoon after lunch", "As needed"],
        "home_remedies": [
            "Warm mustard oil with crushed garlic and massage aching joints gently",
            "Alternating warm water compression for 15 minutes",
            "Low-impact swimming or water aerobics to keep joints supple"
        ],
        "health_cafe_diet": {
            "title": "Anti-Inflammatory Joint Vitality Plan",
            "breakfast": "Warm golden turmeric milk with chia seeds and walnuts",
            "lunch": "Brown rice with salmon or tofu, steamed broccoli, and black beans",
            "dinner": "Turmeric-spiced lentil soup with carrots and whole wheat flatbread",
            "beverages": "Ginger-turmeric tea, sour cherry juice, filtered water",
            "strictly_avoid": "Refined sugar, red meat, saturated fats, processed refined flour"
        },
        "precautions": [
            "Avoid strenuous high-impact jumping or heavy weight lifting",
            "Keep joints warm during cold weather or air-conditioned environments"
        ]
    },
    "Asthma & Respiratory Distress": {
        "specialist": "Pulmonologist",
        "severity": "High / Requires Inhaler",
        "medicines": ["Salbutamol Inhaler (100 mcg)", "Budesonide Inhaler (200 mcg)", "Montelukast 10 mg"],
        "dosage": [
            "Salbutamol (Reliever): 2 puffs during sudden wheezing/attack",
            "Budesonide (Preventer): 1 puff twice daily after gargling",
            "Montelukast: 1 tablet daily at night"
        ],
        "timings": ["As needed during bronchospasm", "Morning & Night regularly", "Night after dinner"],
        "home_remedies": [
            "Sit upright and take slow, pursed-lip deep breaths during mild tightness",
            "Sip warm caffeinated black coffee (acts as a mild natural bronchodilator)",
            "Keep bedroom completely free of dust, carpets, and pet dander"
        ],
        "health_cafe_diet": {
            "title": "Vitamin C & Lung Cleansing Plan",
            "breakfast": "Warm oatmeal with kiwi, strawberries, and crushed pumpkin seeds",
            "lunch": "Spinach and lentil soup with warm boiled brown rice",
            "dinner": "Steamed vegetables with garlic, ginger, and vegetable stew",
            "beverages": "Warm water with lemon and honey, peppermint tea",
            "strictly_avoid": "Preservative-heavy foods (sulfites in dried fruits), chilled iced water"
        },
        "precautions": [
            "ALWAYS carry your emergency rescue Salbutamol inhaler wherever you go",
            "Wash mouth thoroughly with water after using steroid inhalers",
            "Call emergency immediately if lips turn pale/blue or breath is gasping"
        ]
    },
    "Typhoid Fever": {
        "specialist": "Infectious Disease Specialist / Physician",
        "severity": "High / Bacterial Infection",
        "medicines": ["Azithromycin 500 mg (Prescription Only)", "Paracetamol 650 mg", "ORS Electrolytes"],
        "dosage": [
            "Azithromycin: 1 tablet daily for 7 days (as directed by physician)",
            "Paracetamol: 1 tablet every 6 hours if temperature rises",
            "ORS: Continuous hydration throughout the day"
        ],
        "timings": ["Morning 1 hour before meal", "As needed after food", "All day"],
        "home_remedies": [
            "Drink only boiled and cooled water or bottled mineral water",
            "Adequate bed rest for at least 10 to 14 days",
            "Cold compresses for high temperature spikes"
        ],
        "health_cafe_diet": {
            "title": "High-Calorie Soft Recovery Diet",
            "breakfast": "Soft boiled eggs or soaked oats porridge with ripe banana",
            "lunch": "Well-mashed rice with moong dal and boiled carrots",
            "dinner": "Lentil soup with boiled potatoes and plain toast",
            "beverages": "Tender coconut water, barley water, fresh apple juice",
            "strictly_avoid": "Raw salads, unpeeled fruits, street food, high-fiber coarse bran"
        },
        "precautions": [
            "Complete full antibiotic course even after fever subsides",
            "Practice strict hand hygiene before touching any food",
            "Isolate utensils and wash with boiling water"
        ]
    },
    "Malaria": {
        "specialist": "Physician",
        "severity": "High / Parasitic Infection",
        "medicines": ["Artemether + Lumefantrine (Prescription Anti-Malarial)", "Paracetamol 650 mg", "ORS"],
        "dosage": [
            "Anti-Malarial: Complete 3-day course strictly as instructed by doctor",
            "Paracetamol: 1 tab for fever and joint pain",
            "ORS: 1-2 liters daily"
        ],
        "timings": ["With milk or fatty food as directed", "After meals as needed", "All day"],
        "home_remedies": [
            "Cold water compresses during high fever stages",
            "Wear warm blankets during chills and shivering phases",
            "Keep environment mosquito-free with nets and repellents"
        ],
        "health_cafe_diet": {
            "title": "High Energy Liver-Friendly Plan",
            "breakfast": "Fresh fruit bowl with papaya, apple, and warm goat milk or skimmed milk",
            "lunch": "Soft cooked rice with light dal soup and steamed squash",
            "dinner": "Vegetable clear soup with soft cottage cheese and rice",
            "beverages": "Coconut water, orange juice, sugarcane juice from clean source",
            "strictly_avoid": "Oily foods, deep-fried snacks, spicy masalas, alcohol"
        },
        "precautions": [
            "Confirm diagnosis with a peripheral blood smear or Rapid Antigen Test",
            "Ensure mosquito repellents and screens are used to prevent transmission"
        ]
    },
    "Jaundice & Hepatic Stress": {
        "specialist": "Hepatologist / Gastroenterologist",
        "severity": "High / Liver Care Required",
        "medicines": ["Silymarin (Milk Thistle) 140 mg", "Multivitamin with B-Complex", "Digestive Enzymes"],
        "dosage": [
            "Silymarin: 1 capsule twice daily",
            "B-Complex: 1 tablet morning",
            "Enzymes: 1 tablet with principal meals"
        ],
        "timings": ["Morning & Night with water", "Morning after breakfast", "With lunch & dinner"],
        "home_remedies": [
            "Complete bed rest to allow liver cell regeneration",
            "Drink freshly crushed sugarcane juice prepared hygienically",
            "Consume radish leaf juice (traditional liver cleanser) in moderation"
        ],
        "health_cafe_diet": {
            "title": "Low-Fat Hepatic Detox Plan",
            "breakfast": "Boiled porridge or soft idli with tomato soup (zero oil)",
            "lunch": "Steamed white rice with boiled bottle gourd, ridge gourd, and yellow dal",
            "dinner": "Boiled vegetable stew with barley water",
            "beverages": "Fresh tender coconut water, sugarcane juice, buttermilk",
            "strictly_avoid": "All cooking oils, butter, ghee, meat, alcohol, heavy pulses"
        },
        "precautions": [
            "Total abstinence from alcohol and hepatotoxic medications",
            "Perform Liver Function Tests (LFT) weekly to monitor Bilirubin levels"
        ]
    },
    "Urinary Tract Infection (UTI)": {
        "specialist": "Urologist / Nephrologist",
        "severity": "Moderate",
        "medicines": ["Nitrofurantoin 100 mg (Prescription Antibiotic)", "Potassium Citrate Oral Solution", "Paracetamol 650 mg"],
        "dosage": [
            "Nitrofurantoin: 1 capsule twice daily with meals for 5 days",
            "Urine Alkalinizer: 2 teaspoons in a glass of water 3 times a day",
            "Paracetamol: 1 tablet for pelvic discomfort as needed"
        ],
        "timings": ["Morning & Night with food", "Morning, Afternoon, Night", "As needed"],
        "home_remedies": [
            "Drink 3 to 4 liters of clean water daily to flush out urinary bacteria",
            "Drink unsweetened pure cranberry extract or juice",
            "Apply warm heating pad to lower abdomen for cramping"
        ],
        "health_cafe_diet": {
            "title": "Urinary Alkalizing & Hydration Plan",
            "breakfast": "Watermelon slices, oats with blueberries and pumpkin seeds",
            "lunch": "Steamed rice with boiled bottle gourd and fresh unsalted buttermilk",
            "dinner": "Light vegetable soup with cucumber salad and soft bread",
            "beverages": "Unsweetened cranberry juice, barley water, tender coconut water",
            "strictly_avoid": "Spicy chilies, caffeine, alcohol, artificial sweeteners, carbonated drinks"
        },
        "precautions": [
            "Do not hold urine; empty bladder whenever the urge arises",
            "Wipe from front to back after urinating to prevent bacterial entry",
            "Consult doctor if fever rises with lower back pain (kidney concern)"
        ]
    },
    "Anxiety & Chronic Fatigue": {
        "specialist": "Psychiatrist / Clinical Psychologist",
        "severity": "Mild to Moderate",
        "medicines": ["Ashwagandha Extract 500 mg", "Magnesium Glycinate 400 mg", "Melatonin 3 mg (for sleep)"],
        "dosage": [
            "Ashwagandha: 1 capsule twice daily with warm milk",
            "Magnesium: 1 tablet at dinner",
            "Melatonin: 1 tablet 30 minutes before bed"
        ],
        "timings": ["Morning & Night", "Night with dinner", "30 mins before bedtime"],
        "home_remedies": [
            "4-7-8 Breathing technique (Inhale 4s, Hold 7s, Exhale 8s) for 10 minutes",
            "Spend 20 minutes in natural morning sunlight",
            "Digital detox: shut off smartphones and screens 1 hour before sleeping"
        ],
        "health_cafe_diet": {
            "title": "Neuro-Calming Serotonin Boosting Plan",
            "breakfast": "Warm rolled oats with crushed walnuts, sliced banana, and dark chocolate nibs",
            "lunch": "Quinoa bowl with avocado, chickpeas, spinach, and sunflower seeds",
            "dinner": "Warm sweet potato and lentil soup with chamomile tea",
            "beverages": "Chamomile tea, warm spiced nutmeg milk, green matcha tea",
            "strictly_avoid": "Excessive espresso coffee, energy drinks, refined sugars, late-night alcohol"
        },
        "precautions": [
            "Reach out to certified therapists or support groups",
            "Exercise moderately daily to release natural endorphins",
            "Avoid isolation; spend time outdoors and with supportive loved ones"
        ]
    }
}


def generate_synthetic_dataset(samples_per_disease=150):
    """
    Generates a realistic multi-symptom dataset with natural clinical noise.
    """
    data_rows = []
    labels = []

    for disease_name, profile in DISEASE_PROFILES.items():
        primary = profile["primary"]
        secondary = profile["secondary"]

        for _ in range(samples_per_disease):
            # Create a 0-filled symptom vector
            symptom_vector = {s: 0 for s in SYMPTOMS_LIST}

            # Primary symptoms have 85-95% probability of presenting
            for s in primary:
                if np.random.rand() < 0.90:
                    symptom_vector[s] = 1

            # Secondary symptoms have 30-60% probability
            for s in secondary:
                if np.random.rand() < 0.45:
                    symptom_vector[s] = 1

            # Random general noise (5% chance of sporadic symptom)
            for s in SYMPTOMS_LIST:
                if s not in primary and s not in secondary:
                    if np.random.rand() < 0.04:
                        symptom_vector[s] = 1

            # Ensure at least 2 symptoms are present
            if sum(symptom_vector.values()) < 2:
                for s in primary[:2]:
                    symptom_vector[s] = 1

            data_rows.append(symptom_vector)
            labels.append(disease_name)

    df = pd.DataFrame(data_rows)
    return df, labels


def train_and_save_model():
    print("Generating disease-symptom clinical training dataset...")
    df, labels = generate_synthetic_dataset(samples_per_disease=180)

    print(f"Dataset generated: {len(df)} samples across {len(set(labels))} diseases.")

    X = df[SYMPTOMS_LIST]
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Random Forest Classifier (100 estimators)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Training Accuracy on Test Set: {acc * 100:.2f}%")

    os.makedirs("models", exist_ok=True)

    # Save model and artifacts
    joblib.dump(clf, "models/disease_rf_model.joblib")
    joblib.dump(SYMPTOMS_LIST, "models/symptoms_list.joblib")
    joblib.dump(CLINICAL_KNOWLEDGE, "models/clinical_knowledge.joblib")

    print("ML Model, Symptoms Metadata, and Clinical Knowledge saved to models/ successfully!")


if __name__ == "__main__":
    train_and_save_model()
