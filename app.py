"""
Smart Healthcare & Health Cafe Platform - Main Application Server
Powered by Flask, SQLAlchemy (SQLite), and Scikit-Learn Random Forest ML Engine.
Features Dual Role-Based Portals (Admin & Patient) with Left Sidebar Navigation.
"""

import os
import json
from datetime import datetime
from functools import wraps
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, send_file

# Internal Modules
from models_db import db, User, PrescriptionRecord, Appointment
from ml_engine import predict_disease, CATEGORIZED_SYMPTOMS

app = Flask(__name__)
app.secret_key = 'smarthealth-ai-secret-key-2026'

# Configure persistent SQLite database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'smarthealth.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)


# ----------------------------------------------------------------------------
# Database Initialization & Default Users Seeding (Admin + Patient)
# ----------------------------------------------------------------------------
def init_database():
    with app.app_context():
        db.create_all()

        # 1. Seed Master Administrator (polamsettiramu@gmail.com)
        admin_user = User.query.filter_by(email='polamsettiramu@gmail.com').first()
        if not admin_user:
            admin_user = User(
                email='polamsettiramu@gmail.com',
                full_name='Ramu Polamsetti (Administrator)',
                role='admin',
                phone='+91 9000000000',
                age=35,
                gender='Male',
                blood_group='O+',
                address='Smart Healthcare Central Headquarters, Medical Tech Park',
                emergency_contact='Hospital Ops - 9000000001',
                medical_history='Master System Administrator'
            )
            admin_user.set_password('admin@123')
            db.session.add(admin_user)
            print(" * Admin seeded: polamsettiramu@gmail.com / admin@123")
        else:
            admin_user.role = 'admin'
            admin_user.full_name = 'Ramu Polamsetti (Administrator)'
            admin_user.set_password('admin@123')

        # 2. Seed Default Patient User
        patient_user = User.query.filter_by(email='user@example.com').first()
        if not patient_user:
            patient_user = User(
                email='user@example.com',
                full_name='Dr. Alex Reynolds',
                role='patient',
                phone='+91 9876543210',
                age=28,
                gender='Female',
                blood_group='O+',
                address='Jubilee Hills, Hyderabad, Telangana',
                emergency_contact='Kavitha Reynolds (Mother) - 9876500000',
                medical_history='Mild dust allergy, no chronic illness'
            )
            patient_user.set_password('password123')
            db.session.add(patient_user)
            print(" * Patient seeded: user@example.com / password123")

        # 3. Seed Clinical Doctors & Specialists
        doctor_seeds = [
            {
                "email": "dranil@smarthealth.com",
                "full_name": "Dr. Anil Kumar",
                "specialization": "General Medicine",
                "hospital": "Apollo Hospitals, Jubilee Hills",
                "qualifications": "MBBS, MD (General Medicine - AIIMS Delhi), FICP",
                "consultation_fee": "₹600",
                "phone": "+91 9876543210",
                "address": "Block A, 2nd Floor, OPD Room No. 204, Jubilee Hills, Hyderabad"
            },
            {
                "email": "drradhika@smarthealth.com",
                "full_name": "Dr. S. Radhika",
                "specialization": "Cardiology",
                "hospital": "Care Heart Institute, Banjara Hills",
                "qualifications": "MBBS, MD, DM (Cardiology), FACC (USA)",
                "consultation_fee": "₹900",
                "phone": "+91 9988776655",
                "address": "Cardiology Tower, 3rd Floor, OPD Room No. 312, Banjara Hills, Hyderabad"
            },
            {
                "email": "drneha@smarthealth.com",
                "full_name": "Dr. Neha Sharma",
                "specialization": "Dermatology",
                "hospital": "Yashoda Hospitals, Secunderabad",
                "qualifications": "MBBS, MD (Dermatology - CMC Vellore)",
                "consultation_fee": "₹750",
                "phone": "+91 9123456780",
                "address": "OPD Block B, 1st Floor, OPD Room No. 118, Secunderabad"
            },
            {
                "email": "drraviteja@smarthealth.com",
                "full_name": "Dr. Ravi Teja",
                "specialization": "Neurology",
                "hospital": "KIMS Hospitals, Begumpet",
                "qualifications": "MBBS, MD, DM (Neurology - NIMS), Stroke Fellow",
                "consultation_fee": "₹850",
                "phone": "+91 9012345678",
                "address": "Bone & Neuro Pavilion, 4th Floor, Room No. 405, Begumpet"
            }
        ]
        for d in doctor_seeds:
            doc_user = User.query.filter_by(email=d["email"]).first()
            if not doc_user:
                doc_user = User(
                    email=d["email"],
                    full_name=d["full_name"],
                    role="doctor",
                    phone=d["phone"],
                    specialization=d["specialization"],
                    hospital=d["hospital"],
                    qualifications=d["qualifications"],
                    consultation_fee=d["consultation_fee"],
                    address=d["address"]
                )
                doc_user.set_password("doctor@123")
                db.session.add(doc_user)
                print(f" * Doctor seeded: {d['email']} / doctor@123")
            else:
                doc_user.role = "doctor"
                doc_user.full_name = d["full_name"]
                doc_user.specialization = d["specialization"]
                doc_user.hospital = d["hospital"]
                doc_user.qualifications = d["qualifications"]
                doc_user.consultation_fee = d["consultation_fee"]
                doc_user.address = d["address"]
                doc_user.set_password("doctor@123")

        db.session.commit()

        # Seed sample appointments for Dr. Anil Kumar if none exist
        anil_appts = Appointment.query.filter(Appointment.doctor_name.ilike('%Anil Kumar%')).count()
        if anil_appts == 0 and patient_user:
            sample_appts = [
                Appointment(
                    user_id=patient_user.id,
                    doctor_name="Dr. Anil Kumar",
                    specialization="General Medicine",
                    hospital="Apollo Hospitals, Jubilee Hills",
                    appointment_type="Video",
                    appointment_date="2026-10-15",
                    time_slot="10:00 AM - 10:30 AM",
                    doctor_phone="+91 9876543210",
                    patient_notes="Mild fever and seasonal dry cough for 3 days",
                    status="Confirmed"
                ),
                Appointment(
                    user_id=patient_user.id,
                    doctor_name="Dr. Anil Kumar",
                    specialization="General Medicine",
                    hospital="Apollo Hospitals, Jubilee Hills",
                    appointment_type="In-Person",
                    appointment_date="2026-10-16",
                    time_slot="11:30 AM - 12:30 PM",
                    doctor_phone="+91 9876543210",
                    patient_notes="Follow-up clinical blood pressure review",
                    status="Confirmed"
                ),
                Appointment(
                    user_id=patient_user.id,
                    doctor_name="Dr. Anil Kumar",
                    specialization="General Medicine",
                    hospital="Apollo Hospitals, Jubilee Hills",
                    appointment_type="Video",
                    appointment_date="2026-10-14",
                    time_slot="04:00 PM - 04:30 PM",
                    doctor_phone="+91 9876543210",
                    patient_notes="Discussion on lab reports for viral infection recovery",
                    status="Completed"
                ),
                Appointment(
                    user_id=patient_user.id,
                    doctor_name="Dr. Anil Kumar",
                    specialization="General Medicine",
                    hospital="Apollo Hospitals, Jubilee Hills",
                    appointment_type="In-Person",
                    appointment_date="2026-10-17",
                    time_slot="02:00 PM - 03:00 PM",
                    doctor_phone="+91 9876543210",
                    patient_notes="Routine health checkup and vitamin D evaluation",
                    status="Confirmed"
                )
            ]
            for sa in sample_appts:
                db.session.add(sa)
            db.session.commit()

        print(" * Database initialization complete!")

# Automatically ensure DB tables & seed users exist on WSGI/Gunicorn import
with app.app_context():
    init_database()


# ----------------------------------------------------------------------------
# Context Processor & Auth Hooks
# ----------------------------------------------------------------------------
@app.context_processor
def inject_user():
    current_user = None
    if 'user_id' in session:
        current_user = User.query.get(session['user_id'])
    return dict(current_user=current_user)


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in to access this healthcare service.", "warning")
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return wrapper


def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in with administrator credentials.", "warning")
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            flash("Access denied: Administrator privileges required.", "danger")
            return redirect(url_for('dashboard'))
        return func(*args, **kwargs)
    return wrapper


def doctor_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in with Doctor & Specialist credentials.", "warning")
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_doctor:
            flash("Access denied: Doctor portal access required.", "danger")
            return redirect(url_for('dashboard'))
        return func(*args, **kwargs)
    return wrapper


# ----------------------------------------------------------------------------
# AUTHENTICATION ROUTES
# ----------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            elif user.is_doctor:
                return redirect(url_for('doctor_dashboard'))
            return redirect(url_for('dashboard'))

    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        user = User.query.filter_by(email=email).first()
        if not user:
            error = f"The email address '{email}' is not registered in our system. Only existing accounts are allowed. Please check your email or sign up as a new patient."
        elif not user.check_password(password):
            error = "Incorrect password entered for this account. Please verify your credentials and try again."
        else:
            # Update last login timestamp
            user.last_login = datetime.utcnow()
            db.session.commit()

            session['user_id'] = user.id
            session['user_email'] = user.email
            session['user_role'] = user.role

            if user.is_admin:
                flash(f"Welcome Administrator, {user.full_name}!", "success")
                return redirect(url_for('admin_dashboard'))
            elif user.is_doctor:
                flash(f"Welcome to Clinical Portal, {user.full_name}!", "success")
                return redirect(url_for('doctor_dashboard'))
            else:
                flash(f"Welcome back, {user.full_name}!", "success")
                return redirect(url_for('dashboard'))

    return render_template('login.html', error=error)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    error = None
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        age = request.form.get('age', 28)
        gender = request.form.get('gender', 'Not specified')
        blood_group = request.form.get('blood_group', 'O+')
        phone = request.form.get('phone', '').strip()

        if not email or not password or not full_name:
            error = "Please fill in all mandatory fields (*)."
        elif User.query.filter_by(email=email).first():
            error = "An account with this email address already exists."
        else:
            new_user = User(
                email=email,
                full_name=full_name,
                role='patient',
                age=int(age) if str(age).isdigit() else 28,
                gender=gender,
                blood_group=blood_group,
                phone=phone,
                last_login=datetime.utcnow()
            )
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()

            session['user_id'] = new_user.id
            session['user_email'] = new_user.email
            session['user_role'] = new_user.role
            flash("Patient account created successfully! Welcome to Smart Healthcare.", "success")
            return redirect(url_for('dashboard'))

    return render_template('signup.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been signed out successfully.", "success")
    return redirect(url_for('login'))


# ----------------------------------------------------------------------------
# PATIENT DASHBOARD & PROFILE
# ----------------------------------------------------------------------------

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    if user.is_admin:
        return redirect(url_for('admin_dashboard'))
    if user.is_doctor:
        return redirect(url_for('doctor_dashboard'))

    prescriptions = PrescriptionRecord.query.filter_by(user_id=user.id).order_by(PrescriptionRecord.created_at.desc()).limit(10).all()
    appointments = Appointment.query.filter_by(user_id=user.id).order_by(Appointment.created_at.desc()).limit(10).all()

    return render_template('dashboard.html', prescriptions=prescriptions, appointments=appointments)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        user.full_name = request.form.get('full_name', '').strip() or user.full_name
        user.phone = request.form.get('phone', '').strip()
        user.address = request.form.get('address', '').strip()

        if user.is_doctor:
            user.specialization = request.form.get('specialization', user.specialization)
            user.hospital = request.form.get('hospital', user.hospital)
            user.qualifications = request.form.get('qualifications', user.qualifications)
            user.consultation_fee = request.form.get('consultation_fee', user.consultation_fee)
            db.session.commit()
            flash("Doctor credentials and professional profile updated successfully!", "success")
            return redirect(url_for('profile'))
        else:
            age_val = request.form.get('age', '')
            if age_val and age_val.isdigit():
                user.age = int(age_val)
            user.gender = request.form.get('gender', user.gender)
            user.blood_group = request.form.get('blood_group', user.blood_group)
            user.emergency_contact = request.form.get('emergency_contact', '').strip()
            user.medical_history = request.form.get('medical_history', '').strip()
            db.session.commit()
            flash("Your medical profile & Health ID details have been updated!", "success")
            return redirect(url_for('profile'))

    user_prescriptions = []
    if not user.is_doctor:
        user_prescriptions = PrescriptionRecord.query.filter_by(user_id=user.id).order_by(PrescriptionRecord.created_at.desc()).all()
    return render_template('profile.html', user_prescriptions=user_prescriptions)


# ----------------------------------------------------------------------------
# AI PRESCRIPTION & CLINICAL DIAGNOSIS
# ----------------------------------------------------------------------------

@app.route('/prescription', methods=['GET', 'POST'])
@login_required
def prescription():
    user = User.query.get(session['user_id'])
    if user.is_doctor:
        flash("You are signed in as a Doctor. AI Diagnosis is a patient service. Please use your Doctor Dashboard.", "info")
        return redirect(url_for('doctor_dashboard'))

    if request.method == 'POST':
        patient_name = request.form.get('name', '').strip() or user.full_name
        age_str = request.form.get('age', '').strip()
        patient_age = int(age_str) if age_str.isdigit() else (user.age or 28)
        address = request.form.get('address', '').strip() or user.address
        time_period = request.form.get('time_period', '3 days').strip()
        symptoms_text = request.form.get('symptoms', '').strip()
        selected_tags_str = request.form.get('selected_symptoms', '').strip()

        # Parse selected tags list
        selected_tags = [t.strip() for t in selected_tags_str.split(',') if t.strip()]

        # Run Random Forest Machine Learning Diagnosis Engine
        result = predict_disease(
            symptoms_text=symptoms_text,
            selected_tags=selected_tags
        )

        # Combine descriptions
        combined_symptoms_desc = symptoms_text
        if selected_tags:
            readable_tags = [t.replace('_', ' ').title() for t in selected_tags]
            tag_summary = ", ".join(readable_tags)
            combined_symptoms_desc = f"{tag_summary}. {symptoms_text}".strip()

        # Save to SQLite Database
        record = PrescriptionRecord(
            user_id=user.id,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=user.gender,
            patient_address=address,
            duration=time_period,
            symptoms=combined_symptoms_desc,
            diagnosed_disease=result['disease'],
            confidence_score=result['confidence'],
            specialist=result['specialist'],
            severity=result['severity'],
            medicines_json=json.dumps(result['medicines']),
            dosage_json=json.dumps(result['dosage']),
            timings_json=json.dumps(result['timings']),
            home_remedies_json=json.dumps(result['home_remedies']),
            precautions_json=json.dumps(result['precautions']),
            diet_plan_json=json.dumps(result.get('health_cafe_diet', {}))
        )
        db.session.add(record)
        db.session.commit()

        return render_template(
            'prescription.html',
            result=result,
            patient_name=patient_name,
            patient_age=patient_age,
            time_period=time_period,
            symptoms=combined_symptoms_desc,
            rx_date=datetime.now().strftime('%d %B %Y')
        )

    return render_template(
        'prescription.html',
        result=None,
        categorized_symptoms=CATEGORIZED_SYMPTOMS
    )


@app.route('/prescription/<int:record_id>')
@login_required
def view_prescription(record_id):
    user = User.query.get(session['user_id'])
    if user.is_admin:
        record = PrescriptionRecord.query.get_or_404(record_id)
    else:
        record = PrescriptionRecord.query.filter_by(id=record_id, user_id=user.id).first_or_404()

    return render_template('prescription_detail.html', record=record)


@app.route('/prescription/<int:record_id>/delete', methods=['POST'])
@login_required
def delete_prescription(record_id):
    user = User.query.get(session['user_id'])
    if user.is_admin:
        record = PrescriptionRecord.query.get_or_404(record_id)
    else:
        record = PrescriptionRecord.query.filter_by(id=record_id, user_id=user.id).first_or_404()

    disease_name = record.diagnosed_disease
    db.session.delete(record)
    db.session.commit()
    flash(f"Diagnosis record for '{disease_name}' has been successfully deleted.", "info")

    referrer = request.referrer or ''
    if 'profile' in referrer:
        return redirect(url_for('profile'))
    elif 'admin' in referrer:
        return redirect(url_for('admin_prescriptions'))
    return redirect(url_for('dashboard'))



# ----------------------------------------------------------------------------
# RICH CLINICAL SPECIALISTS DATA (VIDEO TELEHEALTH & HOSPITAL OPD)
# ----------------------------------------------------------------------------

VIDEO_DOCTORS = [
    # 1. General Medicine
    {
        "category": "General Medicine",
        "name": "Dr. Anil Kumar",
        "specialization": "Senior General Physician & Infectious Diseases",
        "qualifications": "MBBS, MD (General Medicine - AIIMS Delhi), FICP",
        "experience": "14+ Years Clinical Experience",
        "hospital": "Apollo Hospitals Telehealth",
        "fee": "₹600",
        "rating": "4.9",
        "reviews": "1,420+ Consultations",
        "languages": "English, Hindi, Telugu",
        "next_slot": "Available in 15 mins (Live Now)",
        "focus": "Viral fevers, Dengue/Malaria recovery, Persistent cough, Preventive health",
        "phone": "+91 9876543210"
    },
    {
        "category": "General Medicine",
        "name": "Dr. Kavita Deshmukh",
        "specialization": "Consultant Physician & Family Medicine",
        "qualifications": "MBBS, DNB (Family Medicine), Dip. Geriatric Care",
        "experience": "12+ Years Clinical Experience",
        "hospital": "Fortis Virtual Care Clinic",
        "fee": "₹550",
        "rating": "4.86",
        "reviews": "980+ Consultations",
        "languages": "English, Hindi, Marathi",
        "next_slot": "Today 4:00 PM",
        "focus": "Chronic lifestyle management, Hypertension, Seasonal infections, Elderly care",
        "phone": "+91 9822334455"
    },

    # 2. Cardiology
    {
        "category": "Cardiology",
        "name": "Dr. S. Radhika",
        "specialization": "Chief Interventional Cardiologist",
        "qualifications": "MBBS, MD, DM (Cardiology - Gold Medalist), FACC (USA)",
        "experience": "16+ Years Clinical Experience",
        "hospital": "Care Heart Institute",
        "fee": "₹900",
        "rating": "4.95",
        "reviews": "2,180+ Consultations",
        "languages": "English, Telugu",
        "next_slot": "Today 3:30 PM",
        "focus": "Hypertension, Chest discomfort, Lipid disorders, Post-angioplasty care",
        "phone": "+91 9988776655"
    },
    {
        "category": "Cardiology",
        "name": "Dr. Vikram Malhotra",
        "specialization": "Consultant Clinical Cardiologist & Electrophysiologist",
        "qualifications": "MBBS, MD, DM (Cardiology), FESC",
        "experience": "13+ Years Clinical Experience",
        "hospital": "Max Heart & Vascular Clinic",
        "fee": "₹850",
        "rating": "4.89",
        "reviews": "1,340+ Consultations",
        "languages": "English, Hindi, Punjabi",
        "next_slot": "Tomorrow 11:00 AM",
        "focus": "Heart palpitations, Arrhythmia, Heart failure monitoring, Cholesterol management",
        "phone": "+91 9811223344"
    },

    # 3. Dermatology
    {
        "category": "Dermatology",
        "name": "Dr. Meena Patel",
        "specialization": "Consultant Dermatologist & Allergy Specialist",
        "qualifications": "MBBS, MD (Dermatology & Leprosy - CMC Vellore)",
        "experience": "10+ Years Clinical Experience",
        "hospital": "Sunshine Skin & Allergy Center",
        "fee": "₹700",
        "rating": "4.85",
        "reviews": "1,150+ Consultations",
        "languages": "English, Hindi, Gujarati",
        "next_slot": "Today 4:30 PM",
        "focus": "Eczema, Psoriasis, Acute rashes & urticaria, Skin barrier repair",
        "phone": "+91 9123456780"
    },
    {
        "category": "Dermatology",
        "name": "Dr. Siddharth Verma",
        "specialization": "Aesthetic Dermatologist & Trichologist",
        "qualifications": "MBBS, DVD, MD (Dermatology)",
        "experience": "9+ Years Clinical Experience",
        "hospital": "Kaya Skin Clinic Online",
        "fee": "₹650",
        "rating": "4.82",
        "reviews": "890+ Consultations",
        "languages": "English, Hindi",
        "next_slot": "Today 5:00 PM",
        "focus": "Acne vulgaris, Hair fall & alopecia, Fungal skin infections, Pigmentation",
        "phone": "+91 9944556677"
    },

    # 4. Neurology
    {
        "category": "Neurology",
        "name": "Dr. Ravi Teja",
        "specialization": "Consultant Neurologist & Migraine Specialist",
        "qualifications": "MBBS, MD, DM (Neurology - NIMS), Fellowship in Stroke",
        "experience": "12+ Years Clinical Experience",
        "hospital": "Global Neuro & Spine Institute",
        "fee": "₹850",
        "rating": "4.9",
        "reviews": "980+ Consultations",
        "languages": "English, Telugu, Hindi",
        "next_slot": "Tomorrow 10:00 AM",
        "focus": "Chronic migraine, Cluster headaches, Cervical spine strain, Vertigo & tremors",
        "phone": "+91 9012345678"
    },
    {
        "category": "Neurology",
        "name": "Dr. Ananya Sen",
        "specialization": "Senior Consultant Neurologist & Neurophysiologist",
        "qualifications": "MBBS, MD, DM (Neurology - PGIMER)",
        "experience": "15+ Years Clinical Experience",
        "hospital": "Narayana Brain Health Virtual",
        "fee": "₹900",
        "rating": "4.92",
        "reviews": "1,670+ Consultations",
        "languages": "English, Bengali, Hindi",
        "next_slot": "Tomorrow 2:00 PM",
        "focus": "Peripheral neuropathy, Epilepsy management, Sleep disorders, Memory issues",
        "phone": "+91 9833445566"
    },

    # 5. Pediatrics
    {
        "category": "Pediatrics",
        "name": "Dr. Priya Sharma",
        "specialization": "Senior Pediatrician & Child Health Specialist",
        "qualifications": "MBBS, MD (Pediatrics), DNB (Pediatrics), IAP Member",
        "experience": "11+ Years Clinical Experience",
        "hospital": "Rainbow Children's Hospital",
        "fee": "₹650",
        "rating": "4.92",
        "reviews": "1,850+ Consultations",
        "languages": "English, Hindi",
        "next_slot": "Today 5:15 PM",
        "focus": "Child viral fevers, Pediatric nutrition, Respiratory infections, Growth milestones",
        "phone": "+91 8899001122"
    },
    {
        "category": "Pediatrics",
        "name": "Dr. Rohan Gupta",
        "specialization": "Consultant Pediatrician & Adolescent Care",
        "qualifications": "MBBS, DCH, DNB (Pediatrics)",
        "experience": "8+ Years Clinical Experience",
        "hospital": "Cloudnine Pediatric Telecare",
        "fee": "₹600",
        "rating": "4.88",
        "reviews": "920+ Consultations",
        "languages": "English, Hindi",
        "next_slot": "Today 6:00 PM",
        "focus": "Child immunization, Colic & gastric issues, Pediatric allergies, Teen health",
        "phone": "+91 9822114433"
    },

    # 6. Gastroenterology
    {
        "category": "Gastroenterology",
        "name": "Dr. K. Srinivas Rao",
        "specialization": "Consultant Gastroenterologist & Liver Specialist",
        "qualifications": "MBBS, MD, DM (Gastroenterology - AIG), FACG",
        "experience": "15+ Years Clinical Experience",
        "hospital": "Asian Institute of Gastroenterology (AIG)",
        "fee": "₹800",
        "rating": "4.88",
        "reviews": "1,620+ Consultations",
        "languages": "English, Telugu, Hindi",
        "next_slot": "Tomorrow 11:30 AM",
        "focus": "GERD & Acid Reflux, Gastritis, Fatty liver management, Gut microbiome",
        "phone": "+91 9888776611"
    },
    {
        "category": "Gastroenterology",
        "name": "Dr. Shalini Joshi",
        "specialization": "Consultant Hepatologist & Digestive Specialist",
        "qualifications": "MBBS, MD (Medicine), DM (Gastroenterology)",
        "experience": "11+ Years Clinical Experience",
        "hospital": "Manipal Digestive Health Clinic",
        "fee": "₹750",
        "rating": "4.85",
        "reviews": "1,050+ Consultations",
        "languages": "English, Hindi, Kannada",
        "next_slot": "Tomorrow 3:30 PM",
        "focus": "Irritable bowel syndrome (IBS), Constipation, Celiac disease, Liver enzymes",
        "phone": "+91 9845012345"
    },

    # 7. Orthopedics
    {
        "category": "Orthopedics",
        "name": "Dr. Arvind Swamy",
        "specialization": "Senior Orthopedic & Sports Medicine Consultant",
        "qualifications": "MBBS, MS (Orthopedics - AIIMS), M.Ch (Ortho - UK)",
        "experience": "16+ Years Clinical Experience",
        "hospital": "Sports Ortho Virtual Clinic",
        "fee": "₹800",
        "rating": "4.91",
        "reviews": "2,400+ Consultations",
        "languages": "English, Tamil, Hindi",
        "next_slot": "Today 6:30 PM",
        "focus": "Knee arthritis, Ligament tears, Frozen shoulder, Ergonomic posture pain",
        "phone": "+91 9840123456"
    },
    {
        "category": "Orthopedics",
        "name": "Dr. Manoj Chawla",
        "specialization": "Spine & Joint Replacement Specialist",
        "qualifications": "MBBS, MS (Ortho), Fellowship in Spine Surgery",
        "experience": "12+ Years Clinical Experience",
        "hospital": "Joint & Spine Care Telehealth",
        "fee": "₹750",
        "rating": "4.87",
        "reviews": "1,120+ Consultations",
        "languages": "English, Hindi, Punjabi",
        "next_slot": "Tomorrow 10:30 AM",
        "focus": "Sciatica, Lumbar spondylosis, Slip disc rehab, Osteoporosis management",
        "phone": "+91 9814012345"
    },

    # 8. Pulmonology
    {
        "category": "Pulmonology",
        "name": "Dr. Rajesh Nair",
        "specialization": "Consultant Pulmonologist & Sleep Medicine Specialist",
        "qualifications": "MBBS, MD (Pulmonary Medicine - CMC), FCCP",
        "experience": "14+ Years Clinical Experience",
        "hospital": "Apex Chest & Respiratory Care",
        "fee": "₹750",
        "rating": "4.89",
        "reviews": "1,310+ Consultations",
        "languages": "English, Malayalam, Hindi",
        "next_slot": "Today 4:45 PM",
        "focus": "Asthma inhaler therapy, Chronic bronchitis, Post-COVID lung rehab, Sleep apnea",
        "phone": "+91 9847012345"
    },
    {
        "category": "Pulmonology",
        "name": "Dr. Sunita Kulkarni",
        "specialization": "Chest Physician & Allergy Immunotherapist",
        "qualifications": "MBBS, DNB (Respiratory Diseases), European Dip. Adult Resp. Med",
        "experience": "10+ Years Clinical Experience",
        "hospital": "PulmoCare Telehealth",
        "fee": "₹700",
        "rating": "4.84",
        "reviews": "890+ Consultations",
        "languages": "English, Marathi, Hindi",
        "next_slot": "Tomorrow 1:00 PM",
        "focus": "Allergic bronchitis, Smoker's cough, Pneumonia recovery, Inhaler technique",
        "phone": "+91 9820012345"
    },

    # 9. Endocrinology
    {
        "category": "Endocrinology",
        "name": "Dr. Alok Bansal",
        "specialization": "Consultant Endocrinologist & Diabetologist",
        "qualifications": "MBBS, MD (Medicine), DM (Endocrinology - PGI)",
        "experience": "15+ Years Clinical Experience",
        "hospital": "EndoHealth & Diabetes Virtual Clinic",
        "fee": "₹850",
        "rating": "4.93",
        "reviews": "2,100+ Consultations",
        "languages": "English, Hindi",
        "next_slot": "Today 5:30 PM",
        "focus": "Type 1 & 2 Diabetes, Insulin optimization, Thyroid nodules & Hashimotos, PCOD",
        "phone": "+91 9810012345"
    },
    {
        "category": "Endocrinology",
        "name": "Dr. Ritu Saxena",
        "specialization": "Metabolic Specialist & Clinical Diabetologist",
        "qualifications": "MBBS, MD, Fellowship in Diabetes (Royal Liverpool Academy)",
        "experience": "11+ Years Clinical Experience",
        "hospital": "Metabolic Care Virtual",
        "fee": "₹750",
        "rating": "4.86",
        "reviews": "1,180+ Consultations",
        "languages": "English, Hindi",
        "next_slot": "Tomorrow 11:00 AM",
        "focus": "Hypothyroidism, Gestational diabetes, Metabolic syndrome, Hormonal weight",
        "phone": "+91 9811012345"
    },

    # 10. Psychiatry
    {
        "category": "Psychiatry",
        "name": "Dr. Maya Krishnan",
        "specialization": "Consultant Psychiatrist & Behavioral Therapist",
        "qualifications": "MBBS, MD (Psychiatry - NIMHANS), MRCPsych (UK)",
        "experience": "13+ Years Clinical Experience",
        "hospital": "MindWell Behavioral Telehealth",
        "fee": "₹900",
        "rating": "4.96",
        "reviews": "1,750+ Consultations",
        "languages": "English, Tamil, Hindi",
        "next_slot": "Today 7:00 PM",
        "focus": "Anxiety & Panic disorder, Clinical depression, Work burnout, Insomnia therapy",
        "phone": "+91 9841012345"
    },
    {
        "category": "Psychiatry",
        "name": "Dr. Vivek Oberoi",
        "specialization": "Neuropsychiatrist & Addiction Medicine Consultant",
        "qualifications": "MBBS, DNB (Psychiatry), Dip. Psychological Medicine",
        "experience": "10+ Years Clinical Experience",
        "hospital": "Hope Mental Health Center",
        "fee": "₹800",
        "rating": "4.88",
        "reviews": "920+ Consultations",
        "languages": "English, Hindi",
        "next_slot": "Tomorrow 4:00 PM",
        "focus": "OCD, Adult ADHD, Stress-induced psychosomatic symptoms, Mood stability",
        "phone": "+91 9818012345"
    },

    # 11. Gynecology
    {
        "category": "Gynecology",
        "name": "Dr. Sunita Reddy",
        "specialization": "Senior Obstetrician & Gynecological Consultant",
        "qualifications": "MBBS, MS (OBG), MRCOG (London), FICOG",
        "experience": "17+ Years Clinical Experience",
        "hospital": "Women's Health Virtual Care",
        "fee": "₹800",
        "rating": "4.94",
        "reviews": "2,600+ Consultations",
        "languages": "English, Telugu, Hindi",
        "next_slot": "Today 3:45 PM",
        "focus": "PCOS/PCOD balance, Irregular cycles, Menopause transitions, Prenatal advice",
        "phone": "+91 9849012345"
    },
    {
        "category": "Gynecology",
        "name": "Dr. Deepa Nambiar",
        "specialization": "Consultant Gynecologist & Reproductive Health Specialist",
        "qualifications": "MBBS, DGO, DNB (OBG), Fellowship in Reproductive Med",
        "experience": "12+ Years Clinical Experience",
        "hospital": "Motherhood Telecare Network",
        "fee": "₹750",
        "rating": "4.9",
        "reviews": "1,420+ Consultations",
        "languages": "English, Malayalam, Hindi",
        "next_slot": "Tomorrow 10:00 AM",
        "focus": "Fertility counseling, Pelvic pain, Endometriosis management, Contraception",
        "phone": "+91 9848012345"
    },

    # 12. ENT
    {
        "category": "ENT",
        "name": "Dr. Farhan Qureshi",
        "specialization": "Consultant ENT, Head & Neck Surgeon",
        "qualifications": "MBBS, MS (ENT - Safdarjung Hospital Delhi), DNB (ENT)",
        "experience": "12+ Years Clinical Experience",
        "hospital": "Apex ENT Teleconsult",
        "fee": "₹650",
        "rating": "4.87",
        "reviews": "1,140+ Consultations",
        "languages": "English, Hindi, Urdu",
        "next_slot": "Today 6:15 PM",
        "focus": "Chronic allergic rhinitis, Sinusitis, Tinnitus & ear fullness, Tonsil infections",
        "phone": "+91 9821012345"
    },
    {
        "category": "ENT",
        "name": "Dr. Sneha Agarwal",
        "specialization": "Otolaryngologist & Voice Care Specialist",
        "qualifications": "MBBS, DLO, DNB (Otorhinolaryngology)",
        "experience": "9+ Years Clinical Experience",
        "hospital": "Sinus & Voice Health Virtual",
        "fee": "₹600",
        "rating": "4.83",
        "reviews": "780+ Consultations",
        "languages": "English, Hindi",
        "next_slot": "Tomorrow 12:00 PM",
        "focus": "Hoarseness of voice, Acid reflux throat burn, Eustachian tube dysfunction",
        "phone": "+91 9819012345"
    },

    # 13. Oncology
    {
        "category": "Oncology",
        "name": "Dr. Sandeep Ghosh",
        "specialization": "Senior Medical Oncologist & Cancer Consultant",
        "qualifications": "MBBS, MD (Medicine), DM (Medical Oncology - Tata Memorial)",
        "experience": "18+ Years Clinical Experience",
        "hospital": "Comprehensive Cancer Care Virtual",
        "fee": "₹1,000",
        "rating": "4.97",
        "reviews": "1,980+ Consultations",
        "languages": "English, Bengali, Hindi",
        "next_slot": "Tomorrow 5:00 PM",
        "focus": "Second opinions on biopsy/PET scans, Chemotherapy side effects, Cancer screening",
        "phone": "+91 9831012345"
    },

    # 14. Ophthalmology
    {
        "category": "Ophthalmology",
        "name": "Dr. Poornima Iyer",
        "specialization": "Consultant Ophthalmologist & Cornea Specialist",
        "qualifications": "MBBS, MS (Ophthalmology), DNB, FICO (UK)",
        "experience": "11+ Years Clinical Experience",
        "hospital": "Vision & Eye Care Telehealth",
        "fee": "₹600",
        "rating": "4.89",
        "reviews": "1,220+ Consultations",
        "languages": "English, Tamil, Hindi",
        "next_slot": "Today 4:15 PM",
        "focus": "Digital eye strain & dry eye syndrome, Conjunctivitis, Blurred vision triage",
        "phone": "+91 9842012345"
    },

    # 15. Nephrology
    {
        "category": "Nephrology",
        "name": "Dr. Harish Bhatt",
        "specialization": "Chief Nephrologist & Renal Transplant Consultant",
        "qualifications": "MBBS, MD, DM (Nephrology - AIIMS), FASN (USA)",
        "experience": "16+ Years Clinical Experience",
        "hospital": "Kidney Care & Dialysis Virtual Network",
        "fee": "₹900",
        "rating": "4.94",
        "reviews": "1,870+ Consultations",
        "languages": "English, Hindi, Gujarati",
        "next_slot": "Tomorrow 11:45 AM",
        "focus": "Chronic kidney disease (CKD), Proteinuria in diabetes, Kidney stone prevention",
        "phone": "+91 9825012345"
    }
]

OFFLINE_DOCTORS = [
    # 1. General Medicine
    {
        "category": "General Medicine",
        "name": "Dr. Anil Kumar",
        "specialization": "Senior General Physician & Internal Medicine",
        "qualifications": "MBBS, MD (General Medicine - AIIMS Delhi)",
        "hospital": "Apollo Hospitals, Jubilee Hills",
        "address": "Road No. 72, Opposite Bharatiya Vidya Bhavan, Film Nagar, Jubilee Hills, Hyderabad - 500033",
        "room": "Block A, 2nd Floor, OPD Room No. 204",
        "timings": "Mon – Sat • 10:00 AM – 1:30 PM & 5:00 PM – 7:30 PM",
        "fee": "₹800",
        "rating": "4.9",
        "reviews": "3,400+ Patients",
        "facilities": "24/7 Diagnostic Pathology Lab, In-house Pharmacy, Emergency Trauma Unit, Digital X-Ray & ECG",
        "phone": "+91 9876543210"
    },
    {
        "category": "General Medicine",
        "name": "Dr. Kavita Deshmukh",
        "specialization": "Consultant Physician & Family Medicine",
        "qualifications": "MBBS, DNB (Family Medicine), Dip. Geriatric Care",
        "hospital": "Fortis Hospital, Bannerghatta Road",
        "address": "154/9, Bannerghatta Main Road, Opposite IIMB, Bilekahalli, Bengaluru - 560076",
        "room": "Tower 1, 1st Floor, OPD Chamber 108",
        "timings": "Mon – Fri • 9:30 AM – 1:00 PM",
        "fee": "₹750",
        "rating": "4.86",
        "reviews": "2,150+ Patients",
        "facilities": "Full Body Executive Health Checkup Unit, Daycare Ward, Ultrasound Suite, Rapid Blood Analyzer",
        "phone": "+91 9822334455"
    },

    # 2. Cardiology
    {
        "category": "Cardiology",
        "name": "Dr. Suresh Reddy",
        "specialization": "Chief Interventional Cardiologist",
        "qualifications": "MBBS, MD, DM (Cardiology), FACC",
        "hospital": "Care Hospitals, Banjara Hills",
        "address": "Road No. 1, Prem Nagar, Banjara Hills, Hyderabad - 500034",
        "room": "Cardiology Tower, 3rd Floor, OPD Room No. 312",
        "timings": "Tue – Sun • 11:00 AM – 3:30 PM",
        "fee": "₹1,000",
        "rating": "4.95",
        "reviews": "4,100+ Cardiac Procedures",
        "facilities": "State-of-the-Art Cath Lab, Cardiac ICU, Color Doppler 3D Echo, 24x7 Chest Pain Clinic",
        "phone": "+91 9876501234"
    },
    {
        "category": "Cardiology",
        "name": "Dr. Vikram Malhotra",
        "specialization": "Senior Consultant Interventional Cardiologist",
        "qualifications": "MBBS, MD, DM (Cardiology), FESC",
        "hospital": "Max Super Speciality Hospital, Saket",
        "address": "1, 2, Press Enclave Road, Mandir Marg, Saket, New Delhi - 110017",
        "room": "Cardio-Vascular Center, Ground Floor, OPD Room 18",
        "timings": "Mon – Sat • 10:00 AM – 2:00 PM",
        "fee": "₹1,100",
        "rating": "4.91",
        "reviews": "3,800+ Procedures",
        "facilities": "Hybrid Cath Lab, TMT & Holter Monitoring Lab, Advanced Cardiac CT, Coronary Care Unit",
        "phone": "+91 9811223344"
    },

    # 3. Dermatology
    {
        "category": "Dermatology",
        "name": "Dr. Neha Sharma",
        "specialization": "Consultant Dermatologist & Dermatosurgeon",
        "qualifications": "MBBS, MD (Dermatology - CMC Vellore)",
        "hospital": "Yashoda Hospitals, Secunderabad",
        "address": "Alexander Road, Kirloskar Colony, Secunderabad, Telangana - 500003",
        "room": "OPD Block B, 1st Floor, OPD Room No. 118",
        "timings": "Mon – Fri • 2:00 PM – 6:30 PM",
        "fee": "₹750",
        "rating": "4.85",
        "reviews": "2,900+ Patients Treated",
        "facilities": "Advanced Dermoscopy, Allergy Patch Testing Unit, Phototherapy, Minor Dermatosurgical Suite",
        "phone": "+91 9988776655"
    },
    {
        "category": "Dermatology",
        "name": "Dr. Siddharth Verma",
        "specialization": "Senior Clinical Dermatologist & Trichologist",
        "qualifications": "MBBS, DVD, MD (Dermatology)",
        "hospital": "Kaya Skin Clinic & Hospital, Jubilee Hills",
        "address": "Plot No. 1299, Road No. 36, Near Peddamma Temple, Jubilee Hills, Hyderabad - 500033",
        "room": "Main Skin Suite, 1st Floor, Consultation Room 3",
        "timings": "Mon – Sat • 11:00 AM – 5:00 PM",
        "fee": "₹700",
        "rating": "4.82",
        "reviews": "1,950+ Patients",
        "facilities": "Fractional CO2 Laser, Hair Follicle Microscope, Chemical Peel Center, Cryotherapy Unit",
        "phone": "+91 9944556677"
    },

    # 4. Neurology
    {
        "category": "Neurology",
        "name": "Dr. Ravi Teja",
        "specialization": "Consultant Neurologist & Stroke Specialist",
        "qualifications": "MBBS, MD, DM (Neurology - NIMS), Stroke Fellow",
        "hospital": "KIMS Hospitals (Krishna Institute of Medical Sciences)",
        "address": "1-8-31/1, Minister Road, Krishna Nagar Colony, Begumpet, Secunderabad - 500003",
        "room": "Bone & Neuro Pavilion, 4th Floor, OPD Room No. 405",
        "timings": "Mon – Sat • 9:30 AM – 1:00 PM",
        "fee": "₹850",
        "rating": "4.9",
        "reviews": "3,200+ Cases",
        "facilities": "Comprehensive Neuro ICU, 32-Channel Digital EEG, Video EMG Lab, 3T MRI & CT Angiography",
        "phone": "+91 9012345678"
    },
    {
        "category": "Neurology",
        "name": "Dr. Ananya Sen",
        "specialization": "Chief Neurologist & Movement Disorder Specialist",
        "qualifications": "MBBS, MD, DM (Neurology - PGIMER)",
        "hospital": "Narayana Health City, Gachibowli",
        "address": "Plot No. 3 & 4, Gachibowli Financial District, Nanakramguda, Hyderabad - 500032",
        "room": "Brain & Spine Tower, 2nd Floor, Room 214",
        "timings": "Mon – Fri • 10:00 AM – 3:00 PM",
        "fee": "₹900",
        "rating": "4.92",
        "reviews": "2,800+ Consultations",
        "facilities": "Advanced Sleep Study Polysomnography, Neuro-Rehab Center, Botox for Migraine Unit",
        "phone": "+91 9833445566"
    },

    # 5. Pediatrics
    {
        "category": "Pediatrics",
        "name": "Dr. Priya Mehta",
        "specialization": "Senior Consultant Pediatrician & Neonatologist",
        "qualifications": "MBBS, MD (Pediatrics), Fellowship in Neonatal Intensive Care",
        "hospital": "Rainbow Children’s Hospital, Kondapur",
        "address": "Survey No. 49, Road No. 2, Botanical Garden Road, Kondapur, Hyderabad - 500084",
        "room": "Child Health Wing, Ground Floor, OPD Room No. 12",
        "timings": "Mon – Sat • 10:00 AM – 2:00 PM",
        "fee": "₹700",
        "rating": "4.92",
        "reviews": "5,000+ Pediatric Visits",
        "facilities": "Level III PICU & NICU, Child Immunization Center, Pediatric Emergency, Child-Friendly Area",
        "phone": "+91 9012345678"
    },
    {
        "category": "Pediatrics",
        "name": "Dr. Rohan Gupta",
        "specialization": "Consultant Pediatrician & Child Development Specialist",
        "qualifications": "MBBS, DCH, DNB (Pediatrics)",
        "hospital": "Cloudnine Hospital, Hitec City",
        "address": "Plot 18, Siddiq Nagar, Gachibowli - Hitec City Main Rd, Hyderabad - 500081",
        "room": "Pediatric Floor, 2nd Level, Room 206",
        "timings": "Mon – Sat • 11:30 AM – 4:00 PM",
        "fee": "₹650",
        "rating": "4.88",
        "reviews": "3,100+ Visits",
        "facilities": "Well-Baby Clinic, Pediatric Allergy Testing, Emergency Pediatric Resuscitation Bay",
        "phone": "+91 9822114433"
    },

    # 6. Gastroenterology
    {
        "category": "Gastroenterology",
        "name": "Dr. K. Srinivas Rao",
        "specialization": "Chief Gastroenterologist & Hepatologist",
        "qualifications": "MBBS, MD, DM (Gastroenterology), FACG",
        "hospital": "AIG Hospitals (Asian Institute of Gastroenterology)",
        "address": "1-66/AIG/2 to 9, Mindspace Rd, Gachibowli, Hyderabad - 500032",
        "room": "Gastroenterology Tower, 2nd Floor, OPD Room No. 220",
        "timings": "Mon – Fri • 9:00 AM – 1:00 PM",
        "fee": "₹900",
        "rating": "4.88",
        "reviews": "3,800+ Consultations",
        "facilities": "World-class Endoscopy & Colonoscopy Suite, Liver FibroScan Center, 24x7 GI Bleed Emergency",
        "phone": "+91 9888776611"
    },
    {
        "category": "Gastroenterology",
        "name": "Dr. Shalini Joshi",
        "specialization": "Consultant Hepatologist & Therapeutic Endoscopist",
        "qualifications": "MBBS, MD (Medicine), DM (Gastroenterology)",
        "hospital": "Manipal Hospitals, Old Airport Road",
        "address": "98, HAL Old Airport Rd, Kodihalli, Bengaluru, Karnataka - 560017",
        "room": "Digestive Diseases Wing, 1st Floor, Room 104",
        "timings": "Mon – Sat • 10:00 AM – 2:30 PM",
        "fee": "₹800",
        "rating": "4.85",
        "reviews": "2,400+ Procedures",
        "facilities": "High-Definition Video Endoscopy, Capsule Endoscopy, Liver ICU, GI Motility Lab",
        "phone": "+91 9845012345"
    },

    # 7. Orthopedics
    {
        "category": "Orthopedics",
        "name": "Dr. Arvind Swamy",
        "specialization": "Senior Joint Replacement & Robotic Orthopedic Surgeon",
        "qualifications": "MBBS, MS (Orthopedics - AIIMS), M.Ch (Ortho - UK)",
        "hospital": "Continental Hospitals, Gachibowli",
        "address": "Plot No. 3, Road No. 2, IT & Financial District, Nanakramguda, Gachibowli, Hyderabad - 500032",
        "room": "Ortho & Trauma Pavilion, 3rd Floor, Room 308",
        "timings": "Mon – Sat • 10:00 AM – 3:00 PM",
        "fee": "₹850",
        "rating": "4.93",
        "reviews": "4,500+ Surgeries",
        "facilities": "Mako Robotic Knee Replacement, Computer-Assisted Navigation, Hydrotherapy Sports Physio Unit",
        "phone": "+91 9840123456"
    },
    {
        "category": "Orthopedics",
        "name": "Dr. Manoj Chawla",
        "specialization": "Consultant Spine Surgeon & Arthroscopy Specialist",
        "qualifications": "MBBS, MS (Ortho), Fellowship in Spine Surgery",
        "hospital": "KIMS Bone & Joint Pavilion, Begumpet",
        "address": "1-8-31/1, Minister Road, Begumpet, Secunderabad - 500003",
        "room": "Orthopedics Block, 1st Floor, Room 112",
        "timings": "Mon – Fri • 11:00 AM – 4:00 PM",
        "fee": "₹800",
        "rating": "4.87",
        "reviews": "2,700+ Spine Cases",
        "facilities": "Minimally Invasive Spine Surgery Suite, DEXA Bone Mineral Scan, In-House Orthotics Workshop",
        "phone": "+91 9814012345"
    },

    # 8. Pulmonology
    {
        "category": "Pulmonology",
        "name": "Dr. Rajesh Nair",
        "specialization": "Senior Pulmonologist & Interventional Bronchoscopist",
        "qualifications": "MBBS, MD (Pulmonary Medicine - CMC), FCCP",
        "hospital": "Apollo Chest Hospital, Hyderguda",
        "address": "3-5-836 to 839, Hyderguda Main Road, Old MLA Quarters, King Koti, Hyderabad - 500029",
        "room": "Chest Wing, 2nd Floor, Room 202",
        "timings": "Mon – Sat • 10:00 AM – 2:00 PM",
        "fee": "₹800",
        "rating": "4.89",
        "reviews": "3,100+ Respiratory Patients",
        "facilities": "Advanced Pulmonary Function Lab (PFT with DLCO), Rigid & Flexible Bronchoscopy Suite",
        "phone": "+91 9847012345"
    },
    {
        "category": "Pulmonology",
        "name": "Dr. Sunita Kulkarni",
        "specialization": "Consultant Chest Physician & Allergy Specialist",
        "qualifications": "MBBS, DNB (Respiratory Diseases), European Dip. Resp. Med",
        "hospital": "Yashoda Pulmonology Center, Somajiguda",
        "address": "Raj Bhavan Road, Somajiguda, Hyderabad - 500082",
        "room": "Tower 2, 3rd Floor, Room 315",
        "timings": "Mon – Sat • 11:00 AM – 3:30 PM",
        "fee": "₹750",
        "rating": "4.84",
        "reviews": "2,200+ Cases",
        "facilities": "Spirometry Testing Bay, Fractional Exhaled Nitric Oxide (FeNO), Inhalation Therapy Lounge",
        "phone": "+91 9820012345"
    },

    # 9. Endocrinology
    {
        "category": "Endocrinology",
        "name": "Dr. Alok Bansal",
        "specialization": "Chief Endocrinologist & Diabetologist",
        "qualifications": "MBBS, MD (Medicine), DM (Endocrinology - PGI)",
        "hospital": "Fortis C-DOC Center of Excellence for Diabetes",
        "address": "B-16, Chirag Enclave, Greater Kailash, New Delhi - 110048",
        "room": "Endocrine Pavilion, Ground Floor, Room 102",
        "timings": "Mon – Fri • 9:30 AM – 2:30 PM",
        "fee": "₹900",
        "rating": "4.94",
        "reviews": "4,200+ Patients",
        "facilities": "Continuous Glucose Monitoring (CGM) Lab, Diabetic Foot Care Unit, Comprehensive Hormone Assay",
        "phone": "+91 9810012345"
    },
    {
        "category": "Endocrinology",
        "name": "Dr. Ritu Saxena",
        "specialization": "Consultant Endocrinologist & Thyroid Specialist",
        "qualifications": "MBBS, MD, Fellowship in Diabetes (Royal Liverpool)",
        "hospital": "Medanta Endocrine Wing, Gurgaon",
        "address": "CH Bakhtawar Singh Rd, Medicity, Islampur Colony, Sector 38, Gurugram - 122001",
        "room": "OPD Block A, 2nd Floor, Room 228",
        "timings": "Mon – Sat • 10:00 AM – 3:00 PM",
        "fee": "₹850",
        "rating": "4.87",
        "reviews": "2,900+ Patients",
        "facilities": "Thyroid Fine Needle Biopsy Unit, Insulin Pump Clinic, Bone Density DEXA Scanner",
        "phone": "+91 9811012345"
    },

    # 10. Psychiatry
    {
        "category": "Psychiatry",
        "name": "Dr. Maya Krishnan",
        "specialization": "Senior Consultant Psychiatrist & Neurobehaviorist",
        "qualifications": "MBBS, MD (Psychiatry - NIMHANS), MRCPsych (UK)",
        "hospital": "Asha Hospital, Banjara Hills",
        "address": "Plot No. 298, Road No. 14, Banjara Hills, Hyderabad - 500034",
        "room": "Mind Wellness Wing, 1st Floor, Room 105",
        "timings": "Mon – Sat • 10:30 AM – 4:00 PM",
        "fee": "₹950",
        "rating": "4.96",
        "reviews": "3,400+ Patients",
        "facilities": "Repetitive Transcranial Magnetic Stimulation (rTMS), Clinical Biofeedback, Quiet Therapy Rooms",
        "phone": "+91 9841012345"
    },
    {
        "category": "Psychiatry",
        "name": "Dr. Vivek Oberoi",
        "specialization": "Consultant Psychiatrist & De-Addiction Specialist",
        "qualifications": "MBBS, DNB (Psychiatry), Dip. Psychological Medicine",
        "hospital": "NIMHANS Outpatient Pavilion, Bangalore",
        "address": "Hosur Road, Near Dairy Circle, Lakkasandra, Bengaluru - 560029",
        "room": "Main OPD Block, Ground Floor, Room 14",
        "timings": "Mon – Fri • 9:00 AM – 1:30 PM",
        "fee": "₹800",
        "rating": "4.89",
        "reviews": "2,600+ Patients",
        "facilities": "Comprehensive Cognitive Testing, Adult Neurodevelopmental Clinic, Sleep Therapy Suite",
        "phone": "+91 9818012345"
    },

    # 11. Gynecology
    {
        "category": "Gynecology",
        "name": "Dr. Sunita Reddy",
        "specialization": "Senior Consultant Obstetrician & Laparoscopic Surgeon",
        "qualifications": "MBBS, MS (OBG), MRCOG (London), FICOG",
        "hospital": "Fernandez Hospital, Hyderguda",
        "address": "3-6-282, Opp. Old MLA Quarters, Hyderguda, Hyderabad - 500029",
        "room": "Maternity & Gyne Block, 2nd Floor, Room 204",
        "timings": "Mon – Sat • 9:30 AM – 2:00 PM",
        "fee": "₹850",
        "rating": "4.95",
        "reviews": "5,800+ Deliveries & Surgeries",
        "facilities": "Advanced 4D Fetal Ultrasound, Fetal Medicine Unit, High-Risk Obstetrics ICU, LDR Suites",
        "phone": "+91 9849012345"
    },
    {
        "category": "Gynecology",
        "name": "Dr. Deepa Nambiar",
        "specialization": "Consultant Gynecologist & Laparoscopic Surgeon",
        "qualifications": "MBBS, DGO, DNB (OBG), Fellowship in Reproductive Med",
        "hospital": "Motherhood Hospital, Indiranagar",
        "address": "2240, 80 Feet Road, HAL 3rd Stage, Indiranagar, Bengaluru - 560038",
        "room": "Women’s Health Pavilion, 1st Floor, Room 107",
        "timings": "Mon – Sat • 10:30 AM – 3:30 PM",
        "fee": "₹800",
        "rating": "4.9",
        "reviews": "3,100+ Consultations",
        "facilities": "Minimally Invasive Gynecological Surgery Suite, Colposcopy Clinic, Antenatal Training Hall",
        "phone": "+91 9848012345"
    },

    # 12. ENT
    {
        "category": "ENT",
        "name": "Dr. Farhan Qureshi",
        "specialization": "Chief ENT, Cochlear Implant & Skull Base Surgeon",
        "qualifications": "MBBS, MS (ENT - Safdarjung), DNB (ENT)",
        "hospital": "KIMS ENT Center, Secunderabad",
        "address": "1-8-31/1, Minister Road, Begumpet, Secunderabad - 500003",
        "room": "ENT & Audiology Pavilion, 3rd Floor, Room 302",
        "timings": "Mon – Sat • 10:00 AM – 2:30 PM",
        "fee": "₹750",
        "rating": "4.88",
        "reviews": "3,400+ Surgeries",
        "facilities": "Soundproof Audiometry Booth, Videolaryngoscopy & Stroboscopy Suite, Microdebrider Sinus Unit",
        "phone": "+91 9821012345"
    },
    {
        "category": "ENT",
        "name": "Dr. Sneha Agarwal",
        "specialization": "Consultant ENT Surgeon & Voice Specialist",
        "qualifications": "MBBS, DLO, DNB (Otorhinolaryngology)",
        "hospital": "Apollo ENT & Hearing Center, Jubilee Hills",
        "address": "Road No. 72, Film Nagar, Jubilee Hills, Hyderabad - 500033",
        "room": "Surgical Block, 2nd Floor, Room 218",
        "timings": "Mon – Fri • 11:00 AM – 4:00 PM",
        "fee": "₹700",
        "rating": "4.84",
        "reviews": "2,100+ Consultations",
        "facilities": "Tympanometry & OAE Hearing Testing, Nasal Endoscopy, Coblation Adenoidectomy Unit",
        "phone": "+91 9819012345"
    },

    # 13. Oncology
    {
        "category": "Oncology",
        "name": "Dr. Sandeep Ghosh",
        "specialization": "Senior Medical Oncologist & Chemotherapy Specialist",
        "qualifications": "MBBS, MD (Medicine), DM (Medical Oncology - Tata Memorial)",
        "hospital": "Basavatarakam Indo-American Cancer Hospital & Research Institute",
        "address": "Road No. 10, Banjara Hills, Hyderabad, Telangana - 500034",
        "room": "Medical Oncology Tower, 2nd Floor, OPD Room 208",
        "timings": "Mon – Fri • 9:30 AM – 2:00 PM",
        "fee": "₹1,000",
        "rating": "4.97",
        "reviews": "4,900+ Cancer Patients",
        "facilities": "State-of-the-Art Daycare Chemotherapy Ward, PET-CT Scanner, Linear Accelerator Radiotherapy Unit",
        "phone": "+91 9831012345"
    },

    # 14. Ophthalmology
    {
        "category": "Ophthalmology",
        "name": "Dr. Poornima Iyer",
        "specialization": "Chief Cornea & Refractive Surgeon",
        "qualifications": "MBBS, MS (Ophthalmology), DNB, FICO (UK)",
        "hospital": "L V Prasad Eye Institute (LVPEI), Banjara Hills",
        "address": "Kallam Anji Reddy Campus, Road No. 2, Banjara Hills, Hyderabad - 500034",
        "room": "Cornea Clinic, 1st Floor, OPD Room 114",
        "timings": "Mon – Sat • 9:00 AM – 1:30 PM",
        "fee": "₹700",
        "rating": "4.92",
        "reviews": "4,100+ Eye Procedures",
        "facilities": "Femtosecond LASIK Laser, Ocular Surface Center, Digital Slit Lamp Photography, Corneal Topography",
        "phone": "+91 9842012345"
    },

    # 15. Nephrology
    {
        "category": "Nephrology",
        "name": "Dr. Harish Bhatt",
        "specialization": "Chief Nephrologist & Renal Transplant Physician",
        "qualifications": "MBBS, MD, DM (Nephrology - AIIMS), FASN (USA)",
        "hospital": "Care Kidney Institute, Nampally",
        "address": "Mukarram Jahi Road, Near Exhibition Grounds, Nampally, Hyderabad - 500001",
        "room": "Nephrology Block, 2nd Floor, OPD Room 210",
        "timings": "Mon – Sat • 10:00 AM – 3:00 PM",
        "fee": "₹900",
        "rating": "4.94",
        "reviews": "3,700+ Kidney Patients",
        "facilities": "30-Bed Hemodialysis Unit with CRRT, Kidney Biopsy Ultrasound Unit, Water Treatment RO Plant",
        "phone": "+91 9825012345"
    }
]


# ----------------------------------------------------------------------------
# CONSULTATIONS & APPOINTMENTS (PATIENT)
# ----------------------------------------------------------------------------

@app.route('/video_consultancy')
@login_required
def video_consultancy():
    user = User.query.get(session['user_id'])
    if user.is_doctor:
        flash("You are signed in as a Doctor. Video Consultant is a patient service. Please use your Doctor Dashboard.", "info")
        return redirect(url_for('doctor_dashboard'))
    return render_template('video_consultancy.html', doctors=VIDEO_DOCTORS)


@app.route('/offline_doctor')
@login_required
def offline_doctor():
    user = User.query.get(session['user_id'])
    if user.is_doctor:
        flash("You are signed in as a Doctor. Hospital OPD Visit is a patient service. Please use your Doctor Dashboard.", "info")
        return redirect(url_for('doctor_dashboard'))
    return render_template('offline_doctor.html', doctors=OFFLINE_DOCTORS)


@app.route('/online_treatment')
@login_required
def online_treatment():
    user = User.query.get(session['user_id'])
    if user.is_doctor:
        flash("You are signed in as a Doctor. Smart Health Cafe is a patient service. Please use your Doctor Dashboard.", "info")
        return redirect(url_for('doctor_dashboard'))
    return render_template('online_treatment.html')


@app.route('/book_appointment', methods=['POST'])
@login_required
def book_appointment():
    user = User.query.get(session['user_id'])

    doctor_name = request.form.get('doctor_name', 'Consultant Doctor')
    specialization = request.form.get('specialization', 'General')
    hospital = request.form.get('hospital', '')
    appointment_type = request.form.get('appointment_type', 'In-Person')
    appointment_date = request.form.get('appointment_date', '')
    time_slot = request.form.get('time_slot', '10:00 AM')
    phone = request.form.get('phone', '')
    patient_notes = request.form.get('patient_notes', '')

    appt = Appointment(
        user_id=user.id,
        doctor_name=doctor_name,
        specialization=specialization,
        hospital=hospital,
        appointment_type=appointment_type,
        appointment_date=appointment_date,
        time_slot=time_slot,
        doctor_phone=phone,
        patient_notes=patient_notes,
        status="Confirmed"
    )
    db.session.add(appt)
    db.session.commit()

    flash(f"Appointment booked with {doctor_name} for {appointment_date} ({time_slot})!", "success")
    return redirect(url_for('dashboard'))


# ----------------------------------------------------------------------------
# ADMINISTRATOR CONSOLE ROUTES (TRACKING & MANAGEMENT)
# ----------------------------------------------------------------------------

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    total_users = User.query.filter_by(role='patient').count()
    total_appointments = Appointment.query.count()
    total_prescriptions = PrescriptionRecord.query.count()
    video_count = Appointment.query.filter_by(appointment_type='Video').count()
    offline_count = Appointment.query.filter_by(appointment_type='In-Person').count()

    recent_users = User.query.filter_by(role='patient').order_by(User.last_login.desc()).limit(8).all()
    recent_appointments = Appointment.query.order_by(Appointment.created_at.desc()).limit(8).all()
    recent_prescriptions = PrescriptionRecord.query.order_by(PrescriptionRecord.created_at.desc()).limit(8).all()

    # Visualizations Data (Only High-Value Essential Clinical Info)
    # 1. Monthly Patient Usage Trend (Line Chart with EXACT Database Query Counts)
    now = datetime.utcnow()
    monthly_labels = []
    monthly_values = []
    for i in range(5, -1, -1):
        year = now.year
        month = now.month - i
        while month <= 0:
            month += 12
            year -= 1
        m_start = datetime(year, month, 1)
        if month == 12:
            m_end = datetime(year + 1, 1, 1)
        else:
            m_end = datetime(year, month + 1, 1)
        m_label = m_start.strftime('%b %Y')
        monthly_labels.append(m_label)
        
        # Exact SQL counts of patient activity from database tables
        m_appts = Appointment.query.filter(Appointment.created_at >= m_start, Appointment.created_at < m_end).count()
        m_rxs = PrescriptionRecord.query.filter(PrescriptionRecord.created_at >= m_start, PrescriptionRecord.created_at < m_end).count()
        exact_month_total = m_appts + m_rxs
        monthly_values.append(exact_month_total)

    # 2. Top Diagnosed Clinical Conditions (Doughnut Chart)
    all_prescriptions = PrescriptionRecord.query.all()
    disease_counts = Counter([r.diagnosed_disease for r in all_prescriptions if r.diagnosed_disease])
    if not disease_counts:
        disease_counts = Counter({
            "Viral Fever": 6,
            "Common Cold & Flu": 5,
            "Gastritis & Acid Reflux": 4,
            "Migraine & Tension Headache": 3,
            "Skin Allergy & Dermatitis": 2
        })
    top_diseases = disease_counts.most_common(6)
    disease_labels = [item[0] for item in top_diseases]
    disease_values = [item[1] for item in top_diseases]

    # 3. Consultation Modality (Telehealth Video vs In-Person Hospital Visits)
    consultation_labels = ["Telehealth Video Calls", "In-Person Hospital Visits"]
    consultation_values = [video_count, offline_count]
    if video_count == 0 and offline_count == 0:
        consultation_values = [8, 5]

    return render_template(
        'admin_dashboard.html',
        total_users=total_users,
        total_appointments=total_appointments,
        total_prescriptions=total_prescriptions,
        video_count=video_count,
        offline_count=offline_count,
        recent_users=recent_users,
        recent_appointments=recent_appointments,
        recent_prescriptions=recent_prescriptions,
        monthly_labels=json.dumps(monthly_labels),
        monthly_values=json.dumps(monthly_values),
        disease_labels=json.dumps(disease_labels),
        disease_values=json.dumps(disease_values),
        consultation_labels=json.dumps(consultation_labels),
        consultation_values=json.dumps(consultation_values)
    )


@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.filter_by(role='patient').order_by(User.last_login.desc()).all()
    return render_template('admin_users.html', users=users)


@app.route('/admin/appointments')
@admin_required
def admin_appointments():
    appointments = Appointment.query.order_by(Appointment.created_at.desc()).all()
    return render_template('admin_appointments.html', appointments=appointments)


@app.route('/admin/appointment/<int:appt_id>/status', methods=['POST'])
@admin_required
def admin_update_appointment(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    new_status = request.form.get('status', 'Confirmed')
    appt.status = new_status
    db.session.commit()
    flash(f"Appointment #{appt.id} for {appt.doctor_name} updated to {new_status}!", "success")
    return redirect(url_for('admin_appointments'))


@app.route('/admin/prescriptions')
@admin_required
def admin_prescriptions():
    prescriptions = PrescriptionRecord.query.order_by(PrescriptionRecord.created_at.desc()).all()
    return render_template('admin_prescriptions.html', prescriptions=prescriptions)


# ----------------------------------------------------------------------------
# APP STARTUP
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# DOCTOR & CLINICAL SPECIALIST PORTAL ROUTES
# ----------------------------------------------------------------------------

@app.route('/doctor/dashboard')
@doctor_required
def doctor_dashboard():
    doctor = User.query.get(session['user_id'])

    # Find all appointments appointed to this doctor
    doc_last_name = doctor.full_name.split()[-1] if doctor.full_name else ""
    appointments = Appointment.query.filter(
        (Appointment.doctor_name.ilike(f"%{doctor.full_name}%")) |
        (Appointment.doctor_name.ilike(f"%{doc_last_name}%"))
    ).order_by(Appointment.created_at.desc()).all()

    if not appointments:
        appointments = Appointment.query.order_by(Appointment.created_at.desc()).limit(10).all()

    # Exact metrics for doctor cards
    unique_patients = set([a.user_id for a in appointments])
    total_patients = len(unique_patients)
    total_appointments = len(appointments)
    video_count = len([a for a in appointments if a.appointment_type == 'Video'])
    opd_count = len([a for a in appointments if a.appointment_type == 'In-Person'])
    completed_count = len([a for a in appointments if a.status == 'Completed'])
    confirmed_count = len([a for a in appointments if a.status == 'Confirmed'])

    # Recent prescriptions for patients treated
    prescriptions = PrescriptionRecord.query.filter(
        PrescriptionRecord.user_id.in_(unique_patients) if unique_patients else False
    ).order_by(PrescriptionRecord.created_at.desc()).limit(8).all()

    return render_template(
        'doctor_dashboard.html',
        doctor=doctor,
        appointments=appointments,
        total_patients=total_patients,
        total_appointments=total_appointments,
        video_count=video_count,
        opd_count=opd_count,
        completed_count=completed_count,
        confirmed_count=confirmed_count,
        prescriptions=prescriptions
    )


@app.route('/doctor/appointment/<int:appt_id>/status', methods=['POST'])
@doctor_required
def doctor_update_appointment(appt_id):
    appt = Appointment.query.get_or_404(appt_id)
    new_status = request.form.get('status', 'Confirmed')
    appt.status = new_status
    db.session.commit()
    flash(f"Appointment #{appt.id} for {appt.doctor_name} updated to '{new_status}'!", "success")
    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/prescriptions')
@doctor_required
def doctor_prescriptions():
    return redirect(url_for('doctor_dashboard'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f" * Smart Healthcare Platform running on http://127.0.0.1:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
