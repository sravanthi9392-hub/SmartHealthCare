"""
Database Models for Smart Healthcare Platform using Flask-SQLAlchemy
Supports Role-Based Users (Patient & Admin), Patient Profiles,
Prescription Records, and Appointments with Admin Tracking.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default="patient")  # 'patient' or 'admin'
    full_name = db.Column(db.String(100), default="Patient")
    phone = db.Column(db.String(25), default="")
    age = db.Column(db.Integer, default=28)
    gender = db.Column(db.String(20), default="Not specified")
    blood_group = db.Column(db.String(10), default="O+")
    address = db.Column(db.Text, default="")
    emergency_contact = db.Column(db.String(100), default="")
    medical_history = db.Column(db.Text, default="No known chronic allergies")
    specialization = db.Column(db.String(120), default="")
    hospital = db.Column(db.String(150), default="")
    qualifications = db.Column(db.String(150), default="")
    consultation_fee = db.Column(db.String(50), default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    prescriptions = db.relationship('PrescriptionRecord', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    appointments = db.relationship('Appointment', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == "admin"

    @property
    def is_doctor(self):
        return self.role == "doctor"


class PrescriptionRecord(db.Model):
    __tablename__ = 'prescriptions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_age = db.Column(db.Integer, nullable=False)
    patient_gender = db.Column(db.String(20), default="Not specified")
    patient_address = db.Column(db.String(255), default="")
    duration = db.Column(db.String(50), default="3 days")
    symptoms = db.Column(db.Text, nullable=False)
    diagnosed_disease = db.Column(db.String(120), nullable=False)
    confidence_score = db.Column(db.Float, default=95.0)
    specialist = db.Column(db.String(100), default="General Physician")
    severity = db.Column(db.String(50), default="Moderate")

    # Serialized JSON structures
    medicines_json = db.Column(db.Text, default="[]")
    dosage_json = db.Column(db.Text, default="[]")
    timings_json = db.Column(db.Text, default="[]")
    home_remedies_json = db.Column(db.Text, default="[]")
    precautions_json = db.Column(db.Text, default="[]")
    diet_plan_json = db.Column(db.Text, default="{}")

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def medicines(self):
        try:
            return json.loads(self.medicines_json)
        except:
            return []

    @property
    def dosage(self):
        try:
            return json.loads(self.dosage_json)
        except:
            return []

    @property
    def timings(self):
        try:
            return json.loads(self.timings_json)
        except:
            return []

    @property
    def home_remedies(self):
        try:
            return json.loads(self.home_remedies_json)
        except:
            return []

    @property
    def precautions(self):
        try:
            return json.loads(self.precautions_json)
        except:
            return []

    @property
    def diet_plan(self):
        try:
            return json.loads(self.diet_plan_json)
        except:
            return {}


class Appointment(db.Model):
    __tablename__ = 'appointments'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    doctor_name = db.Column(db.String(100), nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    hospital = db.Column(db.String(120), default="")
    appointment_type = db.Column(db.String(50), default="In-Person")  # 'Video' or 'In-Person'
    appointment_date = db.Column(db.String(50), nullable=False)
    time_slot = db.Column(db.String(50), nullable=False)
    doctor_phone = db.Column(db.String(50), default="")
    patient_notes = db.Column(db.Text, default="")
    status = db.Column(db.String(30), default="Confirmed")  # 'Confirmed', 'Completed', 'Cancelled'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
