from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class PromptConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    components = db.relationship('PromptComponent', backref='config', lazy=True, cascade='all, delete-orphan')
    experiments = db.relationship('PromptExperiment', backref='config', lazy=True)

class PromptComponent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    config_id = db.Column(db.Integer, db.ForeignKey('prompt_config.id'), nullable=False)
    component_type = db.Column(db.String(50), nullable=False)  # 'base', 'instructions', 'keywords', 'examples'
    content = db.Column(db.Text, nullable=False)
    is_enabled = db.Column(db.Boolean, default=True)
    order_index = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PromptExperiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    config_id = db.Column(db.Integer, db.ForeignKey('prompt_config.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    results = db.Column(db.JSON)  # Store accuracy, precision, etc.
    test_dataset = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)