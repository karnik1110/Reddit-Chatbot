from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class Chat(db.Model):
     id = db.Column(db.Integer, primary_key=True) 
     data = db.Column(db.String(15000000))
     date = db.Column(db.DateTime(timezone=True), default=func.now())
     target_user_id = db.Column(db.Integer)
     source_user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True) 
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    user_name = db.Column(db.String(150))
    chats = db.relationship('Chat')
