from flask import Flask, render_template, request, redirect, url_for, session, flash

from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import generate_password_hash, check_password_hash

import pickle

import numpy as np

import os

app = Flask(__name__)

app.secret_key = 'super_secret_key_for_project' 

# Configure Local SQLite Database

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 1. Define Database Model for Users

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(50), unique=True, nullable=False)

    password = db.Column(db.String(256), nullable=False)

# Create database tables automatically

with app.app_context():

    db.create_all()

# Load the ML model safely

with open('autism_model.pkl', 'rb') as file:

    model = pickle.load(file)

# --- ROUTES ---

@app.route('/')

def index():

    if 'user_id' in session:

        return render_template('index.html', username=session['username'])

    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])

def signup():

    if request.method == 'POST':

        username = request.form['username']

        password = request.form['password']

        

        existing_user = User.query.filter_by(username=username).first()

        if existing_user:

            flash('Username already taken!', 'danger')

            return redirect(url_for('signup'))

            

        hashed_password = generate_password_hash(password, method='scrypt')

        new_user = User(username=username, password=hashed_password)

        db.session.add(new_user)

        db.session.commit()

        

        flash('Account created successfully! Please log in.', 'success')

        return redirect(url_for('login'))

        

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])

def login():

    if request.method == 'POST':

        username = request.form['username']

        password = request.form['password']

        

        user = User.query.filter_by(username=username).first()

        

        if user and check_password_hash(user.password, password):

            session['user_id'] = user.id

            session['username'] = user.username

            session.modified = True 

            return redirect(url_for('index'))

        else:

            flash('Invalid username or password!', 'danger')

            return redirect(url_for('login'))

            

    return render_template('login.html')

@app.route('/logout')

def logout():

    session.pop('user_id', None)

    session.pop('username', None)

    flash('Logged out successfully.', 'info')

    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])

def predict():

    if 'user_id' not in session:

        return redirect(url_for('login'))

        

    if request.method == 'POST':

        try:

            features = []

            for i in range(1, 11):

                features.append(int(request.form[f'q{i}']))

            

            features.append(int(request.form['age']))

            features.append(int(request.form['jaundice']))

            features.append(int(request.form['gender']))

            

            final_features = [np.array(features)]

            prediction = model.predict(final_features)

            

            try:

                probability = model.predict_proba(final_features)[0][1] * 100

                confidence_text = f"{probability:.2f}%"

            except:

                confidence_text = "N/A"

            

            if prediction[0] == 1:

                result = "High probability of Autism Spectrum Disorder (ASD) traits observed."

            else:

                result = "Low probability of Autism Spectrum Disorder (ASD) traits observed."

                

            return render_template('result.html', prediction_text=result, confidence=confidence_text)

        except Exception as e:

            return f"An error occurred: {str(e)}"

if __name__ == '__main__':

    app.run(debug=True)