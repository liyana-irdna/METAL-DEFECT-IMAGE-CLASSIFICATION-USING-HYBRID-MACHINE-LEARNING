import os 
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify, flash
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
 
 
app = Flask(__name__)
 
 
app.secret_key = 'your secret key'
 
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'metal'
# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
 
mysql = MySQL(app)
 

@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        if role == 'user':
            cursor.execute('SELECT * FROM user WHERE username = %s AND password = %s', (username, password,))
            account = cursor.fetchone()

            if account:
                session['loggedin'] = True
                session['id'] = account['idUser']
                session['username'] = account['username']
                session['role'] = role
                msg = 'Logged in successfully!'
                return render_template('index.html', msg=msg, role=role)
            else:
                msg = 'Incorrect username/password or role!'

        elif role == 'admin':
            cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (username, password,))
            account = cursor.fetchone()

            if account:
                session['loggedin'] = True
                session['id'] = account['idAdmin']
                session['username'] = account['username']
                session['role'] = role
                msg = 'Logged in successfully!'
                return render_template('indexAdmin.html', msg=msg, role=role)
            else:
                msg = 'Incorrect username/password or role!'


    return render_template('login.html', msg=msg)

@app.route('/indexAdmin')
def indexAdmin():
    return render_template('indexAdmin.html')

@app.route('/indexUser')
def indexUser():
    return render_template('index.html')
 
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('idUser', None)
    session.pop('username', None)
    return redirect(url_for('login'))
 
@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form and 'phoneNum' in request.form :
        username = request.form['username']
        password = request.form['password']
        phoneNum = request.form['phoneNum']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE username = % s', (username, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO user VALUES (NULL, % s, % s,% s, % s)', (username, password, phoneNum, email, ))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg)

@app.route('/view_admin/<int:admin_id>')
def view_admin(admin_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Assuming 'admin' is the table name in your database
    cursor.execute('SELECT * FROM admin WHERE idAdmin = %s', (admin_id,))
    admin = cursor.fetchone()

    if admin:
        return render_template('view_admin.html', admin=admin)
    else:
        return render_template('admin_not_found.html')

@app.route('/edit_admin/<int:admin_id>', methods=['GET', 'POST'])
def edit_admin(admin_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if request.method == 'POST':
        new_username = request.form['new_username']
        new_password = request.form['new_password']
        new_phoneNum = request.form['new_phoneNum']
        new_email = request.form['new_email']

        # Update the admin's information in the database
        cursor.execute('UPDATE admin SET username=%s, password=%s, phoneNum=%s, email=%s WHERE idAdmin=%s',
                       (new_username, new_password, new_phoneNum, new_email, admin_id))
        mysql.connection.commit()
        
        
        # Flash a success message
        flash('Admin information updated successfully', 'success')

        return redirect(url_for('view_admin', admin_id=admin_id))

    # Fetch the current admin information
    cursor.execute('SELECT * FROM admin WHERE idAdmin = %s', (admin_id,))
    admin_info = cursor.fetchone()

    return render_template('edit_admin.html', admin_info=admin_info)

# User View and Edit fucntion
@app.route('/view_users/<int:user_id>')
def view_user(user_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Assuming 'admin' is the table name in your database
    cursor.execute('SELECT * FROM user WHERE idUser = %s', (user_id,))
    user = cursor.fetchone()

    if user:
        return render_template('view_users.html', user=user)
    else:
        return render_template('users_not_found.html')

@app.route('/edit_users/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if request.method == 'POST':
        new_username = request.form['new_username']
        new_password = request.form['new_password']
        new_phoneNum = request.form['new_phoneNum']
        new_email = request.form['new_email']

        # Update the admin's information in the database
        cursor.execute('UPDATE user SET username=%s, password=%s, phoneNo=%s, email=%s WHERE idUser=%s',
                       (new_username, new_password, new_phoneNum, new_email, user_id))
        mysql.connection.commit()
        
        
        # Flash a success message
        flash('User information updated successfully', 'success')

        return redirect(url_for('view_user', user_id=user_id))

    # Fetch the current admin information
    cursor.execute('SELECT * FROM user WHERE idUser = %s', (user_id,))
    user_info = cursor.fetchone()

    return render_template('edit_users.html', user_info=user_info)





# Load pre-trained models and encoder
cnn_model = load_model(r'C:\Users\irdina\Desktop\metalNEw\cnn_model.h5')
feature_extraction_model = load_model(r'C:\Users\irdina\Desktop\metalNEw\feature_extraction_model.h5')
svm_model = joblib.load(r'C:\Users\irdina\Desktop\metalNEw\svm_model.pkl')
label_encoder = joblib.load(r'C:\Users\irdina\Desktop\metalNEw\label_encoder.pkl')


from PIL import Image
# Load and preprocess an image for prediction
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((200, 200))
    image = np.array(image.convert('L')) / 255.0
    image = image.reshape(1, 200, 200, 1)
    return image

def evaluate_image(image_path, feature_extraction_model, svm_model, label_encoder):
    # Class labels
    class_names = {
        0: "Crazing",
        1: "Inclusion",
        2: "Patches",
        3: "Pitted",
        4: "Rolled",
        5: "Scratches"
    }
    image_data = preprocess_image(image_path)

    # Extract features using the CNN model
    cnn_features = feature_extraction_model.predict(image_data)
    cnn_features_flattened = cnn_features.flatten().reshape(1, -1)

    # Predict using the SVM model
    svm_prediction = svm_model.predict(cnn_features_flattened)

    # Decode the predicted class label
    predicted_class = label_encoder.inverse_transform(svm_prediction)[0]
    # Get the class name for the predicted class label
    predicted_class_name = class_names[predicted_class]

    # Get the confidence score
    confidence = svm_model.decision_function(cnn_features_flattened)[0]
        
    confidence_score = (1.0 / (1.0 + np.exp(-confidence))) * 100.0
    # Find the index of the class with the highest confidence
    highest_confidence_index = np.argmax(confidence_score)
    highest_confidence_score = confidence_score[highest_confidence_index]

    return predicted_class_name, highest_confidence_score

# Example usage
# image_path = r'C:\Users\irdina\Desktop\metal\NEU Metal Surface Defects Data\train\Crazing\Cr_1.bmp'
# predicted_class, accuracy_percentage = predict_class(image_path)
# print("Predicted Class:", predicted_class)
# print("Accuracy Percentage:", accuracy_percentage)





@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist('file')
    # highest_accuracy = -1  # Initialize to a negative value
    images = []

    for file in uploaded_files:
        if file is not None and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get the predicted class
            # Make predictions using both models
            # cnn_prediction = predict_cnn(file_path)
            predicted_class_name,highest_confidence_score = evaluate_image(file_path, feature_extraction_model, svm_model, label_encoder)
            # predicted_class, confidence_score = predict_class(file, feature_extraction_model, svm_model, label_encoder)
            # # Find the maximum value in the accuracy array
            # max_accuracy = np.max(np.array([[accuracy]]))

            image_info = {
                'filename': filename,
                'predicted_class': predicted_class_name,
                'confidence_score': highest_confidence_score
            }
            images.append(image_info)
        else:
            jsonify({'error': 'Invalid file extension'})
        
    return render_template('results.html', results=images)

@app.route('/uploadAdmin', methods=['POST'])
def uploadAdmin():
    uploaded_files = request.files.getlist('file')
    # highest_accuracy = -1  # Initialize to a negative value
    images = []

    for file in uploaded_files:
        if file is not None and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get the predicted class
            # Make predictions using both models
            # cnn_prediction = predict_cnn(file_path)
            predicted_class_name,highest_confidence_score = evaluate_image(file_path, feature_extraction_model, svm_model, label_encoder)

            # # Find the maximum value in the accuracy array
            # max_accuracy = np.max(np.array([[accuracy]]))

            image_info = {
                'filename': filename,
                'predicted_class': predicted_class_name,
                'confidence_score': highest_confidence_score
            }
            images.append(image_info)
        else:
            jsonify({'error': 'Invalid file extension'})
        
    return render_template('resultsAdmin.html', results=images)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)




