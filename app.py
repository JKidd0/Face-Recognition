from flask import Flask, render_template, Response, request, session, redirect
from flaskext.mysql import MySQL
import cv2
import datetime
import os
import numpy as np
from threading import Thread
from flask_bcrypt import Bcrypt
import json
import time
import zmq
import concurrent.futures

global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
url = "http://192.168.8.103:81/stream"

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

mysql = MySQL()
app = Flask(__name__)
bcrypt = Bcrypt(app)

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'pyweb'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)

#app.secret_key = 'why would I tell you my secret key?'

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640 pixels
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480 pixels

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/modelo.yml')

start_time = time.time()
detected_person = ''

def detect_face(frame):
    global start_time, detected_person

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id, confidence = recognizer.predict(roi_gray)

        if confidence < 60:
            recognized_name = "Recognized"
            if recognized_name != detected_person:
                start_time = time.time()
                detected_person = recognized_name
        else:
            recognized_name = "Unknown"
            if recognized_name != detected_person:
                start_time = time.time()
                detected_person = recognized_name

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, recognized_name, (x, y-10), font, 0.9, (0, 255, 0) if recognized_name == "Recognized" else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) if recognized_name == "Recognized" else (0, 0, 255), 2)

    current_time = time.time()
    time_diff = current_time - start_time
    if time_diff >= 10.0 and detected_person == "Unknown":
        cv2.putText(frame, "INTRUSO", (10, 30), font, 1, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.putText(frame, "Tiempo transcurrido: {:.2f} segundos".format(time_diff), (10, 60), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return cv2.flip(cv2.flip(frame, 1), 1)  # Flip horizontally twice

def gen_frames():
    while True:
        success, frame = camera.read()
        if success:
            if capture:
                frame = detect_face(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/api/login', methods=['POST'])
def apiLogin():
    try:
        _email = request.form['email']
        _password = request.form['password']
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.callproc('sp_checkUserByEmail', (_email,))
        data = cursor.fetchall()

        if len(data) > 0:
            if bcrypt.check_password_hash(str(data[0][4]), _password):
                session['user'] = data[0][1]
                return redirect('/user/index')
            else:
                return render_template('index.html', error="Incorrect password")
        else:
            return render_template('index.html', error="Email not registered")

    except Exception as e:
        return render_template('index.html', error=str(e))
    finally:
        cursor.close()
        conn.close()

@app.route('/api/register', methods=['POST'])
def apiRegister():
    try:
        _name = request.form['name']
        _username = request.form['username']
        _email = request.form['email']
        _password = request.form['password']

        if _name and _username and _email and _password:
            conn = mysql.connect()
            cursor = conn.cursor()
            _hashed_password = bcrypt.generate_password_hash(_password).decode('utf-8')

            cursor.callproc('sp_createUser', (_name, _username, _email, _hashed_password))
            data = cursor.fetchall()

            if len(data) == 0:
                conn.commit()
                return render_template('index.html', error='Registered successfully!')
            else:
                return render_template('register.html', error='Email already exists in the database!')

        else:
            return redirect('/register')

    except Exception as e:
        return json.dumps({'error': str(e)})
    finally:
        cursor.close()
        conn.close()

@app.route('/user/index')
def userIndex():
    if session.get('user'):
        return render_template('/user/index.html')
    else:
        return render_template('index.html', error='You must log in')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/user/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Activar':
            global capture
            capture = 1
        elif request.form.get('stop') == 'Stop/Start':
            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640 pixels
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480 pixels
                switch = 1
    elif request.method == 'GET':
        return render_template('/user/index.html')
    return render_template('/user/index.html')

if __name__ == "__main__":
    app.run(debug=True)

camera.release()
cv2.destroyAllWindows()
