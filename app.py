from flask import Flask, request, render_template, redirect, url_for, jsonify, session
import sqlite3
from io import BytesIO
import hashlib
from flask import Flask, render_template, url_for, redirect, Response,request, after_this_request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pygame
import imutils
import cv2
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for the session

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE,
              password_hash TEXT,
              email VARCHAR,
              phone_no INTEGER,
              R_address VARCHAR(255),
              gender VARCHAR,
              age INTEGER,
              dob DATE)''')

    conn.commit()
    conn.close()

init_db()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            
            # Hash the password for comparison
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, hashed_password))
            user = c.fetchone()
            conn.close()
            
            if user:
                # Store user details in session
                session['user'] = {
                    'id': user[0],
                    'username': user[1],
                    'email': user[3],
                    'phone_no': user[4],
                    'R_address': user[5],
                    'gender': user[6],
                    'age': user[7],
                    'dob': user[8]
                }
                # Redirect to home page
                return redirect('/index')
            else:
                # Invalid credentials, render login page with error message
                return render_template('login1.html', error='Invalid username or password')
        
        except Exception as e:
            # Handle any exceptions
            return render_template('error.html', message="An error occurred during login. Please try again later.")

    # If it's a GET request, render the login page
    return render_template('login1.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            email = request.form['email']
            phone = request.form['phone_no']
            R_address = request.form['R_address']
            gender = request.form['gender']
            age = request.form['age']
            dob = request.form['dob']

            # Hash the password
            password_hash = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password_hash, email, phone_no, R_address, gender, age, dob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (username, password_hash, email, phone, R_address, gender, age, dob))
            conn.commit()
            conn.close()

            return "Registration successful!", 200

        except Exception as e:
            print("Error during registration:", e)
            return "An error occurred during registration. Please try again later.", 500

    return render_template('register1.html')

@app.route('/index',methods=['GET','POST'])
def index():
    # Retrieve user details from session
    user = session.get('user', None)
    if user:
        @after_this_request
        def close_camera(response):
            release_camera()
            return response
        return render_template("index.html", user=user)
    else:
        # Redirect to login page if user is not logged in
        return redirect(url_for('login'))


# @app.route('/index',methods=['GET','POST'])
# def index():
#     @after_this_request
#     def close_camera(response):
#         release_camera()
#         return response   
#     return render_template('index.html')




cap = None

def init_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    

def release_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

#cap=cv2.VideoCapture(0)

def init_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)

def release_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

#cap=cv2.VideoCapture(0)
pygame.init()
pygame.mixer.init()
alarm_sound=pygame.mixer.music.load(os.getcwd()+"/sound.mp3")

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32) 

	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = os.getcwd()+"/face_detector/deploy.prototxt"
weightsPath =os.getcwd()+"/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model( os.getcwd()+"/mask_detector.model")

def mask_frames(cap):
    
    while True:
        flag, frame = cap.read()
        frame = imutils.resize(frame, width=640,height=480)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            if mask > withoutMask:
                pygame.mixer.music.play(-1)
                label="Mask"
                color = (0, 255, 0)
                
            else:
                
                pygame.mixer.music.stop()
                label= "No Mask"
                color=(0, 0, 255)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = np.asarray(buffer, dtype=np.uint8)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n')


def motion_detection(cap):
    #cap = cv2.VideoCapture(0)
    frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

    out = cv2.VideoWriter("output.avi", fourcc, 5.0, (440,330))

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    #print(frame1.shape)
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 900:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

            
        ret, buffer = cv2.imencode('.jpg', frame1)
        frame1 = buffer.tobytes()
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+ frame1 +b'\r\n\r\n')
        frame1 = frame2
        ret, frame2 = cap.read()



def name_detection(cap):
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.getcwd()+'/name/trainer/trainer.yml')
    cascadePath = os.getcwd()+"/name/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    #initiate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'amar']

    # Initialize and start realtime video capture
    #cam = cv2.VideoCapture(0)
    cap.set(3, 440) # set video widht
    cap.set(4, 330) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cap.get(3)
    minH = 0.1*cap.get(4)

    while True:

        ret, frame =cap.read()
        # img = cv2.flip(img, -1) # Flip vertically

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                pygame.mixer.music.stop()
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                pygame.mixer.music.play(-1)
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                
                
            
            cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            #cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        #cv2.imshow('camera',img) 
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+ frame +b'\r\n\r\n')
    
 

@app.route('/name', methods=['GET','POST'])
def name():
    return render_template('name.html')

@app.route('/name_video',methods=['GET','POST'])
def name_video():
    return Response(name_detection(cap), mimetype='multipart/x-mixed-replace; boundary=frame')



   
# @app.route('/',methods=['GET','POST'])
# def first():
#     return redirect(url_for('start'))


@app.route('/mask', methods=['GET'])
def mask():
    return render_template('mask.html')

@app.route('/mask_video', methods=['GET'])
def mask_video():
    return Response(mask_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/motion', methods=['GET'])
def motion():
    return render_template('motion.html')

@app.route('/motion_video',methods=['GET'])
def motion_video():
    return Response(motion_detection(cap),mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route("/start",methods=["GET", "POST"])
# def start():       
#     return render_template("start.html")

@app.before_request
def before_request():
    init_camera()

@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    # Redirect to the login page
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
