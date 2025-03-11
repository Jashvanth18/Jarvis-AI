# Jarvis AI Voice Assistant with Groq AI Integration, Facial Recognition, and Mood Detection
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os
import random
import requests
import json
import time
import cv2
import numpy as np
import face_recognition
import dlib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import pickle
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
from deepface import DeepFace
import pyautogui
import ctypes

# Load environment variables from .env file
load_dotenv('API_KEY.env')
WEATHER_API_KEY = os.getenv('API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

class JarvisAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.voice_data = ''
        self.weather_api_key = WEATHER_API_KEY
        self.groq_api_key = GROQ_API_KEY
        self.email = EMAIL_ADDRESS
        self.email_password = EMAIL_PASSWORD
        
        # Check API keys
        if not self.weather_api_key:
            print("Warning: Weather API key not found in API_KEY.env file")
        else:
            print("Weather API key loaded successfully")
            
        if not self.groq_api_key:
            print("Warning: Groq API key not found in API_KEY.env file")
        else:
            print("Groq API key loaded successfully")
            
        if not self.email or not self.email_password:
            print("Warning: Email credentials not found in API_KEY.env file")
        else:
            print("Email credentials loaded successfully")
        
        # Configure voice
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id)  # Default male voice
        self.engine.setProperty('rate', 175)  # Speed of speech
        
        self.wake_word = "jarvis"
        self.wake_word_required = False  # Set to True if you want to enforce the wake word
        
        # Face recognition variables
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_detection_active = False
        self.last_face_detected_time = time.time()
        self.user_present = False
        self.stranger_detected = False
        self.face_recognition_model = "hog"  # Options: "hog" (CPU) or "cnn" (GPU, more accurate)
        
        # Load facial landmark predictor model
        self.predictor_68_point_model = "./shape_predictor_68_face_landmarks.dat"
        try:
            # Load the model from the current directory
            self.pose_predictor_68_point = dlib.shape_predictor(self.predictor_68_point_model)
            print("✅ Facial landmark model loaded successfully!")
        except FileNotFoundError:
            print("❌ Error: Model file not found.")
            print("👉 Please download the model file from:")
            print("👉 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("👉 Extract it and place it in the same directory as this script.")
        except RuntimeError as e:
            print(f"❌ Error: Failed to load the model. Reason: {str(e)}")
            print("👉 Ensure the model file is not corrupted or invalid.")
        except Exception as e:
            print(f"❌ Unexpected Error: {str(e)}")
        
        # User credentials
        self.credentials = {
            "youtube": {"username": "", "password": ""},
            "gmail": {"username": "", "password": ""},
            "netflix": {"username": "", "password": ""},
            "spotify": {"username": "", "password": ""}
        }
        
        # Autologin status
        self.auto_login_active = False
        
        # Load known faces if file exists
        self.load_known_faces()
        
        # Timers for auto-lock and shutdown
        self.auto_lock_timer = 120  # seconds (2 minutes)
        self.auto_shutdown_timer = 600  # seconds (10 minutes)
        
        # Start face detection thread
        self.stop_threads = False
        self.face_thread = threading.Thread(target=self.face_detection_loop)
        self.face_thread.daemon = True
        
        # Blink detection variables
        self.blink_threshold = 0.19  # Eye aspect ratio threshold
        self.blink_consec_frames = 3  # Number of consecutive frames to consider as a blink
        self.eye_frames_counter = 0
        self.total_blinks = 0
        self.last_blink_time = time.time()
        self.blink_detection_active = False
        
    def speak(self, text):
        """Convert text to speech"""
        print(f"Jarvis: {text}")
        self.engine.say(text)
        if self.engine._inLoop:
            self.engine.endLoop()  # Stop ongoing loop before speaking
        self.engine.runAndWait()
    
    def listen(self, ask=False):
        """Listen for user input through microphone"""
        if ask:
            self.speak(ask)
            
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source)
            
            try:
                self.voice_data = self.recognizer.recognize_google(audio)
                print(f"You said: {self.voice_data}")
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio. Please try again.")
                return ""  # Return empty string instead of crashing
            except sr.RequestError:
                print("Speech recognition service is unavailable.")
                return ""
                
        return self.voice_data.lower()
    
    def get_weather(self, location):
        """Get weather information using API"""
        if not self.weather_api_key:
            self.speak("I'm sorry, but I can't access the weather API without an API key.")
            return
            
        try:
            # Example using OpenWeatherMap API
            base_url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': self.weather_api_key,
                'units': 'metric'  # For Celsius
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                temperature = data['main']['temp']
                description = data['weather'][0]['description']
                self.speak(f"The weather in {location} is {description} with a temperature of {temperature} degrees Celsius.")
            else:
                self.speak(f"I couldn't get weather information. Error: {data['message']}")
        except Exception as e:
            self.speak(f"An error occurred while fetching weather data: {str(e)}")
    
    def ask_groq(self, question):
        """Send a question to Groq AI and get a response"""
        if not self.groq_api_key:
            self.speak("I'm sorry, but I can't access Groq AI without an API key.")
            return
            
        try:
            # Groq API endpoint
            url = "https://api.groq.com/openai/v1/chat/completions"
            
            # Headers with API key
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            # Request data
            data = {
                "model": "llama3-8b-8192",  # You can change the model as needed
                "messages": [
                    {"role": "system", "content": "You are Jarvis, an AI assistant. Provide concise, helpful answers."},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.7,
                "max_tokens": 400
            }
            
            # Send request to Groq API
            print("Asking Groq AI...")
            response = requests.post(url, headers=headers, json=data)
            
            # Check if request was successful
            if response.status_code == 200:
                response_json = response.json()
                ai_response = response_json["choices"][0]["message"]["content"]
                return ai_response
            else:
                print(f"Error from Groq API: {response.status_code}")
                print(response.text)
                return f"I'm sorry, but I encountered an error when trying to answer your question. Status code: {response.status_code}"
                
        except Exception as e:
            print(f"Exception when calling Groq API: {str(e)}")
            return f"I'm sorry, but I encountered an error: {str(e)}"
    
    def load_known_faces(self):
        """Load known faces from pickle file if it exists"""
        try:
            if os.path.exists("known_faces.pkl"):
                with open("known_faces.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
                    print(f"Loaded {len(self.known_face_encodings)} known faces")
            else:
                print("No known faces file found. Starting with empty known faces.")
        except Exception as e:
            print(f"Error loading known faces: {str(e)}")
    
    def save_known_faces(self):
        """Save known faces to pickle file"""
        try:
            with open("known_faces.pkl", "wb") as f:
                data = {
                    "encodings": self.known_face_encodings,
                    "names": self.known_face_names
                }
                pickle.dump(data, f)
                print(f"Saved {len(self.known_face_encodings)} known faces")
        except Exception as e:
            print(f"Error saving known faces: {str(e)}")
    
    def register_face(self, name):
        """Register a new face"""
        self.speak(f"I'll now register your face as {name}. Please look at the camera.")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.speak("Could not access the camera. Face registration failed.")
            return False
        
        # Capture 5 images for better accuracy
        face_encodings = []
        count = 0
        max_count = 5
        
        while count < max_count:
            ret, frame = cap.read()
            if not ret:
                self.speak("Failed to capture image. Face registration failed.")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            # Convert to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_frame, model=self.face_recognition_model)
            
            if face_locations:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                face_encodings.append(face_encoding)
                count += 1
                self.speak(f"Captured image {count} of {max_count}")
                time.sleep(1)
            else:
                self.speak("No face detected. Please position yourself in front of the camera.")
                time.sleep(2)
        
        cap.release()
        cv2.destroyAllWindows()
        
        if face_encodings:
            # Average the encodings for better accuracy
            average_encoding = np.mean(face_encodings, axis=0)
            self.known_face_encodings.append(average_encoding)
            self.known_face_names.append(name)
            self.save_known_faces()
            self.speak(f"Face registration successful. I'll now recognize you as {name}.")
            return True
        else:
            self.speak("Face registration failed. Please try again.")
            return False
    
    def detect_face(self, frame):
        """Detect and recognize faces in a frame"""
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model=self.face_recognition_model)
        
        if not face_locations:
            return None, None, None
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Check each face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Try to match with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            # Get best match
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    return name, face_location, rgb_frame
            
            # If no match, it's an unknown face
            return "Unknown", face_location, rgb_frame
        
        return None, None, None
    
    def detect_facial_landmarks(self, frame):
        """Detect facial landmarks using dlib"""
        try:
            # Make sure the predictor is loaded
            if not hasattr(self, 'pose_predictor_68_point'):
                print("❌ Facial landmark predictor not loaded. Cannot detect landmarks.")
                return []
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using dlib
            faces = dlib.get_frontal_face_detector()(gray)
            
            landmarks_list = []
            
            for face in faces:
                # Get facial landmarks
                landmarks = self.pose_predictor_68_point(gray, face)
                
                # Convert to numpy array
                landmarks_points = []
                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    landmarks_points.append((x, y))
                
                landmarks_list.append(landmarks_points)
            
            return landmarks_list
        except Exception as e:
            print(f"❌ Error detecting facial landmarks: {str(e)}")
            return []
    
    def get_eye_aspect_ratio(self, eye_points):
        """Calculate eye aspect ratio"""
        # Compute the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def detect_blink(self, frame):
        """Detect eye blinks"""
        landmarks_list = self.detect_facial_landmarks(frame)
        
        if not landmarks_list:
            return False, 0
        
        for landmarks in landmarks_list:
            # Get left eye landmarks (points 36-41 in the 68 point model)
            left_eye = landmarks[36:42]
            
            # Get right eye landmarks (points 42-47 in the 68 point model)
            right_eye = landmarks[42:48]
            
            # Calculate eye aspect ratios
            left_ear = self.get_eye_aspect_ratio(left_eye)
            right_ear = self.get_eye_aspect_ratio(right_eye)
            
            # Average the eye aspect ratio for both eyes
            ear = (left_ear + right_ear) / 2.0
            
            # Check if below the blink threshold
            if ear < self.blink_threshold:
                self.eye_frames_counter += 1
            else:
                # If the eyes were closed for a sufficient number of frames, count as a blink
                if self.eye_frames_counter >= self.blink_consec_frames:
                    self.total_blinks += 1
                    current_time = time.time()
                    time_since_last_blink = current_time - self.last_blink_time
                    self.last_blink_time = current_time
                    
                    # Reset counter
                    self.eye_frames_counter = 0
                    
                    return True, time_since_last_blink
                
                # Reset counter
                self.eye_frames_counter = 0
        
        return False, 0
    
    def detect_emotion(self, frame):
        """Detect emotion in a face"""
        try:
            # Use DeepFace to analyze emotion
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            # Get the dominant emotion
            emotion = analysis['dominant_emotion']
            return emotion
        except Exception as e:
            print(f"Error detecting emotion: {str(e)}")
            return None
    
    def send_email_with_image(self, image, subject="Jarvis Security Alert"):
        """Send an email with an image attachment"""
        if not self.email or not self.email_password:
            print("Email credentials not set. Can't send email.")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = self.email
            msg['Subject'] = subject
            
            # Attach text
            text = MIMEText(f"Jarvis detected an intruder at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
            msg.attach(text)
            
            # Convert image to JPEG
            _, img_encoded = cv2.imencode('.jpg', image)
            
            # Attach image
            image_attachment = MIMEImage(img_encoded.tobytes())
            image_attachment.add_header('Content-Disposition', 'attachment', filename='intruder.jpg')
            msg.attach(image_attachment)
            
            # Connect to SMTP server and send
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email, self.email_password)
                server.send_message(msg)
            
            print("Email sent successfully")
            return True
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False
    
    def lock_pc(self):
        """Lock the PC"""
        try:
            ctypes.windll.user32.LockWorkStation()
            print("PC locked")
            return True
        except Exception as e:
            print(f"Error locking PC: {str(e)}")
            return False
    
    def shutdown_pc(self):
        """Shutdown the PC"""
        try:
            os.system("shutdown /s /t 60")
            self.speak("PC will shut down in 1 minute. Say 'cancel shutdown' to abort.")
            return True
        except Exception as e:
            print(f"Error shutting down PC: {str(e)}")
            return False
    
    def restart_pc(self):
        """Restart the PC"""
        try:
            os.system("shutdown /r /t 60")
            self.speak("PC will restart in 1 minute. Say 'cancel restart' to abort.")
            return True
        except Exception as e:
            print(f"Error restarting PC: {str(e)}")
            return False
    
    def cancel_shutdown(self):
        """Cancel scheduled shutdown or restart"""
        try:
            os.system("shutdown /a")
            self.speak("Shutdown cancelled.")
            return True
        except Exception as e:
            print(f"Error cancelling shutdown: {str(e)}")
            return False
    
    def auto_login(self, service):
        """Auto login to a service using Selenium"""
        if not self.credentials[service]["username"] or not self.credentials[service]["password"]:
            self.speak(f"I don't have your {service} credentials. Please set them up first.")
            return False
        
        try:
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--start-maximized")
            
            # Initialize Chrome driver
            driver = webdriver.Chrome(options=chrome_options)
            
            if service == "youtube":
                driver.get("https://accounts.google.com/signin")
                
                # Enter email
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "identifier"))
                ).send_keys(self.credentials[service]["username"])
                
                # Click next
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Next')]"))
                ).click()
                
                # Enter password
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "password"))
                ).send_keys(self.credentials[service]["password"])
                
                # Click next/sign in
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Next')]"))
                ).click()
                
                # Navigate to YouTube
                WebDriverWait(driver, 10).until(
                    EC.url_contains("myaccount.google.com")
                )
                driver.get("https://www.youtube.com")
                
                self.speak("Logged in to YouTube successfully")
                return True
                
            elif service == "gmail":
                driver.get("https://accounts.google.com/signin")
                
                # Similar process as YouTube
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "identifier"))
                ).send_keys(self.credentials[service]["username"])
                
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Next')]"))
                ).click()
                
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "password"))
                ).send_keys(self.credentials[service]["password"])
                
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Next')]"))
                ).click()
                
                # Navigate to Gmail
                WebDriverWait(driver, 10).until(
                    EC.url_contains("myaccount.google.com")
                )
                driver.get("https://mail.google.com")
                
                self.speak("Logged in to Gmail successfully")
                return True
                
            elif service == "netflix":
                driver.get("https://www.netflix.com/login")
                
                # Enter email
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "userLoginId"))
                ).send_keys(self.credentials[service]["username"])
                
                # Enter password
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "password"))
                ).send_keys(self.credentials[service]["password"])
                
                # Click sign in
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Sign In')]"))
                ).click()
                
                self.speak("Logged in to Netflix successfully")
                return True
                
            elif service == "spotify":
                driver.get("https://accounts.spotify.com/en/login")
                
                # Enter username
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "login-username"))
                ).send_keys(self.credentials[service]["username"])
                
                # Enter password
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "login-password"))
                ).send_keys(self.credentials[service]["password"])
                
                # Click login button
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "login-button"))
                ).click()
                
                self.speak("Logged in to Spotify successfully")
                return True
            
            else:
                self.speak(f"I don't know how to log in to {service}")
                driver.quit()
                return False
                
        except Exception as e:
            print(f"Error logging in to {service}: {str(e)}")
            self.speak(f"I encountered an error logging in to {service}")
            return False
    
    def play_music_based_on_mood(self, mood):
        """Play music based on detected mood"""
        try:
            # Define playlists for different moods
            playlists = {
                "sad": "https://www.youtube.com/watch?v=OW_S_t5fxZA&list=RDQMyHJrpLvmock&start_radio=1",  # Motivational playlist.
                "angry": "https://www.youtube.com/watch?v=SdcAN3dobz4&list=RDQM5Auu_J59opM&start_radio=1",  # Calming playlist.
                "happy": "https://youtube.com/playlist?list=PLu-0nL-lZjTvqA_BMlbxYF-Z_o-tiX1c6&si=JZlOfoQMBAR6OyO5",  # Happy playlist.
                "neutral": "https://www.youtube.com/watch?v=5wRWniH7rt8&list=RDcIZhlFIyJ_Y&start_radio=1",  # Lofi playlist.
                "fear": "https://www.youtube.com/watch?v=SdcAN3dobz4&list=RDQM5Auu_J59opM&start_radio=1",  # Calming playlist.
                "disgust": "https://www.youtube.com/watch?v=OW_S_t5fxZA&list=RDQMyHJrpLvmock&start_radio=1",  # Uplifting playlist.
                "surprise": "https://www.youtube.com/watch?v=5wRWniH7rt8&list=RDcIZhlFIyJ_Y&start_radio=1"  # Lofi playlist.
            }
            
            # Default to neutral if mood not found
            url = playlists.get(mood.lower(), playlists["neutral"])
            
            # Open in browser
            webbrowser.open(url)
            
            if mood.lower() == "sad":
                self.speak("You seem a bit down. Playing some motivational music to lift your spirits. Come on boss! You got this!!")
            elif mood.lower() == "happy":
                self.speak("You look happy today! Playing some upbeat music to match your mood.")
            elif mood.lower() in ["neutral", "surprise"]:
                self.speak("You look chill today! Playing some lo-fi music to keep the vibe going.")
            else:
                self.speak(f"Playing music based on your {mood} mood.")
                
            return True
        except Exception as e:
            print(f"Error playing music: {str(e)}")
            return False
    
    def check_fatigue(self, blink_rate, frame=None):
        """Check for signs of fatigue based on blink rate and other factors"""
        # Normal blink rate is 15-20 blinks per minute
        # Lower rate might indicate fatigue
        if blink_rate < 10:  # Less than 10 blinks per minute
            self.speak("I've noticed your blink rate is low. You might be experiencing eye strain or fatigue.")
            self.speak("Consider taking a short break from the screen.")
            return True
        return False
    
    def measure_head_pose(self, landmarks):
        """Measure head pose using facial landmarks"""
        if not landmarks or len(landmarks) == 0:
            return None, None, None
        
        # Get specific landmark points
        # Nose tip (point 30)
        nose = landmarks[0][30]
        
        # Chin (point 8)
        chin = landmarks[0][8]
        
        # Left eye left corner (point 36)
        left_eye = landmarks[0][36]
        
        # Right eye right corner (point 45)
        right_eye = landmarks[0][45]
        
        # Left mouth corner (point 48)
        left_mouth = landmarks[0][48]
        
        # Right mouth corner (point 54)
        right_mouth = landmarks[0][54]
        
        # Calculate approximate pitch, yaw, roll
        # This is a simplified approximation
        
        # Yaw (horizontal head rotation)
        # Compare horizontal distances from nose to left/right eye
        left_dist = abs(nose[0] - left_eye[0])
        right_dist = abs(nose[0] - right_eye[0])
        yaw = (left_dist - right_dist) / (left_dist + right_dist) * 100
        
        # Pitch (vertical head tilt)
        # Compare vertical position of nose relative to eyes and mouth
        eye_level = (left_eye[1] + right_eye[1]) / 2
        mouth_level = (left_mouth[1] + right_mouth[1]) / 2
        pitch = (nose[1] - eye_level) / (mouth_level - eye_level) * 100 - 50
        
        # Roll (tilting head side to side)
        # Compare vertical positions of eyes
        roll = (left_eye[1] - right_eye[1]) / (left_eye[0] - right_eye[0]) * 100 if left_eye[0] != right_eye[0] else 0
        
        return pitch, yaw, roll
    
    def face_detection_loop(self):
        """Main loop for face detection"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.speak("Could not access the camera. Face detection disabled.")
            return
        
        # Variables for face detection timing
        last_unknown_detection = time.time()
        last_mood_check = time.time()
        last_blink_check = time.time()
        last_fatigue_check = time.time()
        blink_count = 0
        blink_start_time = time.time()
        
        try:
            while not self.stop_threads:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image from camera")
                    time.sleep(1)
                    continue
                
                # Detect face
                name, face_location, rgb_frame = self.detect_face(frame)
                current_time = time.time()
                
                # Handle face detection results
                if name:
                    if name != "Unknown":
                        # Known user detected
                        if not self.user_present:
                            self.speak(f"Welcome back, {name}!")
                            self.user_present = True
                            self.stranger_detected = False
                        
                        # Reset timers when user is present
                        self.last_face_detected_time = current_time
                        
                        # Blink detection (every 3 seconds)
                        if self.blink_detection_active and current_time - last_blink_check > 3:
                            blinked, time_since_last_blink = self.detect_blink(frame)
                            if blinked:
                                blink_count += 1
                                
                            # If we've been monitoring for a minute, check blink rate
                            if current_time - blink_start_time > 60:
                                # Check fatigue every minute
                                if current_time - last_fatigue_check > 60:
                                    self.check_fatigue(blink_count)
                                    last_fatigue_check = current_time
                                    blink_count = 0
                                    blink_start_time = current_time
                                    
                            last_blink_check = current_time
                        
                        # Mood detection (every 5 minutes)
                        if current_time - last_mood_check > 300:  # 300 seconds = 5 minutes
                            mood = self.detect_emotion(frame)
                            if mood:
                                print(f"Detected mood: {mood}")
                                # Only offer music for certain moods
                                if mood.lower() in ["sad", "angry"]:
                                    self.speak(f"I notice you seem {mood}. Would you like me to play some music to help?")
                                    
                            last_mood_check = current_time
                        
                        # Get head pose to detect drowsiness
                        landmarks_list = self.detect_facial_landmarks(frame)
                        if landmarks_list:
                            pitch, yaw, roll = self.measure_head_pose(landmarks_list)
                            # Check if head is drooping (indicating drowsiness)
                            if pitch and pitch > 30:  # Head tilted down significantly
                                if current_time - last_fatigue_check > 60:  # Check once per minute
                                    self.speak("Your head position suggests you might be drowsy. Consider taking a break.")
                                    last_fatigue_check = current_time
            else:
                # ✅ Unknown person detected (Intruder)
                if name == "Unknown":
                    # ✅ Check if the intruder has not been detected before
                    if not self.stranger_detected:
                        print("⚠️ Intruder detected!")
                        self.speak("Intruder alert! Taking a picture and sending security alert.")
                        
                        # ✅ Send email with the intruder's image
                        self.send_email_with_image(frame)
                        
                        # ✅ Lock the PC once (not repeatedly)
                        if self.auto_lock_timer > 0:
                            self.lock_pc()
                            print("🔒 PC Locked due to intruder.")
                        
                        # ✅ Set the flag that the intruder was detected
                        self.stranger_detected = True
                        self.user_present = False  # 🚨 Reset owner presence
                        last_unknown_detection = current_time
                    
                    # ✅ If the intruder stays, do NOTHING (silent mode)
                    else:
                        print("⚠️ Intruder still present... Staying silent.")

                # ✅ If the intruder leaves or no face is detected
                else:
                    # ✅ Reset everything if the intruder leaves
                    if self.stranger_detected:
                        print("✅ Intruder has left. Resetting detection.")
                        self.speak("Intruder has left. Everything is back to normal.")
                        self.stranger_detected = False

                    # ✅ If no face is detected, lock the PC after a timeout
                    time_without_face = current_time - self.last_face_detected_time
                    if self.user_present and self.auto_lock_timer > 0 and time_without_face > self.auto_lock_timer:
                        print(f"No face detected for {time_without_face:.1f} seconds. Locking PC.")
                        self.lock_pc()
                        self.user_present = False
                    
                    # ✅ Handle auto-shutdown if no face is detected for long
                    if self.user_present and self.auto_shutdown_timer > 0 and time_without_face > self.auto_shutdown_timer:
                        print(f"No face detected for {time_without_face:.1f} seconds. Initiating shutdown.")
                        self.shutdown_pc()
                        self.user_present = False

                # ✅ Prevent CPU over-usage (avoid looping continuously)
                time.sleep(0.1)

                
        except Exception as e:
            print(f"Error in face detection loop: {str(e)}")
        finally:
            cap.release()
            
    def start_face_detection(self):
        """Start the face detection thread"""
        if not self.face_detection_active:
            self.stop_threads = False
            self.face_thread = threading.Thread(target=self.face_detection_loop)
            self.face_thread.daemon = True
            self.face_thread.start()
            self.face_detection_active = True
            self.speak("Face detection activated")
    
    def stop_face_detection(self):
        """Stop the face detection thread"""
        if self.face_detection_active:
            self.stop_threads = True
            if self.face_thread.is_alive():
                self.face_thread.join(timeout=2)
            self.face_detection_active = False
            self.speak("Face detection deactivated")
    
    def set_credentials(self, service, username, password):
        """Set login credentials for a service"""
        if service.lower() in self.credentials:
            self.credentials[service.lower()] = {
                "username": username,
                "password": password
            }
            self.speak(f"Credentials for {service} have been saved")
            return True
        else:
            self.speak(f"I don't support the service {service} yet")
            return False
    
    def toggle_auto_login(self):
        """Toggle auto-login feature"""
        self.auto_login_active = not self.auto_login_active
        if self.auto_login_active:
            self.speak("Auto login activated. I'll log you in automatically when requested.")
        else:
            self.speak("Auto login deactivated.")
    
    def toggle_blink_detection(self):
        """Toggle blink detection feature"""
        self.blink_detection_active = not self.blink_detection_active
        if self.blink_detection_active:
            self.speak("Blink detection activated. I'll monitor your blink rate for signs of fatigue.")
        else:
            self.speak("Blink detection deactivated.")
    
    def set_auto_lock_timer(self, seconds):
        """Set the auto-lock timer"""
        try:
            seconds = int(seconds)
            if seconds < 0:
                self.speak("Timer must be a positive number of seconds.")
                return False
                
            self.auto_lock_timer = seconds
            if seconds == 0:
                self.speak("Auto-lock disabled.")
            else:
                self.speak(f"Auto-lock timer set to {seconds} seconds.")
            return True
        except ValueError:
            self.speak("Please provide a valid number of seconds.")
            return False
    
    def set_auto_shutdown_timer(self, seconds):
        """Set the auto-shutdown timer"""
        try:
            seconds = int(seconds)
            if seconds < 0:
                self.speak("Timer must be a positive number of seconds.")
                return False
                
            self.auto_shutdown_timer = seconds
            if seconds == 0:
                self.speak("Auto-shutdown disabled.")
            else:
                self.speak(f"Auto-shutdown timer set to {seconds} seconds.")
            return True
        except ValueError:
            self.speak("Please provide a valid number of seconds.")
            return False
    
    def run(self):
        """Main function to run the assistant"""
        self.speak("Jarvis AI Assistant is now online. How may I assist you today?")
        
        # Start face detection if model is available
        if hasattr(self, 'pose_predictor_68_point'):
            self.start_face_detection()
        
        while True:
            # Listen for voice commands
            voice_data = self.listen()
            
            # Convert to lowercase for easier matching
            if voice_data:
                voice_data = voice_data.lower()
            else:
                continue
            
            # Check for wake word if required
            if self.wake_word_required and not voice_data.startswith(self.wake_word):
                continue
                
            # Remove wake word from voice data if present
            if voice_data.startswith(self.wake_word):
                voice_data = voice_data.replace(self.wake_word, "", 1).strip()
            
            # Process commands
            
            # Exit commands
            if any(word in voice_data for word in ["exit", "quit", "goodbye", "bye"]):
                self.speak("Goodbye! Jarvis AI Assistant is shutting down.")
                self.stop_face_detection()
                break
            
            # Time
            elif "what time" in voice_data or "current time" in voice_data:
                current_time = datetime.datetime.now().strftime("%I:%M %p")
                self.speak(f"The current time is {current_time}")
            
            # Date
            elif "what date" in voice_data or "what day" in voice_data or "current date" in voice_data:
                current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
                self.speak(f"Today is {current_date}")
            
            # Weather
            elif "weather" in voice_data:
                # Extract location
                if "in" in voice_data:
                    location = voice_data.split("in")[1].strip()
                else:
                    location = "New York"  # Default location
                self.get_weather(location)
            
            # Web search
            elif "search for" in voice_data or "google" in voice_data:
                search_term = voice_data.replace("search for", "").replace("google", "").strip()
                self.speak(f"Searching the web for {search_term}")
                webbrowser.open(f"https://www.google.com/search?q={search_term}")
            
            # YouTube
            elif "youtube" in voice_data:
                if "open youtube" in voice_data:
                    self.speak("Opening YouTube")
                    webbrowser.open("https://www.youtube.com")
                elif "search youtube" in voice_data or "youtube search" in voice_data:
                    search_term = voice_data.replace("search youtube", "").replace("youtube search", "").strip()
                    self.speak(f"Searching YouTube for {search_term}")
                    webbrowser.open(f"https://www.youtube.com/results?search_query={search_term}")
            
            # AI assistance
            elif any(word in voice_data for word in ["ask groq", "ask ai", "use ai"]):
                question = voice_data.replace("ask groq", "").replace("ask ai", "").replace("use ai", "").strip()
                self.speak("Let me think about that...")
                response = self.ask_groq(question)
                self.speak(response)
            
            # Face registration
            elif "register my face" in voice_data or "remember my face" in voice_data:
                self.speak("I'll register your face. What's your name?")
                name = self.listen("Please say your name")
                if name:
                    self.register_face(name)
            
            # Start/stop face detection
            elif "start face detection" in voice_data or "enable face detection" in voice_data:
                self.start_face_detection()
            elif "stop face detection" in voice_data or "disable face detection" in voice_data:
                self.stop_face_detection()
            
            # Set credentials
            elif "set credentials" in voice_data or "save login" in voice_data:
                self.speak("Which service do you want to set credentials for?")
                service = self.listen("Please say the service name")
                if service and service.lower() in self.credentials:
                    self.speak(f"Please say your {service} username")
                    username = self.listen("Username")
                    self.speak(f"Please say your {service} password")
                    password = self.listen("Password")
                    self.set_credentials(service, username, password)
                else:
                    self.speak(f"I don't support {service} yet. I currently support YouTube, Gmail, Netflix, and Spotify.")
            
            # Auto login
            elif "login to" in voice_data or "log in to" in voice_data:
                service = None
                for s in self.credentials.keys():
                    if s in voice_data:
                        service = s
                        break
                
                if service:
                    self.speak(f"Logging in to {service}")
                    self.auto_login(service)
                else:
                    self.speak("I couldn't determine which service you want to log in to.")
            
            # Toggle auto login
            elif "toggle auto login" in voice_data:
                self.toggle_auto_login()
            
            # Toggle blink detection
            elif "toggle blink detection" in voice_data:
                self.toggle_blink_detection()
            
            # Set auto-lock timer
            elif "set auto lock" in voice_data:
                try:
                    # Extract seconds
                    words = voice_data.split()
                    for i, word in enumerate(words):
                        if word.isdigit():
                            seconds = int(word)
                            self.set_auto_lock_timer(seconds)
                            break
                    else:
                        self.speak("Please specify a time in seconds for the auto-lock timer.")
                except Exception as e:
                    self.speak(f"Error setting auto-lock timer: {str(e)}")
            
            # Set auto-shutdown timer
            elif "set auto shutdown" in voice_data:
                try:
                    # Extract seconds
                    words = voice_data.split()
                    for i, word in enumerate(words):
                        if word.isdigit():
                            seconds = int(word)
                            self.set_auto_shutdown_timer(seconds)
                            break
                    else:
                        self.speak("Please specify a time in seconds for the auto-shutdown timer.")
                except Exception as e:
                    self.speak(f"Error setting auto-shutdown timer: {str(e)}")
            
            # PC control commands
            elif "lock pc" in voice_data or "lock computer" in voice_data:
                self.speak("Locking your computer")
                self.lock_pc()
            elif "shutdown pc" in voice_data or "shutdown computer" in voice_data:
                self.speak("Initiating shutdown sequence")
                self.shutdown_pc()
            elif "restart pc" in voice_data or "restart computer" in voice_data:
                self.speak("Initiating restart sequence")
                self.restart_pc()
            elif "cancel shutdown" in voice_data or "cancel restart" in voice_data:
                self.cancel_shutdown()
            
            # Help command
            elif "help" in voice_data or "what can you do" in voice_data:
                self.speak("Here are some things I can do:")
                self.speak("Tell the time and date")
                self.speak("Check the weather")
                self.speak("Search the web or YouTube")
                self.speak("Answer questions using AI")
                self.speak("Recognize faces and detect emotions")
                self.speak("Monitor your blink rate for fatigue")
                self.speak("Automatically log in to services")
                self.speak("Lock, shutdown, or restart your computer")
                self.speak("Play music based on your mood")
                self.speak("And more! Just ask me to do something specific.")
            
            # Default response - use AI
            else:
                response = self.ask_groq(voice_data)
                self.speak(response)

        if "lock my computer" in self.voice_data:
            self.speak("Locking your computer.")
            os.system("rundll32.exe user32.dll,LockWorkStation")  # Locks Windows

        elif "shutdown my pc" in self.voice_data:
            self.speak("Shutting down your computer.")
            os.system("shutdown /s /t 0")  # Shutdown immediately

        elif "restart my pc" in self.voice_data:
            self.speak("Restarting your computer.")
            os.system("shutdown /r /t 0")  # Restart immediately

        elif "cancel shutdown" in self.voice_data:
            self.speak("Cancelling scheduled shutdown.")
            os.system("shutdown /a")  # Aborts any scheduled shutdown

        elif "auto-lock timer" in self.voice_data:
            self.speak("Setting auto-lock timer.")
            # Example: Auto-lock after 60 seconds if no face detected
            os.system("timeout /t 60 && rundll32.exe user32.dll,LockWorkStation")

        elif "auto-shutdown timer" in self.voice_data:
            self.speak("Setting auto-shutdown timer.")
            # Example: Auto-shutdown after 300 seconds (5 minutes) of inactivity
            os.system("shutdown /s /t 300")

# Main function
if __name__ == "__main__":
    assistant = JarvisAssistant()
    assistant.run()