The Jarvis AI Voice Assistant in your code provides a variety of features, including AI-powered assistance, facial recognition, mood detection, and PC automation. Here is a structured list of the available features:  

Core Features  
1. Speech recognition & text-to-speech  
   - Listens to user input using `speech_recognition`  
   - Converts text to speech using `pyttsx3`  
   - Supports voice-based commands  

2. AI integration with Groq AI  
   - Uses `Groq API` for answering user queries  
   - Sends questions and retrieves responses from an AI model (`llama3-8b-8192`)  

3. Web & information retrieval  
   - Performs web searches on Google  
   - Searches and plays videos on YouTube  
   - Fetches weather updates using `OpenWeatherMap API`  

4. PC control & security  
   - Lock PC: Automatically locks the computer  
   - Shutdown PC: Schedules or cancels a system shutdown  
   - Restart PC: Initiates a restart sequence  
   - Auto-lock timer: Locks the PC after a specified time of inactivity  
   - Auto-shutdown timer: Shuts down the PC after a specified time  

Facial recognition & mood detection  
5. Face recognition & detection  
   - Registers known faces and stores them  
   - Recognizes registered users  
   - Detects and alerts when an unknown face appears  
   - Saves and loads known faces using a pickle file (`known_faces.pkl`)  

6. Intruder detection & alert system  
   - Detects strangers and alerts the user  
   - Captures an image and sends it via email  
   - Locks the PC upon detecting an unknown face  

7. Facial landmark detection  
   - Uses `dlib`’s shape predictor (68 facial landmarks)  
   - Detects eye positions, mouth, and nose positions  

8. Blink detection & fatigue monitoring  
   - Monitors eye blinks using Eye Aspect Ratio (EAR)  
   - Alerts if the user shows signs of fatigue  
   - Detects if the head is drooping (possible drowsiness)  

9. Mood detection using DeepFace  
   - Analyzes facial expressions  
   - Detects emotions such as happy, sad, angry, and neutral  
   - Suggests music based on detected mood  

Entertainment & media control  
10. Music recommendation based on mood  
    - Plays YouTube playlists based on emotions  
    - Supports moods like sad, angry, happy, and neutral  

Automation & auto-login  
11. Auto-login to websites  
    - Uses `Selenium` for automated login  
    - Supports login to YouTube, Gmail, Netflix, and Spotify  
    - Stores credentials securely  

12. Toggle features  
    - Enables/disables auto-login  
    - Enables/disables blink detection  
    - Allows setting timers for auto-lock and auto-shutdown  

Assistant core commands  
13. General commands  
    - Provides the current time and date  
    - Answers questions using AI  
    - Responds to greetings and exit commands  
    - Provides a help menu listing available features  
