# Jarvis AI - Intelligent Voice Assistant

## Project Overview
Jarvis AI is a Python-based voice assistant designed to emulate Tony Stark's JARVIS from the Iron Man series. It integrates various functionalities such as speech recognition, AI-powered assistance, facial recognition, mood detection, and PC automation to provide users with an interactive and efficient experience.

## Features
- **Speech Recognition & Text-to-Speech**: Utilizes `speech_recognition` for capturing user input and `pyttsx3` for voice responses, facilitating seamless voice-based interactions.
- **AI Integration with Groq AI**: Incorporates the Groq API to process user queries using the `llama3-8b-8192` model, delivering intelligent and context-aware responses.
- **Web & Information Retrieval**: Performs web searches and retrieves information using the `googlesearch-python` library, enabling quick access to online data.
- **Facial Recognition**: Employs `face_recognition` and `dlib` libraries to identify known individuals, enhancing personalized interactions.
- **Mood Detection**: Analyzes facial expressions to determine the user's mood, allowing for empathetic responses.
- **PC Automation**: Automates tasks such as opening applications, controlling system volume, and managing files, streamlining daily operations.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - `speech_recognition`
  - `pyttsx3`
  - `requests`
  - `face_recognition`
  - `dlib`
  - `opencv-python`
  - `googlesearch-python`

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Jashvanth18/Jarvis-AI.git
   cd Jarvis-AI
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   # Activate the virtual environment
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dlib Face Recognition Models**:
   - Obtain the `shape_predictor_68_face_landmarks.dat` file from the dlib model repository and place it in the project directory.

5. **Configure API Keys**:
   - Create a `.env` file in the project directory with your Groq API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

## Usage
1. **Run the Application**:
   ```bash
   python Jarvis.py
   ```

2. **Interact with Jarvis**:
   - Use voice commands to interact with Jarvis. For example:
     - "What's the weather today?"
     - "Open Google Chrome."
     - "Who am I?" (for facial recognition)

3. **Facial Recognition Setup**:
   - To add known faces, use the `encode_faces.py` script to encode images and update the `known_faces.pkl` file.

## Contributing
Contributions are welcome! Feel free to fork the repository, make enhancements, and submit pull requests.

## License
This project is licensed under the MIT License.

## Contact
For any queries or support, reach out to [Jashvanth](mailto:jashvanth@example.com).

