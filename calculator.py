from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import io

app = Flask(__name__)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Google Gemini API
genai.configure(api_key="AIzaSyDz0rDby-dZlcWxZ_ruhvRSX3CbJK4tJQ8")
model = genai.GenerativeModel('gemini-1.5-flash')

# Hand detector
detector = HandDetector(staticMode=False, maxHands=2, detectionCon=0.7, minTrackCon=0.5)

prev_pos = None
canvas = None

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas, img):
    fingers, lmlist = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # Draw with index finger
        current_pos = lmlist[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)

    elif fingers == [1, 0, 0, 0, 0]:  # Clear canvas
        canvas = np.zeros_like(img)

    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # Solve
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this Math Problem", pil_image])
        return response.text if response else "No response from AI."
    return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        global prev_pos, canvas
        while True:
            success, img = cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            if canvas is None:
                canvas = np.zeros_like(img)

            info = getHandInfo(img)
            if info:
                fingers, lmlist = info
                prev_pos, canvas = draw(info, prev_pos, canvas, img)
                answer = sendToAI(model, canvas, fingers)
                if answer:
                    yield f"data:{answer}\n\n"

            image_combined = cv2.addWeighted(img, 0.9, canvas, 0.9, 0)
            _, buffer = cv2.imencode('.jpg', image_combined)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    # Optional: implement separate button-based solving
    pass

if __name__ == '__main__':
    app.run(debug=True)



# import cvzone
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import google.generativeai as genai
# from PIL import Image
# import streamlit as st

# # Streamlit UI Setup
# st.set_page_config(layout="wide")
# st.image('MathGestures.png')

# col1, col2 = st.columns([3,2])
# with col1:
#     run = st.checkbox('Run', value=True)
#     FRAME_WINDOW = st.image([])

# with col2:
#     st.title("Answer")
#     output_text_area = st.subheader("")

# # Configure Google AI Model
# genai.configure(api_key="AIzaSyDz0rDby-dZlcWxZ_ruhvRSX3CbJK4tJQ8")  # Replace with a valid API key
# model = genai.GenerativeModel('gemini-1.5-flash')

# # Initialize webcam
# cap = cv2.VideoCapture(0)  # Change to 1 if using an external webcam
# cap.set(3, 1280)
# cap.set(4, 720)

# # Initialize Hand Detector
# detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# def getHandInfo(img):
#     """Detect hands and return finger information"""
#     hands, img = detector.findHands(img, draw=False, flipType=True)
#     if hands:
#         hand = hands[0]  # Get the first detected hand
#         lmList = hand["lmList"]  # List of 21 landmarks
#         fingers = detector.fingersUp(hand)
#         print(fingers)
#         return fingers, lmList
#     return None

# def draw(info, prev_pos, canvas):
#     """Draw lines on the canvas based on hand gestures"""
#     fingers, lmList = info
#     current_pos = None
#     if fingers == [0, 1, 0, 0, 0]:  # Index finger up
#         current_pos = lmList[8][0:2]
#         if prev_pos is None:
#             prev_pos = current_pos
#         cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
#     elif fingers == [1, 0, 0, 0, 0]:  # Thumb up (Erase)
#         canvas = np.zeros_like(canvas)
#     return current_pos, canvas

# def sendToAI(model, canvas, fingers):
#     """Send the equation image to AI for solving"""
#     if fingers == [1, 1, 1, 1, 0]:  # When all four fingers are up
#         pil_image = Image.fromarray(canvas)
#         response = model.generate_content(["Solve this math problem ", pil_image])
#         return response.text
#     return ""

# prev_pos = None
# canvas = None

# # Streamlit Loop
# while run:
#     success, img = cap.read()
#     if not success or img is None:
#         st.error("Error: No frame captured. Check your webcam.")
#         break

#     img = cv2.flip(img, 1)

#     if canvas is None:
#         canvas = np.zeros_like(img)

#     info = getHandInfo(img)
#     output_text = ""

#     if info:
#         fingers, lmList = info
#         prev_pos, canvas = draw(info, prev_pos, canvas)
#         output_text = sendToAI(model, canvas, fingers)

#     image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
#     FRAME_WINDOW.image(image_combined, channels="BGR")

#     if output_text:
#         output_text_area.text(output_text)

#     cv2.waitKey(1)

# cap.release()
# cv2.destroyAllWindows()








# import cv2
# import numpy as np
# import pytesseract
# import sympy as sp
# import re
# import streamlit as st
# from cvzone.HandTrackingModule import HandDetector

# # Streamlit UI Setup
# st.set_page_config(layout="wide")
# st.image('MathGestures.png')

# col1, col2 = st.columns([3, 2])
# with col1:
#     run = st.checkbox('Run', value=True)
#     FRAME_WINDOW = st.image([])

# with col2:
#     st.title("Answer")
#     output_text_area = st.subheader("")

# # Initialize webcam
# cap = cv2.VideoCapture(0)  
# cap.set(3, 1280)
# cap.set(4, 720)

# # Initialize Hand Detector
# detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# def getHandInfo(img):
#     """Detect hands and return finger information"""
#     hands, img = detector.findHands(img, draw=False, flipType=True)
#     if hands:
#         hand = hands[0]  
#         lmList = hand["lmList"]  
#         fingers = detector.fingersUp(hand)
#         return fingers, lmList
#     return None

# def draw(info, prev_pos, canvas):
#     """Draw lines on the canvas based on hand gestures"""
#     fingers, lmList = info
#     current_pos = None
#     if fingers == [0, 1, 0, 0, 0]:  
#         current_pos = lmList[8][0:2]
#         if prev_pos is None:
#             prev_pos = current_pos
#         cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
#     elif fingers == [1, 0, 0, 0, 0]:  
#         canvas = np.zeros_like(canvas)
#     return current_pos, canvas

# def solve_equation(equation):
#     """Solve the equation using sympy"""
#     try:
#         equation = equation.replace("=", "-(") + ")"  
#         x = sp.symbols('x')  
#         solution = sp.solve(sp.sympify(equation), x)
#         return str(solution)
#     except Exception as e:
#         return f"Error: {e}"

# def clean_equation(equation):
#     """Clean up the extracted equation using regex"""
#     equation = equation.replace(" ", "")  
#     equation = equation.replace("—", "-")  
#     equation = equation.replace("x", "*")  
#     equation = equation.replace("÷", "/")  
    
#     match = re.search(r'^[\d+\-*/=()x]+$', equation)
#     return match.group() if match else None

# def extract_equation_from_image(canvas):
#     """Extract equation using improved OCR preprocessing"""
    
#     gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
#     binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#     kernel = np.ones((2, 2), np.uint8)
#     processed = cv2.dilate(binary, kernel, iterations=1)

#     # OCR with specific PSM mode for single-line math equations
#     custom_config = r'--oem 3 --psm 7'
#     equation = pytesseract.image_to_string(processed, config=custom_config)

#     cleaned_equation = clean_equation(equation.strip())

#     if cleaned_equation:
#         print(f"Extracted Equation: {cleaned_equation}")  
#         return cleaned_equation
#     else:
#         return "Error: Could not recognize equation. Try writing more clearly."

# def process_equation(canvas, fingers):
#     """Process the equation and return the solution"""
#     if fingers == [1, 1, 1, 1, 1]:  
#         equation = extract_equation_from_image(canvas)
#         if "Error" not in equation:
#             solution = solve_equation(equation)
#             return f"Equation: {equation}\nSolution: {solution}"
#     return ""

# prev_pos = None
# canvas = None

# # Streamlit Loop
# while run:
#     success, img = cap.read()
#     if not success or img is None:
#         st.error("Error: No frame captured. Check your webcam.")
#         break

#     img = cv2.flip(img, 1)

#     if canvas is None:
#         canvas = np.zeros_like(img)

#     info = getHandInfo(img)
#     output_text = ""

#     if info:
#         fingers, lmList = info
#         prev_pos, canvas = draw(info, prev_pos, canvas)
#         output_text = process_equation(canvas, fingers)

#     image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
#     FRAME_WINDOW.image(image_combined, channels="BGR")

#     if output_text:
#         output_text_area.text(output_text)

#     cv2.waitKey(1)

# cap.release()
# cv2.destroyAllWindows()
