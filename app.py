
import os
from dotenv import load_dotenv
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai
import threading

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

canvas = None
prev_pos = None
last_ai_output = ""
ai_processing = False
state_lock = threading.Lock()

def get_hand_info(img_input):
    hands, img_output = detector.findHands(img_input, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return (fingers, lmList), img_output
    else:
        return None, img_output

def draw_on_canvas(hand_info, prev_pos, canvas, base_img):
    fingers, lmlist = hand_info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # Only index finger up
        current_pos = lmlist[8][0:2]
        if prev_pos is not None:
            cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
        prev_pos = current_pos
    else:
        prev_pos = None  # Reset prev_pos if not drawing gesture

    if fingers == [1, 0, 0, 0, 0]:  # Only thumb up
        canvas = np.zeros_like(base_img)
        prev_pos = None

    return prev_pos, canvas

def run_ai_inference(canvas_image):
    global last_ai_output, ai_processing
    with state_lock:
        ai_processing = True
        last_ai_output = "Thinking..."
    try:
        pil_img = Image.fromarray(cv2.cvtColor(canvas_image, cv2.COLOR_BGR2RGB))
        response = model.generate_content(["Solve this Math Problem based on the drawing:", pil_img])
        ai_result = response.text if response and response.text else "No response from AI."
    except Exception as e:
        ai_result = f"Error from AI: {str(e)}"
    with state_lock:
        last_ai_output = ai_result
        ai_processing = False

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global canvas, prev_pos, ai_processing
    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)

        with state_lock:
            if canvas is None or canvas.shape != img.shape:
                canvas = np.zeros_like(img)

            hand_info, img_annotated = get_hand_info(img)

            if not ai_processing:
                if hand_info:
                    prev_pos, canvas = draw_on_canvas(hand_info, prev_pos, canvas, img)

            if hand_info and hand_info[0] == [1, 1, 1, 1, 0] and not ai_processing:
                canvas_for_ai = canvas.copy()
                ai_thread = threading.Thread(target=run_ai_inference, args=(canvas_for_ai,))
                ai_thread.start()

            image_combined = cv2.addWeighted(img_annotated, 0.9, canvas, 0.9, 0)

        ret, buffer = cv2.imencode('.jpg', image_combined)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ai_output')
def ai_output():
    global last_ai_output, ai_processing
    with state_lock:
        return jsonify({"output": last_ai_output, "processing": ai_processing})

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
