from flask import Flask, render_template, Response
import cv2
import dlib

app = Flask(__name__)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    eye_positions = []

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = landmarks.part(36).x, landmarks.part(36).y
        right_eye = landmarks.part(45).x, landmarks.part(45).y
        eye_positions.append((left_eye, right_eye))

    return eye_positions

def eye_direction(eye_positions, frame_width, frame_height):
    directions = []
    for (left_eye, right_eye) in eye_positions:
        eye_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2,
        )

        x, y = eye_center
        direction = ""

        if x < frame_width // 3:
            direction = "Left"
        elif x > 2 * frame_width // 3:
            direction = "Right"
        elif y < frame_height // 3:
            direction = "Up"
        elif y > 2 * frame_height // 3:
            direction = "Down"
        else:
            direction = "Center"

        directions.append(direction)

    return directions

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        frame = cv2.flip(frame, 1)
        eye_positions = detect_eyes(frame)
        frame_height, frame_width, _ = frame.shape
        directions = eye_direction(eye_positions, frame_width, frame_height)

        for i, direction in enumerate(directions):
            cv2.putText(
                frame,
                f"Eye {i+1}: {direction}",
                (20, 40 * (i + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
