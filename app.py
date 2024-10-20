from flask import Flask, render_template, Response
import cv2
import torch

app = Flask(__name__)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')

def gen_frames():
    cap = cv2.VideoCapture(0)  # Open the camera
    while True:
        success, frame = cap.read()  # Read a frame from the camera
        if not success:
            break
        else:
            results = model(frame)  # Perform detection using the YOLO model
            accuracy_sum = 0
            count = 0
            for box in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = box
                accuracy_sum += conf.item()  # Sum the confidence scores
                count += 1  # Count the number of detections
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Draw bounding box
                cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Label the box
            
            # Calculate the average accuracy
            avg_accuracy = accuracy_sum / count if count else 0
            # Display the average accuracy on the frame
            cv2.putText(frame, f'Average Accuracy: {avg_accuracy:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)  # Encode the frame as JPEG
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Stream the frame to the client

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
