import cv2
import numpy as np
import onnxruntime as ort

# Load the ONNX model
model_path = "gesture_model.onnx"
session = ort.InferenceSession(model_path)

# Get input and output layer information
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Set the image size (must match the model input size)
img_height, img_width = 64, 64

# Capture video from the camera
cap = cv2.VideoCapture(0)  # Adjust the camera index if needed

# Ensure the camera is opened
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame
    img = cv2.resize(frame, (img_height, img_width))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run inference
    onnx_inputs = {input_name: img}
    predictions = session.run([output_name], onnx_inputs)[0]

    # Get the predicted class
    predicted_class = np.argmax(predictions)

    # Display the predicted class on the frame
    cv2.putText(frame, f"Gesture: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
