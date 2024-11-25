import tensorflow as tf
import tf2onnx
# Load the trained Keras model
model = tf.keras.models.load_model('trained_model.h5')

# Convert the model to ONNX format
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

# Save the ONNX model
with open("trained_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model successfully converted to ONNX format and saved as trained_model.onnx")

