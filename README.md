# ECHS-Lab
TA-1

Navigate to the DeepStream Folder

Open the DeepStream installation folder, typically located at /opt/nvidia/deepstream.
Use the command cd /opt/nvidia/deepstream in your terminal to navigate to this directory. This is where all the required files and test applications for DeepStream are located.
Check CUDA Version

Run nvcc --version in the terminal. This command verifies the installed CUDA version on your system.
Knowing the CUDA version is critical because DeepStream depends on specific versions for compatibility.
Switch to Root User

Use sudo su to gain root privileges. This is required when making system-level changes or modifying files that need administrative access.
Enter your system password to authenticate.
Modify the Makefile to Match the CUDA Version

Open the Makefile in a text editor, such as gedit, by running sudo gedit Makefile.
Locate the section where the CUDA version is specified. Update it to the version you obtained from the nvcc --version command if it differs.
Save the changes and close the editor.
Refer to the README File

The README file in the DeepStream directory typically provides detailed instructions on running specific applications.
Open the file using a text viewer (cat README or less README) or a text editor to find the exact command for running the deepstream-test1-app file.
Build the Application with Root Privileges

Use sudo make to compile the application.
This step ensures all necessary dependencies are linked, and administrative permissions handle system-level builds if required.

TA-2

Gesture Recognition Using ONNX Model on Jetson Orin Nano
Prerequisites
Jetson Orin Nano Setup

Ensure the Jetson Orin Nano is powered and configured with the JetPack SDK installed.
Install OpenCV, Numpy, and ONNX Runtime:
sudo apt update
sudo apt install python3-opencv python3-pip
pip3 install numpy onnxruntime


Clone this project repository:
git clone https://github.com/<username>/<repo-name>.gitcd <repo-name>
Model Preparation

Place the ONNX model file (gesture_model.onnx) in the project directory.
Ensure the model is compatible with the input size 64x64x3.
Steps to Run the Gesture Recognition Project

1. Connect the Camera
Attach a USB camera to the Jetson Orin Nano.
Ensure it is recognized using the command:
ls /dev/video*
Note the camera index (e.g., /dev/video0).
2. Run the Gesture Recognition Script
Open the terminal and navigate to the project directory:
cd <repo-name>
Execute the Python script:
python3 gesture_recognition.py
3. Adjust Camera Settings
Modify the camera index in the script (cv2.VideoCapture(0)) if the default camera is not /dev/video0.
4. Gesture Recognition Process
The script will:
Capture live video from the camera.
Resize and preprocess each frame to match the model's input size.
Perform inference using the ONNX model.
Display the recognized gesture on the video feed.
5. Exit the Application
Press the q key to terminate the program.

