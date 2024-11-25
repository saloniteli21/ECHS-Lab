# ECHS-Lab
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
