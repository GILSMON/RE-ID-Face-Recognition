Face Recognition and Tracking ProjectThis repository contains a suite of scripts for face recognition and tracking using the DeepFace and DeepSORT algorithms. The project is structured to allow for both basic functionality and performance evaluation of the implemented algorithms.Getting StartedPrerequisitesTo get a copy of this project up and running on your local machine, you'll need Python 3.x and pip.InstallationClone the repository:git clone https://github.com/your-username/your-repository.git
cd your-repository
Create and activate a virtual environment:python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Install the required packages. You may need to create a requirements.txt file first by running pip freeze > requirements.txt if you have all your dependencies installed.pip install -r requirements.txt
Project Structureface_rec_deepface_performance_track.py: This script is dedicated to face recognition using the DeepFace algorithm. It includes code to print performance metrics to the terminal and saves the output to a performance file.face_rec_deepface.py: This file also performs face recognition with the DeepFace algorithm but is a lighter version as it does not include the code for printing or saving performance metrics.face_rec_deepSORT_performence_track.py: This script integrates both DeepFace for recognition and DeepSORT for tracking. It is designed to track and evaluate the performance of this combined approach, including printing to the terminal and creating a performance file.face_rec_deepSORT_tracking.py: This file uses both DeepFace and DeepSORT for combined face recognition and tracking without the overhead of performance logging. It is ideal for general-purpose use.UsageInstructions on how to run the main scripts would go here. For example:# To run the DeepFace performance tracker
python face_rec_deepface_performance_track.py

# To run the DeepSORT tracking script
python face_rec_deepSORT_tracking.py
