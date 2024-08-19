# Hand_gesture_recognition
# Description and explanation
This project presents a deep learning-driven approach for hand gesture recognition using wearable vision sensors, designed for interactive virtual museum environments. The classification task consists of recognizing seven distinct hand gestures: Like, Dislike, Ok, Point, Take picture, Slide right, and Slide left. The performance of the proposed method is evaluated using the Interactive Museum for Gesture Recognition Dataset:   http://imagelab.ing.unimore.it/files/ego_virtualmuseum.zip

I initially implemented a basic solution using classical machine learning techniques, including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest. However, the observed performance was not satisfactory, with an accuracy of around 76%. To enhance classification accuracy, I developed a deep learning solution utilizing a Convolutional Neural Network (CNN). This approach improved accuracy to 86%, which was better than the classical methods but still unsatisfactory. Finally, I implemented a solution combining CNN for feature extraction with Long Short-Term Memory (LSTM) for classification. This approach achieved the best results so far, with accuracy exceeding 99%.

# Structure and important details
- The files "Classic_Techniques.ipynb" and "Deep_Learning_Techniques.ipynb" contain the source code used for constructing, training, and evaluating the classic and deep learning techniques using Google Collab.
- To enhance user experience, I added the "Virtual_museum.py" file which provides a user interface representing a virtual museum. Here, users can navigate through the museum and perform the seven hand gestures by either passing videos or using an embedded camera, ideally positioned around the shoulder area.
- It's important to note that the "files" folder contains all necessary resources, including model weights, the user interface, the database, etc. To connect to an already created user, use admin admin.

# Data citation 
If you find this code useful for your scientific research, please cite the following papers associated with this code:
- Nabil Zerrouki, Fouzi Harrou, Amrane Houacine, Riadh Bouarroudj, Mohammed Yazid Cherifi, Ait-Djafer Amina Zouina, Ying Sun, Deep Learning for Hand Gesture Recognition in Virtual Museum Using Wearable Vision Sensors, in IEEE Sensors Journal, vol. 24, no. 6, pp. 8857-8869, 15 March15, 2024. https://doi.org/10.1109/JSEN.2024.3354784
- N. Zerrouki, A. Houacine, F. Harrou, R. Bouarroudj, M. Y. Cherifi and Y. Sun, "Exploiting Deep Learning-Based LSTM Classification for Improving Hand Gesture Recognition to Enhance Visitors’ Museum Experiences," 2022 International Conference on Innovation and Intelligence for Informatics, Computing, and Technologies (3ICT), Sakheer, Bahrain, 2022, pp. 451-456. https://doi.org/10.1109/3ICT56508.2022.9990722
