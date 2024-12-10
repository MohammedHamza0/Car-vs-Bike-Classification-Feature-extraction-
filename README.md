# Vehicle Classification & Segmentation

This project utilizes machine learning and computer vision to classify and segment vehicles in images and videos. The system applies Histogram of Oriented Gradients (HOG) features and a trained Random Forest model for classification, and background subtraction with contour detection for vehicle segmentation. The project provides an intuitive interface via a Streamlit web application.

---

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)

---

## Project Description

This project is designed to classify vehicles as either cars or bikes using an image classifier and count vehicles in videos using segmentation techniques. It leverages powerful machine learning models and computer vision techniques to achieve real-time predictions.

### Key Features:

1. **Vehicle Classification**: Classify uploaded images of vehicles as either a car or a bike using HOG features and a trained Random Forest classifier.
2. **Vehicle Segmentation**: Process uploaded videos to detect and count vehicles using background subtraction and contour detection.

The application is built using Streamlit for a user-friendly interface and supports both image and video processing.

---

## Technologies Used

- **Python**: The main programming language used.
- **Streamlit**: For creating the interactive web application.
- **OpenCV**: Used for image processing and video analysis.
- **Scikit-Image**: For extracting Histogram of Oriented Gradients (HOG) features.
- **Pickle**: For loading the pre-trained Random Forest model.
- **Matplotlib**: For visualizing the HOG features.

---

## Getting Started

Follow these steps to get the project up and running locally:

### Prerequisites

- Python 3.x
- Install the required Python libraries:
  ```bash
  pip install streamlit opencv-python-headless scikit-image matplotlib pickle
  ```

### Clone the Repository

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/MohammedHamza0/Car-vs-Bike-Classification-Feature-extraction.git

   ```

2. Navigate to the project directory:
   ```bash
   cd Car-vs-Bike-Classification-Feature-extraction
   ```

### Running the App

1. To run the application, execute:
   ```bash
   streamlit run app.py
   ```

2. Open the application in your browser (usually at `http://localhost:8501`).

---

## Usage

Once the application is running, you can choose between two options:

### Classification
- Upload an image (JPG, PNG) of a vehicle.
- The model will predict whether the vehicle is a **car** or a **bike** using a trained Random Forest model.

### Segmentation
- Upload a video (MP4, AVI, MOV) containing vehicles.
- The application will process the video, apply background subtraction, and count the vehicles in the video.

---

