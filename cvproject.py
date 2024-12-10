import streamlit as st
import numpy as np
import pickle
from PIL import Image
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

# Load the machine learning model for classification
model_path = r'D:\RandomForestModel.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# HOG parameters for classification
hog_params = {
    'orientations': 18,
    'pixels_per_cell': (6, 6),
    'cells_per_block': (3, 3),
    'block_norm': 'L2-Hys',
}

# Function to preprocess the image and return HOG features and HOG image
def preprocess_image(image):
    image = np.array(image)
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_img = cv2.resize(grayscale_img, (150, 150))
    grayscale_img = cv2.GaussianBlur(grayscale_img, (5, 5), 0)
    
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened_img = cv2.filter2D(grayscale_img, -1, sharpening_kernel)
    
    _, hog_image1 = hog(sharpened_img, visualize=True, **hog_params)
    processed_image = sharpened_img.flatten()
    
    return processed_image, hog_image1

# Function to process video for segmentation
def process_video(uploaded_video):
    offset = 6
    delay = 60
    detec = []
    cars = 0
    
    cap = cv2.VideoCapture(uploaded_video)
    MOG = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=250)
    
    while True:
        ret, frame1 = cap.read()
        if not ret:
            break

        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = MOG.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5), np.uint8))
        contour, _ = cv2.findContours(dilat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.line(frame1, (25, 550), (1200, 550), (176, 130, 39), 2)

        for (i, c) in enumerate(contour):
            (x, y, w, h) = cv2.boundingRect(c)
            if (w >= 80 and h >= 80):
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame1, "Detected", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (200, 50, 0), 3)
                centre = (x + int(w / 2), y + int(h / 2))
                detec.append(centre)
                cv2.circle(frame1, centre, 4, (0, 0, 255), -1)

        # Track detected vehicles
        for (x, y) in detec:
            if ((y < (550 + offset)) and (y > (550 - offset))) or y == 550:
                cars += 1
                cv2.line(frame1, (25, 550), (1200, 550), (0, 127, 255), 3)
                detec.remove((x, y))
        
        cv2.putText(frame1, "VEHICLE COUNT: " + str(cars), (320, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Video Original", frame1)
        cv2.imshow("Segmentation", dilat)
        if cv2.waitKey(1) == 27:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    
    
    
# HTML and CSS for intro page
intro_page_html = st.markdown("""
    <style>
    .header {
        text-align: center;
        padding: 20px;
    }
    .header h1 {
        font-size: 3em;
        color: #2c3e50;
    }
    .header p {
        font-size: 1.2em;
        color: #34495e;
    }
    .content {
        text-align: center;
        padding: 20px;
    }
    .team {
        font-size: 1.1em;
        color: #16a085;
        text-align: left;
        margin: 20px;
    }
    .team h4 {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .start-btn {
        background-color: #2ecc71;
        color: white;
        font-size: 1.5em;
        padding: 15px 40px;
        border: none;
        cursor: pointer;
        border-radius: 5px;
    }
    .start-btn:hover {
        background-color: #27ae60;
    }
    </style>
    """, unsafe_allow_html=True)

# Show the professional project intro and team details if the session_state is not set
if 'started' not in st.session_state or not st.session_state.started:
    st.markdown("""
        <div class="header">
            <h1>🚗🚲 Vehicle Classification & Segmentation</h1>
            <p>A comprehensive vehicle classification and segmentation project that leverages machine learning and computer vision for real-time vehicle detection and counting.</p>
        </div>
        <div class="content">
            <h3>Project Description</h3>
            <p>This project aims to provide two core functionalities: vehicle classification and segmentation. 
            For classification, the application uses Histogram of Oriented Gradients (HOG) features and a Random Forest classifier to distinguish between vehicles like cars and bikes. 
            For segmentation, it applies background subtraction and contour detection to identify and count the vehicles in uploaded video streams.</p>
        </div>
        <div class="team">
            <h4>Project Team:</h4>
            <ul>
                <li><strong>Mohammed Hamza</strong></li>
                <li><strong>Osama Abo-bakr</strong></li>
                <li><strong>Zeyad ayman</strong></li>
                <li><strong>Ehab Negm</strong></li>
                <li><strong>Yousef Selim</strong></li>
                <li><strong>Ali Mohammed</strong></li>
                <li><strong>Rokia Osama</strong></li>
                <li><strong>Alzahraa Ali</strong></li>
                <li><strong>Farah ayman</strong></li>
                <li><strong>Hoda elwakeel</strong></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Get Started"):
        st.session_state.started = True
        st.experimental_rerun()  # Rerun the app to hide the intro page and show the next steps


else:

    # Streamlit app layout
    st.title("🚗🚲 Image Classification or Segmentation")
    st.markdown("""
    ### Choose whether you want to do **Classification** or **Segmentation**.
    """)

    # Page selection: Classification or Segmentation
    option = st.radio("Choose an option", ('Classification', 'Segmentation'))

    if option == 'Classification':
        # Classification page
        st.markdown("""
        Upload an image of a **car** or **bike** to predict the category.
        This application uses Histogram of Oriented Gradients (HOG) and a trained Random Forest model to classify images as either a car or a bike.
        """)

        # Categories for prediction
        Categories = ['Car', 'Bike']

        # File uploader for image
        uploaded_file = st.file_uploader("Upload an image (JPEG/PNG)...", type=["jpg", "png", "jpeg"])

        
        if uploaded_file is not None:
            # Open and display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)
            
            # Preprocess the image and extract HOG features
            processed_image, hog_image1 = preprocess_image(image)
            
            # Display HOG image in grayscale
            st.markdown("### HOG Feature Visualization")
            st.write("The grayscale image below shows the HOG features extracted from the uploaded image. These features are used by the model for prediction.")
            
            fig, ax = plt.subplots()
            ax.imshow(hog_image1, cmap='gray')
            ax.axis('off')  # Hide axis
            st.pyplot(fig)
            
            # Reshape the processed image for model input
            processed_image = processed_image.reshape(1, -1)

            # Make a prediction using the loaded model
            prediction = model.predict(processed_image)

            # Display the prediction result in a professional manner with red color for the category
            st.markdown("## Prediction Result")
            predicted_category = Categories[int(prediction[0])]
            
            # Display a success message
            st.success("Prediction Successful!")
            result_text = f"""
            Based on the image you uploaded, the model has classified it as a **<span style="color:red;">{predicted_category}</span>**.
            
            **Confidence Level:** {np.round(model.predict_proba(processed_image)[0][int(prediction[0])]*100, 2)}%
            """
            
            # Render the result with HTML to display the red color
            st.markdown(result_text, unsafe_allow_html=True)

            # Display additional model insights
            st.markdown("""
            ### About the Model
            The model uses a **Random Forest Classifier** trained on HOG features to distinguish between cars and bikes. This type of classifier works by constructing multiple decision trees and outputting the class prediction that represents the majority vote of the trees.
            """)


    elif option == 'Segmentation':
        # Segmentation page
        st.markdown("""
        Upload a video to detect and count the number of vehicles (cars) using background subtraction and contour detection.
        """)

        # File uploader for video
        uploaded_video = st.file_uploader("Upload a video (MP4)...", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            # Save the video to a file
            with open("uploaded_video.mp4", "wb") as f:
                f.write(uploaded_video.read())

            # Process the video for segmentation
            st.video(uploaded_video)
            st.write("Processing video... Please wait.")
            process_video("uploaded_video.mp4")

