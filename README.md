# Bone Fracture Classifier

## Project Overview
This project is focused on building and deploying a deep learning model that can classify bone X-ray images as either containing a fracture or not. The model is implemented using TensorFlow and Keras, and the web interface is built with Streamlit.

## Features
- **Model Architecture**: A convolutional neural network (CNN) built with TensorFlow/Keras.
- **Image Preprocessing**: Images are resized and normalized before prediction.
- **Web App**: A Streamlit-based app where users can upload X-ray images for real-time classification.

## Technologies Used
- Python (TensorFlow, Keras, NumPy, Pillow)
- Streamlit for the web interface
- Jupyter Notebook for model training

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/parthdave333/bone-fracture-classifier.git

2. Install the required dependencies:
   ```bash
      pip install -r requirements.txt

3. Download the dataset:

Visit [Kaggle's Bone Fracture Classification dataset page](https://www.kaggle.com/datasets/shyamgupta196/bone-fracture-split-classification).
Log in or create a Kaggle account if necessary.
Click on Download to get the dataset and place it in the project directory.
Place the model.h5 file in the project root if not already present.

## Running the App
Run the Streamlit app:

    ```bash
       streamlit run deploy.py
Use the app to upload X-ray images and receive real-time fracture classification results.

## Dataset
The dataset contains X-ray images that are pre-split for training and testing. Ensure that the data is correctly organized for model training or use the provided model.h5 to directly deploy and test the app.

## Results
The model provides predictions on whether an uploaded X-ray image indicates a fracture or not, with accompanying probability scores.

## Future Enhancements
Adding more image preprocessing techniques.
Expanding the model to classify different types of bone fractures.
Improving the UI/UX of the web application for better user interaction.

## Contributing
Contributions are welcome! Fork the repository, make your changes, and submit a pull request.
