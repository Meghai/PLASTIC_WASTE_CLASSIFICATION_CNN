Overview
This project aims to develop a Convolutional Neural Network (CNN) model to classify plastic waste images into categories, improving waste management systems. The goal is to automate waste segregation and recycling using deep learning techniques, contributing to a more sustainable waste disposal system.

Table of Contents
Project Description
Dataset
Model Architecture
Training
Weekly Progress
How to Run
Technologies Used
Future Scope
Contributing
License
Project Description
Plastic pollution is a growing global issue, and efficient waste segregation is key to addressing it. This project utilizes a CNN model to classify images of plastic waste into distinct categories—Organic and Recyclable—using machine learning techniques. The project seeks to automate waste management systems, reducing human error and improving recycling efficiency.

Dataset
The dataset used is the Waste Classification Data by Sashaank Sekar, containing 25,077 labeled images divided into two categories: Organic and Recyclable. This dataset enables the classification of plastic waste, helping to automate waste management systems.
Dataset Link:
You can access the dataset here: [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data).
Dataset Details:

Total Images: 25,077
Training Data: 22,564 images (85%)
Test Data: 2,513 images (15%)
Categories: Organic and Recyclable
Dataset Link: Waste Classification Data

Note: Ensure you follow appropriate dataset licensing and usage guidelines.

Model Architecture
The CNN architecture consists of:

Convolutional Layers: For feature extraction
Pooling Layers: For dimensionality reduction
Fully Connected Layers: For classification
Activation Functions: ReLU and Softmax
A visual representation of the basic CNN architecture is included in the project files.

Training
Optimizer: Adam
Loss Function: Categorical Crossentropy
Epochs: Configurable (default: 25)
Batch Size: Configurable (default: 32)
Data augmentation techniques are used to enhance performance and prevent overfitting.
Weekly Progress
This section will be updated regularly with details on the progress of the project.

Week 1: Libraries, Data Import, and Setup

Date: January 20, 2025 – January 27, 2025
Activities:
Imported libraries and frameworks
Set up project environment
Explored dataset structure
Notebooks:
Week1 - Libraries - Importing Data - Setup
Kaggle Notebook: Link
Week 2: TBD
Details will be added after completion.

Week 3: TBD
Details will be added after completion.

How to Run
To run the project, follow these steps:

Clone the repository:

bash
Copy
git clone https://github.com/Hardik-Sankhla/CNN-Plastic-Waste-Classification  
cd CNN-Plastic-Waste-Classification  
Install the required dependencies:

bash
Copy
pip install -r requirements.txt  
Run the training script (details to be added after completion):

bash
Copy
python train.py  
For inference, use the following command (details to be added after completion):

bash
Copy
python predict.py --image_path /path/to/image.jpg  
Technologies Used
Python
TensorFlow/Keras
OpenCV
NumPy
Pandas
Matplotlib
Future Scope
Expanding the dataset to include more categories of plastic waste.
Deploying the model as a web or mobile application for real-time classification.
Integrating the model with IoT-enabled waste management systems for automated waste sorting.
This repository features a waste management model using a Convolutional Neural Network (CNN) to classify waste into two categories: organic waste and recyclable waste. It utilizes TensorFlow for model construction and training, with training/testing data split. Pie charts are used for visualizing the distribution of waste types.

