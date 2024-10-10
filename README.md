Objective:
This project aimed to develop a machine learning algorithm capable of classifying images of Plasmodium knowlesi parasites based on their developmental stages. The use of Convolutional Neural Networks (CNNs) allowed for the automated identification of parasite stages, which is crucial for understanding the parasite's lifecycle and aiding in disease monitoring.

Approach:

Data Preparation: Images of Plasmodium knowlesi were gathered and preprocessed using TensorFlow’s dataset utilities. The dataset was split into training (80%) and validation (20%) sets, with images resized to a fixed dimension of 128x128 pixels for consistency across the dataset.

Model Architecture:

A CNN was designed with three convolutional layers, each followed by max-pooling layers to progressively capture image features such as edges and textures.
A final fully connected layer provided the classification into distinct stages of the parasite.
The model was trained using Adam optimizer and Sparse Categorical Crossentropy as the loss function to handle multi-class classification effectively.
Training: The model was trained for 100 epochs with monitoring of training and validation performance to ensure proper learning and to avoid overfitting. The results were visualized using accuracy and loss plots over time.

Testing and Evaluation:
A function was implemented to test individual images or batches of test images, predicting the stage of the parasite with a confidence score. The trained model was able to provide accurate classifications, which can significantly reduce manual labor in identifying parasite stages in research or diagnostic settings.
