# self_driving_simluation
This project aims to create a simulated environment for testing and validating the performance of self-driving cars. The simulation is built using PyTorch, a popular open-source machine learning library, and other Python libraries such as NumPy, Pandas, and Matplotlib.

The core of the project is a custom-built Convolutional Neural Network (CNN) model that is trained to predict steering angles based on input images from the car's center camera. The model architecture includes multiple convolutional layers for feature extraction and fully connected layers for regression prediction. 

The training data consists of images from a car's center camera along with corresponding steering angles. The images are preprocessed and augmented using various techniques such as random horizontal flipping, random rotation, and color jittering to enhance the model's ability to generalize.

The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer. The training process includes a mechanism for saving the model's parameters for future use.

This project represents a significant step towards understanding and developing self-driving car technologies. It provides a solid foundation for further exploration and improvement, such as integrating more sensors data (like LIDAR or RADAR), testing on different road conditions, or optimizing the neural network architecture for better performance.

Please note that this is a simulation project and the results obtained are not meant for real-world deployment without further testing and validation under various conditions. Safety is paramount when it comes to self-driving technologies. 
