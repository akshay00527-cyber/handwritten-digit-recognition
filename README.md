

\# ✍️ Handwritten Digit Recognition Using Deep Learning



<p align="center">

&#x20; <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">

&#x20; <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow">

&#x20; <img src="https://img.shields.io/badge/Streamlit-Web%20Application-red?logo=streamlit">

&#x20; <img src="https://img.shields.io/badge/Dataset-MNIST-green">

</p>



\## 📌 Project Overview



Handwritten Digit Recognition is a deep learning-based application that identifies handwritten digits from 0 to 9.



The application allows users to draw digits on a digital canvas using a mouse. The trained neural network processes the drawing and predicts the corresponding digit along with its confidence score.



This project demonstrates the application of deep learning, image preprocessing, neural network training, and interactive web application development.



\## 🎯 Problem Statement



Handwritten digits can vary significantly in shape, size, orientation, and writing style. Recognizing these digits accurately is an important task in computer vision and pattern recognition.



The objective of this project is to develop a deep learning application capable of recognizing handwritten digits and providing predictions through an interactive interface.



\## 🎯 Objectives



\- Develop a handwritten digit recognition system.

\- Train a neural network using the MNIST dataset.

\- Perform image preprocessing before prediction.

\- Build an interactive application using Streamlit.

\- Display predicted digits and confidence scores.

\- Visualize prediction probabilities.



\## ✨ Features



\- Interactive drawing canvas.

\- Recognition of digits from 0 to 9.

\- Real-time digit prediction.

\- Prediction confidence display.

\- Probability visualization for all ten digits.

\- Simple and user-friendly interface.

\- Trained deep learning model.



\## 🛠️ Technologies Used



| Technology | Purpose |

|-----------|---------|

| Python | Programming language |

| TensorFlow | Deep learning framework |

| Keras | Neural network implementation |

| Streamlit | Web application interface |

| NumPy | Numerical computations |

| Pandas | Data handling |

| Matplotlib | Probability visualization |

| MNIST | Handwritten digit dataset |



\## 📚 Dataset



\### MNIST Dataset



The project uses the MNIST handwritten digit dataset.



\- Total images: 70,000

\- Training images: 60,000

\- Testing images: 10,000

\- Image dimensions: 28 × 28 pixels

\- Image type: Grayscale

\- Number of classes: 10

\- Classes: Digits 0–9



Each image contains a handwritten digit represented by pixel intensity values.



\## 🧠 Deep Learning Model



The project uses a fully connected Artificial Neural Network (ANN).



\### Model Architecture



| Layer | Description |

|-------|-------------|

| Input | 28 × 28 grayscale image |

| Flatten | Converts image into 784 features |

| Dense | 128 neurons, ReLU activation |

| Output | 10 neurons, Softmax activation |



\### Model Training



\- Dataset: MNIST

\- Optimizer: Adam

\- Loss Function: Sparse Categorical Crossentropy

\- Evaluation Metric: Accuracy

\- Training Epochs: 5



The trained model is saved as:



`digit\_model.keras`



\## 🔄 Project Workflow



1\. Load the MNIST dataset.

2\. Normalize pixel values.

3\. Train the neural network.

4\. Save the trained model.

5\. Launch the Streamlit application.

6\. Draw a digit on the canvas.

7\. Preprocess the drawing.

8\. Pass the image to the trained model.

9\. Predict the digit.

10\. Display prediction and confidence.



\## 📂 Project Structure



```text

handwritten-digit-recognition/

│

├── app.py

├── train\_model.py

├── digit\_model.keras

├── requirements.txt

├── README.md

└── .gitignore

```



\## ⚙️ Installation and Setup



\### 1. Clone the Repository



```bash

git clone https://github.com/akshay00527-cyber/handwritten-digit-recognition.git

```



\### 2. Navigate to Project Directory



```bash

cd handwritten-digit-recognition

```



\### 3. Create Virtual Environment



```bash

python -m venv .venv

```



\### 4. Activate Virtual Environment



Windows:



```bash

.venv\\Scripts\\activate

```



\### 5. Install Dependencies



```bash

pip install -r requirements.txt

```



\## 🚀 Run the Application



Start the Streamlit application:



```bash

streamlit run app.py

```



The application will open in your browser.



\## 🖥️ How to Use



1\. Open the application.

2\. Draw a digit between 0 and 9 using the mouse.

3\. Click the prediction button.

4\. View the predicted digit.

5\. Check the confidence score.

6\. Analyze the probability chart.



\## 📊 Model Performance



The model achieved approximately 97.78% accuracy during evaluation.



Actual performance may vary depending on training configuration and preprocessing.



\## 🔮 Future Enhancements



\- Implement a Convolutional Neural Network (CNN).

\- Improve recognition accuracy.

\- Add image upload functionality.

\- Deploy the application online.

\- Support recognition of multiple handwritten digits.

\- Improve drawing canvas and user interface.



\## 👨‍💻 Author



\*\*Akshay Singh\*\*



GitHub: \[akshay00527-cyber](https://github.com/akshay00527-cyber)



\## ⭐ Acknowledgements



\- MNIST Dataset

\- TensorFlow

\- Keras

\- Streamlit

\- Python Community



\---



If you find this project interesting, consider giving it a ⭐ on GitHub!



