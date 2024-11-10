#code for connection signal brain with human
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load real EEG data
data = pd.read_csv('eeg_data.csv', delim_whitespace=True)

# Drop the 'Time' column if it's not needed for analysis
data = data.drop(columns=['Time'])

# Convert the DataFrame to numeric values (if necessary)
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values if any
data.dropna(inplace=True)

# Preprocessing
brain_signal = data.values
scaler = StandardScaler()
brain_signal = scaler.fit_transform(brain_signal)  # Standardize the data

# Apply ICA
ica = FastICA(n_components=5, random_state=42)
ica_components = ica.fit_transform(brain_signal)

# Load actual labels (assuming they are in a separate column or file)
# labels = pd.read_csv('labels.csv').values.flatten()  # Example for loading labels
# For demonstration, we will use random labels (replace this with actual labels)
labels = np.random.randint(0, 5, size=(ica_components.shape[0],))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(ica_components, labels, test_size=0.2, random_state=42)

# Define the neural network class
class SimpleNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.leaky_relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.leaky_relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.softmax(self.z3)

    def leaky_relu(self, z, alpha=0.01):
        return np.where(z > 0, z, z * alpha)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        return np.sum(log_likelihood) / m

    def backward(self, x, y_true, y_pred):
        m = y_true.shape[0]
        delta3 = y_pred
        delta3[range(m), y_true] -= 1
        delta3 /= m

        dW3 = np.dot(self.a2.T, delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = np.dot(delta3, self .W3.T) * (self.a2 > 0)
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = np.dot(delta2, self.W2.T) * (self.a1 > 0)
        dW1 = np.dot(x.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    def update_weights(self, dW1, db1, dW2, db2, dW3, db3, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            dW1, db1, dW2, db2, dW3, db3 = self.backward(X, y, y_pred)
            self.update_weights(dW1, db1, dW2, db2, dW3, db3, learning_rate)
            logging.info(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Train the neural network
nn = SimpleNN(input_size=5, hidden_size1=10, hidden_size2=10, output_size=5)
nn.train(X_train, y_train, learning_rate=0.01, epochs=100)

# Evaluate the model
y_pred = nn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
logging.info(f'Test Accuracy: {accuracy:.4f}')

# Print classification report
logging.info(classification_report(y_test, y_pred))

# Print confusion matrix
logging.info(confusion_matrix(y_test, y_pred))

# Visualize the raw EEG signals
plt.figure(figsize=(12, 6))
for i in range(data.shape[1]):
    plt.plot(data.index, data.iloc[:, i], label=f'Channel {i + 1}')
plt.title('Raw EEG Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Visualize true vs predicted labels
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='True Labels', alpha=0.5)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Labels', alpha=0.5)
plt.title('True vs Predicted Labels')
plt.xlabel('Sample Index')
plt.ylabel('Class Label')
plt.legend()
plt.show()