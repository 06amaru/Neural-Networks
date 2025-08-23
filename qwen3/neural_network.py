import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, activation='sigmoid'):
        """
        Initialize the neural network
        
        Parameters:
        layers: list of integers representing the number of neurons in each layer
        learning_rate: learning rate for gradient descent
        activation: activation function ('sigmoid', 'tanh', 'relu')
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights = []
        self.biases = []
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Initialize weights and biases for all layers"""
        np.random.seed(42)
        for i in range(len(self.layers) - 1):
            # Xavier initialization for better convergence
            weight = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0 / self.layers[i])
            bias = np.zeros((1, self.layers[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def activation_function(self, x, derivative=False):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            if derivative:
                return x * (1 - x)
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        
        elif self.activation == 'tanh':
            if derivative:
                return 1 - x**2
            return np.tanh(x)
        
        elif self.activation == 'relu':
            if derivative:
                return (x > 0).astype(float)
            return np.maximum(0, x)
    
    def forward_propagation(self, X):
        """Forward propagation through the network"""
        self.activations = [X]  # Store activations for each layer
        self.z_values = []      # Store z values for each layer
        
        current_input = X
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation function
            if i == len(self.weights) - 1:  # Output layer - use sigmoid for binary classification
                activation = self.activation_function(z, False) if self.activation == 'sigmoid' else 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            else:  # Hidden layers
                activation = self.activation_function(z, False)
            
            self.activations.append(activation)
            current_input = activation
        
        return self.activations[-1]
    
    def compute_cost(self, y_true, y_pred):
        """Compute binary cross-entropy cost"""
        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return cost
    
    def backward_propagation(self, X, y):
        """Backward propagation to compute gradients"""
        m = X.shape[0]
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        dz = self.activations[-1] - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            dW[i] = np.dot(self.activations[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m
            
            # Compute error for previous layer (except for input layer)
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.activation_function(self.activations[i], derivative=True)
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs=1000, verbose=True):
        """Train the neural network"""
        costs = []
        
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            costs.append(cost)
            
            # Backward propagation
            dW, db = self.backward_propagation(X, y)
            
            # Update parameters
            self.update_parameters(dW, db)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")
        
        return costs
    
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward_propagation(X)
        return (y_pred > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.forward_propagation(X)

# Generate sample data
def generate_data():
    """Generate sample classification data"""
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                              n_redundant=10, n_clusters_per_class=1, random_state=42)
    return X, y.reshape(-1, 1)

# Main execution
if __name__ == "__main__":
    # Generate and prepare data
    X, y = generate_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define network architecture: [input_size, hidden_layer1, hidden_layer2, ..., output_size]
    layers = [X_train.shape[1], 16, 8, 1]  # 20 input features -> 16 -> 8 -> 1 output
    
    # Create and train neural network
    nn = NeuralNetwork(layers=layers, learning_rate=0.01, activation='relu')
    
    print("Training Neural Network...")
    print(f"Network Architecture: {layers}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print("-" * 50)
    
    # Train the network
    costs = nn.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Make predictions
    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = np.mean(train_predictions == y_train) * 100
    test_accuracy = np.mean(test_predictions == y_test) * 100
    
    print("-" * 50)
    print("Training Results:")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Plot training cost
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(costs)
    plt.title('Training Cost Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    
    # Plot decision boundary for 2D visualization (using first 2 features)
    plt.subplot(1, 2, 2)
    # Generate a mesh for decision boundary
    h = 0.02
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create a simple 2-layer network for visualization
    X_2d = X_test[:, :2]  # Use only first 2 features
    layers_2d = [2, 8, 1]
    nn_2d = NeuralNetwork(layers_2d, learning_rate=0.1, activation='relu')
    nn_2d.train(X_2d, y_test, epochs=1000, verbose=False)
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn_2d.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_test.ravel(), cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title('Decision Boundary (First 2 Features)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    # Display final weights and biases
    print("\nFinal Network Parameters:")
    for i, (w, b) in enumerate(zip(nn.weights, nn.biases)):
        print(f"Layer {i+1} -> {i+2}:")
        print(f"  Weights shape: {w.shape}")
        print(f"  Biases shape: {b.shape}")