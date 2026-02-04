import numpy as np

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01):
        """
        Initialize weights and biases for a 1-layer encoder and 1-layer decoder.

        Parameters:
            input_dim  -- dimensionality of input (e.g., 784 for MNIST)
            hidden_dim -- dimensionality of bottleneck (e.g., 16, 32, or 64)
            learning_rate -- gradient descent step size
        """
        # Initialize weights with small random values
        self.W_e = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_e = np.zeros((hidden_dim, 1))
        self.W_d = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_d = np.zeros((input_dim, 1))

        self.lr = learning_rate

    #helper funxtions: sigmoid and sigmoid_derivate
    def sigmoid(self, z):
        #sigmoid activation
        return 1 / (1+ np.exp(-z))

    def sigmoid_derivative(self, a):
        #derivate of sigmoid for given activation a
        return a * (1-a)

    def encoder(self, x):
        #x shape: (input_dim, batch_size)
        self.z_in = np.dot(self.W_e, x) + self.b_e
        self.z_out = self.sigmoid(self.z_in)
        return self.z_out

    def decoder(self, z):
        #z shape: (hidden_dim, batch_size)
        self.x_hat_in = np.dot(self.W_d, z) + self.b_d
        self.x_hat_out = self.sigmoid(self.x_hat_in)
        return self.x_hat_out

    def compute_loss(self, x, x_hat):
        #MSE
        m = x.shape[1] 
        loss = (1/m) * np.sum(np.square(x-x_hat))
        return loss

    def backward(self, x, z, x_hat):
        #compute gradients using backpropagation (x: input, x_hat: reconstructed data z:latent representation
        m = x.shape[1]
        
        #Decoder:
        #Gradient of MSE(wrt output) : (2 /m) * (x_hat - x)
        #gradient of sigmoid: x_hat * (1-x_hat)
        #Delta term: dL/d(decoder_activation_input) = (x_hat-x) * sigmoid_derivative

        dz_decoder = (x_hat - x) * self.sigmoid_derivative(x_hat)

        #gradient for decoder bias
        self.db_d = (1/m) * np.sum(dz_decoder, axis=1, keepdims=True)
        #gradient foe decoder weights
        self.dW_d = (1/m) * np.dot(dz_decoder, z.T)

        #hidden layers gradient(encoder)
        dz_encoder_part = np.dot(self.W_d.T, dz_decoder)
        dz_encoder = dz_encoder_part * self.sigmoid_derivative(z)
        
        #gradient for encoder bias
        self.db_e = (1/m) * np.sum(dz_encoder, axis=1, keepdims=True)
        #gradient for encoder weights
        self.dW_e = (1/m) * np.dot(dz_encoder, x.T)

    def step(self, grads=None):
        self.W_e -= self.lr * self.dW_e
        self.b_e -= self.lr * self.db_e
        self.W_d -= self.lr * self.dW_d
        self.b_d -= self.lr * self.db_d

    def train(self, X, epochs=20, batch_size=128):
        # Transpose X to shape (input_dim, num_samples)
        X = X.T 
        input_dim, num_samples = X.shape
        
        for epoch in range(epochs):
            # Shuffle data
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[:, permutation]
            
            epoch_loss = 0
            num_batches = int(np.ceil(num_samples / batch_size))

            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, num_samples)
                x_batch = X_shuffled[:, start:end]

                # Forward pass
                z = self.encoder(x_batch)
                x_hat = self.decoder(z)

                # Compute Loss
                loss = self.compute_loss(x_batch, x_hat)
                epoch_loss += loss

                # Backward pass
                self.backward(x_batch, z, x_hat)

                # Update parameters
                self.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}")
