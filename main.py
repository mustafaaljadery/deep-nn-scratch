import math
import numpy as np 

class NeuralNetwork:
    def __init__(self, sizes, activation="relu"):
      self.sizes = sizes

      self.activation = self.relu if activation == "relu" else self.sigmoid
      
      # weights
      self.params = self.begin()
      # cache activations
      self.cache = {}

    def begin(self):
      input_layer = self.sizes[0]
      hidden_layer = self.sizes[1]
      output_layer = self.sizes[2]

      return {
        "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1. / hidden_layer),
        "B1": np.random.randn(hidden_layer, 1) * np.sqrt(1. / hidden_layer),
        "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1. / output_layer),
        "B2": np.random.randn(output_layer, 1) * np.sqrt(1. / output_layer)
      }
  
    def momentum_optimizer(self):
      return {
        "W1": np.zeros(self.params["W1"].shape),
        "B1": np.zeros(self.params["B1"].shape),
        "W2": np.zeros(self.params["W2"].shape),
        "B2": np.zeros(self.params["B2"].shape),
      }

    def sigmoid(self, x, derivative=False):
      if derivative: 
        return (np.exp(-x))/((np.exp(-x)+1)**2)
      return 1/(1 + np.exp(-1))

    def relu(self, x, derivative=False):
      if derivative: 
        x = np.where(x < 0, 0, x)
        x = np.where(x >= 0 , 1, x)
        return x
      return np.maximum(0, x)
    
    def softmax(self, x):
      return np.exp(x-x.max()) / np.sum(np.exp(x-x.max()), axis=0)
    
    def cross_entropy_loss(self,y,output):
      return -(1./y.shape[0]) * np.sum(np.multiply(y.T, np.log(output)))
    
    def forward(self, x):
      self.cache["X"] = x
      self.cache["Z1"] = np.matmul(self.params["W1"], self.cache["X"].T) + self.params["b1"]
      self.cache["A1"] = self.activation(self.cache["Z1"])
      self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
      self.cache["A2"] = self.softmax(self.cache["Z2"])
      return self.cache["A2"]
    
    def backward(self, y, output):
      batch_size = y.shape[0]

      dw2 = (1./batch_size) * np.matmal(output-y.T, self.cache["A1"].T)
      db2 = (1./batch_size) * np.sum(output-y.T, axis=1, keepdims=True)

      d = np.matmal(self.params["W2"].T, output-y.T) * self.activation(self.cache["Z1"], derivative=True)
      dw1 = (1./batch_size) * np.matmul(d, self.cache["X"])
      db1 = (1./batch_size) * np.sum(d, axis=1, keepdims=True)

      return {
        "W1": dw1,
        "B1": db1,
        "W2": dw2,
        "B2": db2
      }
    
    def accurancy(self, y, out):
      return np.mean(np.argmax(y, axis=0) == np.argmax(out, axis=0))
      

nn = NeuralNetwork()