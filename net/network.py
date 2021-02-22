import math
from typing import List, Tuple, Optional
from random import uniform


class Network:
    def __init__(self, layer_structure: List[int]):
        # Number of neurons in each layer
        self.layer_structure: List[int] = layer_structure

        # Network attributes
        self.layers: List[Layer] = []
        self.fitness = 0.0

        # Add the layers to the network
        self.layers: List[Layer] = [Layer(self.layer_structure, i) for i in range(len(self.layer_structure))]

    def forward(self, input_data: Tuple[float]) -> Tuple[float]:
        """
        Makes one forward pass through the network and returns the
        output of the last layer.
        """
        # Check the input data is the right size
        assert len(input_data) == self.layer_structure[0]

        # Pass data through the network
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer.neurons):

                if layer_index == 0:
                    # Set input data to the first neurons
                    neuron.value = input_data[neuron_index]
                else:
                    neuron.value = 0
                    prev_layer = self.layers[layer_index - 1]
                    for neuron_id in range(prev_layer.num_neurons):
                        neuron.value += prev_layer.neurons[neuron_id].value * neuron.dendrites[neuron_id]

                    # Call the activation function
                    neuron.value = self.tanh(neuron.value + neuron.bias)

        # Return last layer values
        return self.get_output()

    def get_output(self):
        last_layer = self.layers[-1]
        return [neuron.value for neuron in last_layer.neurons]

    @property
    def num_layers(self):
        return len(self.layers)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def tanh(x):
        return math.tanh(x)


class Layer:
    def __init__(self, layer_structure: List[int], layer_index: int):
        # Initialize our layer with the correct number of neurons
        num_neurons = layer_structure[layer_index]
        self.neurons: List[Neuron] = [Neuron() for _ in range(num_neurons)]
        self.layer_index = layer_index

        # Create dendrites to the next layer
        for neuron in self.neurons:
            if layer_index == 0:
                # If we are the first layer set our bias to 0
                neuron.bias = 0
            else:
                # Create a dendrite from every neuron in the previous layer
                neuron.init_dendrites(layer_structure[layer_index - 1])

    def get_weights(self, network: Network):
        next_layer = network.layers[self.layer_index + 1]
        weights = [[0] * next_layer.num_neurons for _ in range(self.num_neurons)]
        for i in range(self.num_neurons):
            for j in range(next_layer.num_neurons):
                weights[i][j] = next_layer.neurons[j].dendrites[i]
        return weights

    @property
    def num_neurons(self) -> int:
        return len(self.neurons)


class Neuron:
    def __init__(self, bias: Optional[float] = None):
        # Neuron inputs
        self.dendrites: List[float] = []

        # Neuron attributes
        self.bias = bias if bias else uniform(-1.0, 1.0)
        self.delta = None
        self.value = None

    def init_dendrites(self, num_dendrites):
        self.dendrites = [uniform(-1.0, 1.0) for _ in range(num_dendrites)]

    @property
    def num_dentrites(self):
        return len(self.dendrites)
