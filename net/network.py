from typing import List, Tuple

import numpy as np


class Network:
    def __init__(self, layer_structure: List[int]):
        # Number of neurons in each layer
        self.layer_structure: List[int] = layer_structure

        # Network attributes
        self.layers: List[Layer] = []
        self.fitness = 0.0

        # Activation function to use [tanh, sigmoid, leaky_ReLu]
        self.activation_func = self.tanh

        # Add the layers to the network
        self.layers: List[Layer] = [Layer(self.layer_structure, i) for i in range(len(self.layer_structure))]

    def forward(self, input_data: Tuple[float]) -> np.array:
        """
        Makes one forward pass through the network and returns the
        output of the last layer.
        """
        # Check the input data is the right size
        assert len(input_data) == self.layer_structure[0], \
            f'input data length is not equal to the number of nodes in the first layer: ' \
            f'{len(input_data)} and {self.layer_structure[0]}'

        # Pass data through the network
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                layer.values = np.array(input_data)
            else:
                prev_layer = self.layers[layer_index - 1]
                layer.values = prev_layer.values.dot(layer.weights)
                layer.values = self.activation_func(layer.values + layer.bias)

        # Return last layer values
        return self.get_output()

    def get_output(self) -> np.array:
        last_layer = self.layers[-1]
        return last_layer.values

    def get_input(self) -> np.array:
        last_layer = self.layers[0]
        return last_layer.values

    def get_dna(self) -> List[float]:
        """Get DNA from the network"""
        dna = np.array([])
        for i in range(self.num_layers):
            layer = self.layers[i]
            dna = np.append(dna, layer.values)
            if i != 0:
                dna = np.append(dna, layer.weights)
                dna = np.append(dna, layer.bias)
        return list(dna)

    def get_sort_matrices(self) -> List:
        """Gets the order to which this layer should be sorted"""
        sort_matrices = []
        for layer in self.layers[1:-1]:
            to_sort = layer.bias.copy()
            layer_sort = np.arange(0, len(to_sort), dtype=int)

            # Bubble sort
            i = 0
            while i < len(to_sort):
                j = 0
                while j < len(to_sort) - 1:
                    if to_sort[j + 1] < to_sort[j]:
                        to_sort[j], to_sort[j+1] = to_sort[j+1], to_sort[j]
                        layer_sort[j], layer_sort[j+1] = layer_sort[j+1], layer_sort[j]
                    j += 1
                i += 1

            sort_matrix = np.zeros(shape=(len(to_sort), len(to_sort)))
            for i in range(len(to_sort)):
                sort_matrix[layer_sort[i]][i] = 1

            sort_matrices.append(sort_matrix)

        return sort_matrices

    def sort(self, sort_matrices):
        """Sorts the network based on the Neuron values"""
        for layer_index in range(len(self.layers))[1:-1]:
            sort_matrix = sort_matrices[layer_index - 1]
            self.layers[layer_index].bias = self.layers[layer_index].bias.dot(sort_matrix)
            self.layers[layer_index].weights = self.layers[layer_index].weights.dot(sort_matrix)
            self.layers[layer_index+1].weights = sort_matrix.transpose().dot(self.layers[layer_index+1].weights)

    def set_dna(self, dna: List[float]) -> None:
        """Set the network depending on the given DNA"""
        gene_index = 0
        for i in range(self.num_layers):
            layer = self.layers[i]

            layer.values = np.array(dna[gene_index: gene_index + layer.num_neurons])
            gene_index += layer.num_neurons

            if i != 0:
                weights_shape = layer.weights.shape
                layer.weights = np.array(dna[gene_index: gene_index + layer.weights.size])
                layer.weights = layer.weights.reshape(weights_shape)
                gene_index += layer.weights.size

                layer.bias = np.array(dna[gene_index: gene_index + layer.num_neurons])
                gene_index += layer.num_neurons

    @property
    def num_layers(self):
        return len(self.layers)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def leaky_ReLu(x):
        return np.maximum(0.1*x, x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)


class Layer:
    def __init__(self, layer_structure: List[int], layer_index: int):
        # Initialize our layer with the correct number of neurons
        num_neurons = layer_structure[layer_index]
        self.layer_index = layer_index

        self.values = np.zeros(num_neurons)
        self.weights = np.ndarray([])
        self.bias = np.array([])

        if layer_index == 0:
            self.bias = np.zeros(num_neurons)
        else:
            num_neurons_prev = layer_structure[layer_index - 1]  # Number of neurons in the previous layer
            self.weights = np.random.rand(num_neurons_prev, num_neurons) * 2 - 1
            self.bias = np.random.rand(num_neurons) * 2 - 1

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'bias':
            if value != []:
                for v in value:
                    if v < -1 or v > 1:
                        import traceback
                        print(f'{v=}')
                        traceback.print_stack()
                        exit(0)

    @property
    def neurons(self):
        return Neurons(self.values, self.weights, self.bias)

    @property
    def num_neurons(self) -> int:
        return len(self.bias)


class Neurons:
    """
    For backwards compatibility
    """
    def __init__(self, values, weights, bias):
        self.values = values
        self.weights = weights
        self.bias = bias

    def __getitem__(self, i):
        if self.weights.shape == ():
            return self.T(self.values[i], self.bias[i], [])
        return self.T(self.values[i], self.bias[i], self.weights[:, i])

    def __iter__(self):
        for i in range(len(self.values)):
            yield self.__getitem__(i)

    class T:
        def __init__(self, value, bias, dendrites):
            self.value = value
            self.bias = bias
            self.dendrites = dendrites
