from typing import List, Optional
from random import randint, random, uniform, choices

import numpy as np

from .network import Network

DNA = List[float]


class GeneticController:
    def __init__(self, population_size: int, mutation_rate: float):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.average_fitness = 0.0
        self.population_fitness = 0.0
        self.best_fitness = 0.0
        self.generation_index = 0

        # Create a population
        layer_structure = [6, 10, 5, 2]
        self.population = [Network(layer_structure) for _ in range(population_size)]

    def __make_baby(self, mother: Network, father: Network) -> Network:
        baby: Chromosome = Chromosome(self.mutation_rate, mother) * Chromosome(self.mutation_rate, father)
        return baby.network

    def __parent_choice_weight(self):
        return [10 ** -x for x in np.linspace(0, 1, self.population_size)]

    def next_generation(self):
        print(f"Generation {self.generation_index}:")

        # Calculate total fitness and fitness ratio of the individuals
        self.population_fitness = sum((entity.fitness for entity in self.population))
        self.average_fitness = self.population_fitness / len(self.population)

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_fitness = self.population[0].fitness

        print(f"\tAverage fitness: {self.average_fitness}")
        print(f"\tBest fitness: {[veh.fitness for veh in self.population][:3]}")

        # Save the best 3 so far
        new_population: List[Network] = self.population[:3]

        while len(new_population) < self.population_size:
            mother, father = choices(list(range(len(self.population))), weights=self.__parent_choice_weight(), k=2)
            if mother == father:
                # Try again to pick a separate mother and father
                continue

            baby = self.__make_baby(self.population[mother], self.population[father])
            new_population.append(baby)

        # Babies grow old
        self.population = new_population
        self.generation_index += 1


class Chromosome:
    def __init__(self, mutation_rate: float, network: Network, dna: Optional[DNA] = None):
        self.dna = dna
        self.mutation_rate: float = mutation_rate

        # Either network or dna should be defined
        assert not (network is dna is None)

        # Create dna if not supplied
        if self.dna is None:
            self.network: Network = network
            self.__create_dna()
        else:
            # If dna is given, update the network so the two are the same
            self.network = Network(network.layer_structure)
            self.__update_network()

    @staticmethod
    def combine(mother: 'Chromosome', father: 'Chromosome') -> 'Chromosome':
        """Create a new Chromosome instance from a mother and father chromosome"""
        new_dna = mother.__cross_over(father)
        baby = Chromosome(mother.mutation_rate, mother.network, dna=new_dna)
        baby.mutate()
        return baby

    def __create_dna(self):
        """Create dna from network"""
        self.dna: DNA = []
        for i in range(self.network.num_layers):
            layer = self.network.layers[i]
            for j in range(layer.num_neurons):
                neuron = layer.neurons[j]

                # Add the bias to the dna
                self.dna.append(neuron.bias)

                # Add the dendrite weights
                self.dna += neuron.dendrites

    def __update_network(self):
        """Update the network weights from the dna"""
        gene_index = 0
        for i in range(self.network.num_layers):
            layer = self.network.layers[i]
            for j in range(layer.num_neurons):
                neuron = layer.neurons[j]

                # Set neuron bias
                neuron.bias = self.dna[gene_index]
                gene_index += 1

                # Set neuron dendrites
                neuron.dendrites = self.dna[gene_index: gene_index + neuron.num_dentrites]
                gene_index += neuron.num_dentrites

    def __cross_over(self, other: 'Chromosome') -> DNA:
        """Thing of beauty"""
        return [i if randint(0, 1) else j for i, j in zip(self.dna, other.dna)]

    def mutate(self):
        """Mutates some genes and rebuilds the network"""
        for i in range(len(self.dna)):
            if random() < self.mutation_rate:
                self.dna[i] = uniform(-1.0, 1.0)
        # Rebuild network
        self.__update_network()

    def __mul__(self, other: 'Chromosome') -> 'Chromosome':
        return Chromosome.combine(self, other)

    def __str__(self):
        return f"DNA:{self.dna}"

    def __repr__(self):
        return self.__str__()