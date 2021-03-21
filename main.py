import numpy as np

import pygame

import net
import track
import car
import vis


# genetic_controller = net.GeneticController(population_size=40, mutation_rate=0.005, layer_structure=[5, 10, 8, 6, 4, 2])
genetic_controller = net.GeneticController.load('monza_pop40_gen64.json')
# genetic_controller.mutation_rate = 0.003
track = track.monza

time_delta = 0.1
clock = pygame.time.Clock()


def update_batch(vehicle: car.Car):
    vehicle.update(time_delta)


def simulate():
    while True:
        vehicles = []
        for i, network in enumerate(genetic_controller.population):
            vehicle = car.Car(i, network, track)
            vehicles.append(vehicle)

        for t in np.arange(0, 30, time_delta):
            # clock.tick(10)
            for vehicle in vehicles:
                vehicle.update(time_delta)

            number_of_active_vehicles = sum((not vehicle.has_crashed for vehicle in vehicles))
            if number_of_active_vehicles == 0:
                break

            # We have some new data, return it to the visualization
            data_to_send = {
                'vehicles': vehicles,
                'time': t,
                'generation': genetic_controller.generation_index
            }
            save_popultion = yield data_to_send
            if save_popultion:
                filename = f'{track.name}' \
                           f'_pop{genetic_controller.population_size}' \
                           f'_gen{genetic_controller.generation_index}'
                genetic_controller.save(filename)

        best_distance = 0
        for vehicle in vehicles:
            vehicle.calculate_fitness()
            if vehicle.distance_travelled > best_distance:
                best_distance = vehicle.distance_travelled
        genetic_controller.next_generation()
        print(f"\tBest distance: {best_distance}")



window = vis.Window(track, simulate())
window.loop()
