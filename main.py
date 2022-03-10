import numpy as np

import pygame

import net
import track
import car
import vis

layer_structure = [7, 10, 10, 10, 10, 10, 10, 2]
genetic_controller = net.GeneticController(population_size=50, mutation_rate=0.002, layer_structure=layer_structure)
# genetic_controller = net.GeneticController.load('monza_pop50_gen124.json')

track = track.monza
time_delta = 0.1
max_time = 40
clock = pygame.time.Clock()


def simulate():
    yield None
    while True:
        vehicles = []
        for i, network in enumerate(genetic_controller.population):
            vehicle = car.Car(i, network, track)
            vehicles.append(vehicle)

        for t in np.arange(0, max_time, time_delta):
            # if genetic_controller.generation_index % 20 == 0:
            #     clock.tick(10)

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
            state = yield data_to_send

            while state.get('paused') and not state.get('step'):
                clock.tick(10)  # Wait for 1/10 second
                state = yield data_to_send

            speed = state.get('speed', np.inf)
            if not state.get('step'):
                clock.tick(speed / time_delta)

            if state.get('save'):
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
