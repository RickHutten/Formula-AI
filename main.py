import time
import numpy as np

import net
import track
import car
import vis


genetic_controller = net.GeneticController(population_size=25, mutation_rate=0.01)
track = track.monza

time_delta = 0.1


def simulate():
    for _ in range(100):
        vehicles = []
        for i, network in enumerate(genetic_controller.population):
            vehicle = car.Car(i, network, track)
            vehicles.append(vehicle)

        for t in np.arange(0, 120, time_delta):
            for vehicle in vehicles:
                vehicle.update(time_delta)
            if all([vehicle.has_crashed for vehicle in vehicles]):
                break
            if t > 5:
                for v in (veh for veh in vehicles if not veh.has_crashed):
                    if v.distance_travelled < 10:
                        v.has_crashed = True

            yield {'vehicles': vehicles, 'time': t, 'generation': genetic_controller.generation_index}

        for vehicle in vehicles:
            vehicle.calculate_fitness()

        genetic_controller.next_generation()
        time.sleep(1)
    return


vis.Window(track, simulate())
