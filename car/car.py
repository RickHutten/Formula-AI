from typing import Tuple, List

import numpy as np

from net import Network
from track import Track


class Car:
    vehicle_sensor_angles = [90, 30, 15, 10, 5, 0, -5, -10, -15, -30, -90]

    def __init__(self, index: int, network: Network, track: Track):
        self.car_index = index
        self.network: Network = network
        self.track = track

        self.distance_finished = 0
        self.distance_travelled = 0
        self.total_time = 0

        # Vehicle state
        self.position, self.heading = track.get_spawn_point_and_heading()  # (y, x)
        self.__local_velocity = np.array([0.0, 0.0])  # (y, x) +x is forwards
        self.__local_acceleration = np.array([0.0, 0.0])  # (y, x) +x is forwards
        self.has_crashed = False
        self.best_angle = 0

        # Vehicle parameters
        self.max_steering_angle = np.radians(40)
        self.max_acceleration = 2 * 9.81
        self.max_deceleration = 5.5 * 9.81
        self.Cdrag = 1.25
        self.Crr = 30 * self.Cdrag  # rolling resistance
        self.mass = 740
        self.wheel_base = 3.698  # Mercedes wheelbase
        self.num_standstill = 0

        # Unused for now
        self.tyre_friction = -0.0150

    def update(self, time_delta: float):
        """Perform a simulation step"""
        if self.has_crashed:
            return

        # Get input to the network to determine next step
        network_input = self.generate_network_input()

        # Propagate through the net
        self.network.forward(network_input)

        # Get the output
        steering_angle, gas_pedal = self.network.get_output()

        # Calculate the steering angle and new heading
        steering_angle *= self.max_steering_angle * np.exp(-(self.speed/40)**2)

        turn_radius = self.wheel_base / (2 * np.sin(0.5 * steering_angle))
        heading_change = (self.speed * time_delta) / turn_radius
        self.heading += heading_change

        # Keep heading inside [0, 2pi]
        if self.heading >= 2 * np.pi:
            self.heading -= 2 * np.pi
        elif self.heading < 0:
            self.heading += 2 * np.pi

        # Traction
        if gas_pedal >= 0:
            self.__local_acceleration = np.array([0, self.max_acceleration * gas_pedal])
        else:
            self.__local_acceleration = np.array([0, self.max_deceleration * gas_pedal])

        # Drag and rolling resistance
        self.__local_acceleration -= (self.Cdrag * self.__local_velocity * self.speed) / self.mass
        self.__local_acceleration -= (self.Crr * self.__local_velocity) / self.mass

        # Update position
        self.__local_velocity += self.__local_acceleration * time_delta
        self.__local_velocity[1] = max(0, self.__local_velocity[1])  # Can't go backwards

        self.position += self.forward_unit_vector * self.__local_velocity[1] * time_delta
        self.position += self.left_unit_vector * self.__local_velocity[0] * time_delta

        # Update total distance travelled
        self.distance_travelled += self.speed * time_delta
        self.total_time += time_delta

        # Check if we crashed
        if self.track.is_outside(self.position):
            self.has_crashed = True

        # Don't continue simulation if the car is standing still
        if self.total_time > 5:
            avg_speed = self.distance_travelled / self.total_time
            if avg_speed < 10:
                self.has_crashed = True
            if self.speed < 1:
                self.num_standstill += 1
                if self.num_standstill > 30:
                    self.has_crashed = True
        if self.total_time == 5:
            progress = self.track.get_lap_progress(self.position)
            if progress > 0.5:
                progress -= 1
            progress *= self.track.track_length_m
            if progress < 20:
                self.has_crashed = True

    def calculate_fitness(self):
        score = self.distance_travelled

        # Calculate wrong heading penalty
        index = self.track.get_closest_green_line_index(self.position)
        behind, ahead = self.track.green_line[index - 2], self.track.green_line[index + 2]
        dy, dx = behind[0] - ahead[0], ahead[1] - behind[1]
        angle = -np.arctan2(dy, dx)
        heading = self.heading
        if heading > np.pi:
            heading -= 2 * np.pi
        diff = angle - heading
        diff -= 2 * np.pi if diff > np.pi else 0
        diff += 2 * np.pi if diff < -np.pi else 0
        diff = abs(diff)
        score -= diff * 10

        # Calculate hard crash penalty
        if self.has_crashed:
            penalty = self.speed ** 2 / 150
            score -= penalty

        self.network.fitness = score

        return

    @property
    def forward_unit_vector(self):
        return np.array([np.sin(self.heading), np.cos(self.heading)])

    @property
    def left_unit_vector(self):
        return np.array([-np.cos(self.heading), np.sin(self.heading)])

    def unit_vector_rotated(self, rot):
        return np.array([np.sin(self.heading - rot), np.cos(self.heading - rot)])

    @property
    def speed(self) -> float:
        """Returns the magnitude of the velocity"""
        return np.linalg.norm(self.__local_velocity)

    @staticmethod
    def rot_matrix(degrees: float):
        theta = np.radians(degrees)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, s), (-s, c)))

    def generate_network_input(self) -> Tuple[float, ...]:
        sensor_input = self.get_sensor_input()
        left = sensor_input[0]
        right = sensor_input[-1]

        angles = self.vehicle_sensor_angles[1:-1]
        values = sensor_input[1:-1]
        avg = sum(values) / len(values)

        tot = 0
        for val, ang in zip(values, angles):
            tot += ang * val
        self.best_angle = tot/sum(values) / 30
        max_dist = max(values)
        return left/10, right/10, self.best_angle, max_dist/250, self.speed/80

    def get_free_distance(self, unit_vector) -> float:
        max_dist = 250
        dist_coarse = max_dist
        for dist in range(10, max_dist+1, 10):
            pos = self.position + unit_vector * dist
            if self.track.is_outside(pos):
                dist_coarse = dist
                break
            elif dist == max_dist:
                # If the last is also free, no need to finetune
                return max_dist

        for dist in range(dist_coarse - 10, dist_coarse+1, 1):
            pos = self.position + unit_vector * dist
            if self.track.is_outside(pos):
                return dist
        return max_dist

    def get_sensor_input(self) -> List[float]:
        return [self.get_free_distance(self.unit_vector_rotated(np.deg2rad(i))) for i in self.vehicle_sensor_angles]
