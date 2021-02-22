from typing import Tuple, Any

import numpy as np

from net import Network
from track import Track


class Car:
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

        # Vehicle parameters
        self.max_steering_angle = np.radians(30)
        self.max_acceleration = 2 * 9.81
        self.max_deceleration = 5.5 * 9.81
        self.Cdrag = 1.2
        self.Crr = 30 * 1.2  # rolling resistance

        # Unused for now
        self.mass = 740
        self.tyre_friction = -0.0150
        self.wheel_base = 3.698  # Mercedes wheelbase

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
        steering_angle *= self.max_steering_angle * np.exp(-self.speed/25)
        self.heading += steering_angle * time_delta * self.speed

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
        # self.acceleration -= self.Cdrag * self.heading_vector * self.speed ** 2
        # self.acceleration -= self.Crr * self.heading_vector * self.speed

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
            # print(f"Car crashed at {self.total_time}")
            self.has_crashed = True

    def calculate_fitness(self):
        self.distance_finished = self.track.get_lap_progress(self.position)  # [0, 1]
        lap_distance = self.distance_finished * self.track.track_length_m
        if self.distance_travelled < 0.5 * lap_distance and self.distance_finished > 0.4:
            # We went backwards
            self.distance_finished *= -1

        avg_speed = (self.distance_travelled / self.total_time)
        avg_speed_ratio = avg_speed / 100  # 100 m/s is about the max speed

        distance_weight = 10
        speed_weight = 2

        self.network.fitness = distance_weight * self.distance_finished + speed_weight * avg_speed_ratio

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

    def generate_network_input(self) -> Tuple[Any, float]:
        return *self.get_sensor_input(), self.speed

    def get_free_distance(self, unit_vector):
        max_dist = 50
        for dist in range(1, max_dist, 1):
            pos = self.position + unit_vector * dist
            if self.track.is_outside(pos):
                return dist
        return max_dist

    def get_sensor_input(self) -> Tuple[int, int, int, int, int]:
        left = self.get_free_distance(self.left_unit_vector)
        left_forward = self.get_free_distance(self.unit_vector_rotated(np.deg2rad(-20)))
        forward = self.get_free_distance(self.forward_unit_vector)
        right_forward = self.get_free_distance(self.unit_vector_rotated(np.deg2rad(20)))
        right = self.get_free_distance(-self.left_unit_vector)

        return left, left_forward, forward, right_forward, right


