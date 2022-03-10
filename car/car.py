from typing import Tuple, List, Optional

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

        # Vehicle parameters
        self.max_steering_angle = np.radians(30)
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

        self.propagate(time_delta, steering_angle, gas_pedal)

    def propagate(self, time_delta: float, steering_angle: float, gas_pedal: float):
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
        diff = abs(self.get_track_car_heading_difference())
        score -= diff * 10

        # Calculate hard crash penalty
        if self.has_crashed:
            penalty = self.speed ** 2 / 150 * diff
            score -= penalty

        # if self.car_index == 0:
        #     print(f"{diff=}\n"
        #           f"{self.speed ** 2 / 150=}\n"
        #           f"{penalty=}")

        self.network.fitness = score
        return

    def get_track_car_heading_difference(self, green_line_index: Optional[int] = None):
        if green_line_index is None:
            green_line_index = self.track.get_closest_green_line_index(self.position)
        angle = self.track_heading(green_line_index)
        heading = self.heading
        if heading > np.pi:
            heading -= 2 * np.pi
        diff = angle - heading
        diff -= 2 * np.pi if diff > np.pi else 0
        diff += 2 * np.pi if diff < -np.pi else 0
        return diff

    @property
    def forward_unit_vector(self):
        return np.array([np.sin(self.heading), np.cos(self.heading)])

    @property
    def left_unit_vector(self):
        return np.array([-np.cos(self.heading), np.sin(self.heading)])

    def unit_vector_rotated(self, rot):
        return np.array([np.sin(self.heading - rot), np.cos(self.heading - rot)])

    def track_unit_vector(self, green_line_index: Optional[int] = None):
        heading = self.track_heading(green_line_index)
        return np.array([np.sin(heading), np.cos(heading)])

    def track_heading(self, green_line_index: Optional[int] = None):
        if green_line_index is None:
            green_line_index = self.track.get_closest_green_line_index(self.position)

        behind = self.track.green_line[green_line_index-10]
        ahead = self.track.green_line[(green_line_index+10) % len(self.track.green_line)]
        dy, dx = behind[0] - ahead[0], ahead[1] - behind[1]
        angle = -np.arctan2(dy, dx)
        return angle  # [-pi, pi]

    def get_green_line_ahead(self, distance, index: Optional[int] = None):
        if index is None:
            index = self.track.get_closest_green_line_index(self.position)
        pixels = int(distance / self.track.scale)
        return self.track.green_line[(index + pixels) % len(self.track.green_line)]

    def get_closest_green_line_position(self):
        index = self.track.get_closest_green_line_index(self.position)
        return self.track.green_line[index]

    def get_track_ahead(self, distance, index: Optional[int] = None):
        if index is None:
            index = self.track.get_closest_green_line_index(self.position)
        vector = self.track_unit_vector(index)
        y, x = self.track.green_line[index] + vector * distance / self.track.scale
        return int(y), int(x)

    def get_forward(self, distance):
        y, x = self.track.vehicle_pos_to_px(self.position) + self.forward_unit_vector * distance / self.track.scale
        return int(y), int(x)

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
        index = self.track.get_closest_green_line_index(self.position)
        pos_y, pos_x = self.track.vehicle_pos_to_px(self.position)
        y, x = self.track.green_line[index]
        dist_px = np.hypot(y - pos_y, x - pos_x)
        distance = dist_px * self.track.scale
        y2, x2 = self.track.green_line[index - 1]

        def is_left(ax, ay, bx, by, cx, cy):
            return ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) > 0

        def area_triangle(ax, ay, bx, by, cx, cy):
            area = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2
            return area

        def get_area_track(dist):
            y1, x1 = self.get_green_line_ahead(10, index)
            y2, x2 = self.get_green_line_ahead(-10, index)
            pos_y, pos_x = self.get_green_line_ahead(dist, index)
            area = area_triangle(x2, y2, x1, y1, pos_x, pos_y)
            return area * self.track.scale * self.track.scale / (10 * dist)

        if is_left(x2, y2, x, y, pos_x, pos_y):
            distance = -distance  # Distance to center line

        diff = self.get_track_car_heading_difference(index)

        input = self.speed/60, distance/10, diff, get_area_track(20), get_area_track(50), get_area_track(100), get_area_track(200)

        return input

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
