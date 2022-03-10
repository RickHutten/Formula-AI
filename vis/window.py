from typing import Generator, List
from threading import Thread

import numpy as np
import pygame
from PIL import Image
from pygame import Surface

from track import Track
from car import Car


class Window:
    def __init__(self, track: Track, update: Generator):
        self.track = track
        self.update = update

        # Default window parameters
        self.window_size = np.array([1200, 900])  # (width, height)
        self.zoom = 2
        self.zoom_target = 2
        self.camera = track.start  # (y, x)
        self.paused = False
        self.speeds = [0.25, 0.5, 1, 2, 4, np.inf]
        self.speed_index = len(self.speeds) - 1
        self.step = False
        self.show_network = True
        self.show_debug = True

        # Init pygame
        pygame.init()
        pygame.display.set_mode(self.window_size)
        self.screen: Surface = pygame.display.get_surface()
        self.text_font = pygame.font.SysFont('Courier', 24)
        self.assets = {}
        self.clock = pygame.time.Clock()
        self.track_start_y = 0
        self.track_start_x = 0

        # Status
        self.vehicles: List[Car] = []
        self.data: dict = {}
        self.running = True
        self.cropped_size = 0

    def update_loop(self):
        """Continuously updates the simulation"""
        next(self.update)  # Initiates iterator
        while self.running:
            state = {
                'paused': self.paused,
                'speed': self.speeds[self.speed_index],
                'step': self.step,
            }
            self.step = False  # Reset step
            try:
                self.data = self.update.send(state)
                self.vehicles = self.data['vehicles']
            except StopIteration:
                # Generator is exhausted
                return

        # Save the generation to a file
        self.update.send({'save': True})

    def loop(self):
        """Runs the window in a loop that """
        update_thread = Thread(target=self.update_loop)
        update_thread.start()

        # Visualization loop
        while self.running:
            # event handling, gets all event from the event queue
            for event in pygame.event.get():
                # When you click on the window close button
                if event.type == pygame.QUIT:
                    self.running = False
                # Zoom in and out
                elif event.type == pygame.MOUSEWHEEL:
                    self.zoom_target *= 1.15 ** event.y
                    self.zoom_target = min(5, max(0.05, self.zoom_target))
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        # Toggle pause
                        self.paused = not self.paused
                    if event.key == pygame.K_COMMA:  # Slow down
                        self.speed_index = max(0, self.speed_index - 1)
                    if event.key == pygame.K_PERIOD:  # Speed up
                        self.speed_index = min(len(self.speeds) - 1, self.speed_index + 1)
                    if event.key == pygame.K_SLASH:  # Perform one step
                        self.step = True
                    if event.key == pygame.K_n:  # Toggle network visibility
                        self.show_network = not self.show_network
                    if event.key == pygame.K_b:  # Toggle debug information visibility
                        self.show_debug = not self.show_debug
                # Pan the camera
                elif event.type == pygame.MOUSEMOTION:
                    if event.buttons[0] == 1:
                        new_camera = self.camera - np.array((event.rel[1], event.rel[0])) / self.zoom
                        new_camera[0] = max(min(self.track.image.shape[0], new_camera[0]), 0)
                        new_camera[1] = max(min(self.track.image.shape[1], new_camera[1]), 0)
                        self.set_camera(new_camera)

            # Let zoom grow to zoom target
            self.zoom = 0.5 * self.zoom + 0.5 * self.zoom_target

            # Update whats drawn on the screen
            self.draw()

            # Limit to 30 fps
            self.clock.tick(30)

    def set_camera(self, center: np.array):
        """
        Sets the center of the camera at the given position.
        Position is the pixel coordinate of the track image.
        """
        self.camera = center

    def draw(self):
        """Draws all the elements on the screen"""
        self.screen.fill(pygame.Color('white'))
        self.draw_track()
        self.draw_vehicles()
        self.draw_hud()

        # Update the display
        pygame.display.flip()

    def draw_track(self):
        # Get the width and height of the image due to the zoom
        width, height = self.window_size[0] // self.zoom, self.window_size[1] // self.zoom
        width, height = int((width // 2) * 2), int((height // 2) * 2)

        # Determine the part of the image that is inside the window
        self.track_start_y, track_end_y = int(self.camera[0] - height // 2), int(self.camera[0] + height // 2)
        self.track_start_x, track_end_x = int(self.camera[1] - width // 2), int(self.camera[1] + width // 2)

        # Calculate the draw position of the image
        draw_pos_y = 0 if self.track_start_y >= 0 else - self.track_start_y * self.zoom
        draw_pos_x = 0 if self.track_start_x >= 0 else - self.track_start_x * self.zoom

        # Calculate crop limits
        crop_start_y = max(0, self.track_start_y)
        crop_start_x = max(0, self.track_start_x)
        crop_end_x = min(self.track.image.shape[1], track_end_x)
        crop_end_y = min(self.track.image.shape[0], track_end_y)

        # Skip x number of pixels if the image is zoomed out enough
        crop_skip = int(max(1, np.ceil(width / self.window_size[0])))

        # Copy the pixels from the array that we need
        cropped = self.track.image[crop_start_y:crop_end_y:crop_skip, crop_start_x:crop_end_x:crop_skip, :]
        self.cropped_size = cropped.shape[0] * cropped.shape[1]

        # Draw the image to the screen
        im = Image.fromarray(cropped)
        arr = bytearray(im.tobytes('raw', 'RGB'))
        image = pygame.image.frombuffer(arr, (cropped.shape[1], cropped.shape[0]), "RGB")
        image = pygame.transform.scale(image, (int(self.zoom * (crop_end_x - crop_start_x+1)), int(self.zoom * (crop_end_y - crop_start_y+1))))
        self.screen.blit(image, (draw_pos_x, draw_pos_y))

    def draw_vehicles(self):
        # Load the asset if not loaded yet
        if 'f1' not in self.assets:
            self.assets['f1'] = pygame.image.load('./vis/assets/f1.png').convert_alpha()

        img: Surface = self.assets['f1']
        car_size = 2 * np.array([5.7, 2])
        car_size_scaled = (self.zoom * car_size / self.track.scale).astype('int')

        # Too small to draw
        if car_size_scaled[0] == 0 or car_size_scaled[1] == 0:
            return

        image = pygame.transform.scale(img, (self.zoom * car_size / self.track.scale).astype('int'))
        img_trans: Surface = image.copy()
        img_trans.set_alpha(128)

        for vehicle in self.vehicles[:10]:
            track_location = self.track.vehicle_pos_to_px(vehicle.position)  # (y, x)
            window_location = self.track_px_to_window_px(track_location)  # (x, y)

            if vehicle.car_index == 0:
                img_rot = pygame.transform.rotate(image, -np.rad2deg(vehicle.heading))
            else:
                img_rot = pygame.transform.rotate(img_trans, -np.rad2deg(vehicle.heading))

            rect_rot = img_rot.get_rect(center=img.get_rect(center=window_location).center)
            self.screen.blit(img_rot, rect_rot.topleft)

    def draw_hud(self):
        # We can't draw anything if we don't have vehicles
        if len(self.vehicles) == 0:
            return

        if self.show_network:
            self.draw_network()
        if self.show_debug:
            self.draw_debug()
        self.draw_hud_text()

    def draw_hud_text(self):
        # Draw the hud text
        def number_to_indicator(x: float):
            """Outputs a number [-1, 1] as an indicator string eg. -0.50 -> '|-|-:---|' """
            str = ''
            x = np.round(x * 10)
            for i in range(-10, 11):
                if i == x:
                    str += "|"
                elif i == 0:
                    str += ":"
                else:
                    str += '-'
            return f'|{str}|'

        def list_format(l: list):
            """Formats a list of floats"""
            s = '['
            for i in l:
                s += f"{i:+.3f},"
            return s + ']'

        start_y = 0
        steering_angle, gas_pedal = self.vehicles[0].network.get_output()
        heading_diff = self.vehicles[0].get_track_car_heading_difference()
        alive = sum([0 if v.has_crashed else 1 for v in self.vehicles])
        text_to_print = [
            f'FPS: {self.clock.get_fps():.0f}',
            f'zoom: {self.zoom:.2f}',
            f'speed: {self.speeds[self.speed_index]:.2f}',
            # f'image size: {self.cropped_size} px',
            f'alive cars: {alive}'
            f'',
            f'time: {self.data.get("time", 0):05.1f}',
            f'generation: {self.data.get("generation", 0):03}',
            f'gas {gas_pedal: .2f} {number_to_indicator(gas_pedal)}',
            f'steering {steering_angle: .2f} {number_to_indicator(steering_angle)}',
            f'speed {3.6 * self.vehicles[0].speed: 06.2f} km/h',
            # f'heading diff {heading_diff: .2f} {number_to_indicator(heading_diff)}',
            f'input {list_format(self.vehicles[0].network.get_input())}'
        ]
        for text in text_to_print:
            text_surface = self.text_font.render(text, True, (0, 0, 0))

            hud_bg = pygame.Surface(text_surface.get_size())
            hud_bg.set_alpha(128)
            hud_bg.fill((255, 255, 255))
            self.screen.blit(hud_bg, (0, start_y))
            self.screen.blit(text_surface, (0, start_y))
            start_y += text_surface.get_size()[1]

        bottom_text = f"[,] slow down; [.] speed up; [space] pause; [/] step; [n] network; [b] debug "
        text_surface = self.text_font.render(bottom_text, True, (0, 0, 0))
        hud_bg = pygame.Surface(text_surface.get_size())
        hud_bg.set_alpha(128)
        y = self.screen.get_height() - text_surface.get_size()[1]
        hud_bg.fill((255, 255, 255))
        self.screen.blit(hud_bg, (0, y))
        self.screen.blit(text_surface, (0, y))


    def draw_network(self):
        """Draws the best network onto the screen"""
        width, height = int(self.window_size[0]/2), int(self.window_size[1]/2)
        surface = pygame.Surface([width, height], pygame.SRCALPHA)
        network = self.vehicles[0].network
        x_spacing = width / network.num_layers
        max_per_layer = max(network.layer_structure)
        y_spacing = height / (max_per_layer + 1)
        x = -0.5 * x_spacing
        green = pygame.Color('green')
        red = pygame.Color('red')
        for i, layer in enumerate(network.layers):
            x += x_spacing
            y = (height - (layer.num_neurons * y_spacing)) / 2
            for j, neuron in enumerate(layer.neurons):
                y += y_spacing
                # if i == 0:
                #     val = min(max(neuron.value, 0), 1)
                #     pygame.draw.circle(surface, red.lerp(green, val), (x, y), y_spacing / 3)
                # else:
                #     val = (neuron.value + 1) / 2
                #     pygame.draw.circle(surface, red.lerp(green, val), (x, y), y_spacing / 3)

                if i == 0:
                    val = min(max((neuron.value + 1) / 2, 0), 1)
                else:
                    val = (neuron.bias + 1) / 2

                assert 0 <= val <= 1, f"val not in [0, 1] but {val}"
                pygame.draw.circle(surface, red.lerp(green, val), (x, y), y_spacing / 3)

                if i + 1 == network.num_layers:
                    # Skip the next for the last layer
                    continue

                next_layer = network.layers[i + 1]
                x_next = x + x_spacing
                y_next = (height - (next_layer.num_neurons * y_spacing)) / 2
                for next_neuron in next_layer.neurons:
                    y_next += y_spacing
                    value = next_neuron.dendrites[j]
                    line_width = int(max(1., abs(value) * 5))
                    val = (value + 1) / 2
                    pygame.draw.line(surface, red.lerp(green, val), (x, y), (x_next, y_next), line_width)

        self.screen.blit(surface, (self.window_size[0] - width,  0))

    def draw_debug(self):
        width, height = self.window_size
        surface = pygame.Surface([width, height], pygame.SRCALPHA)

        red = pygame.Color('red')
        green = pygame.Color('green')
        blue = pygame.Color('blue')
        yellow = pygame.Color('yellow')

        # Current position on green line
        pos = self.vehicles[0].get_closest_green_line_position()
        x, y = self.track_px_to_window_px(pos)  # (x, y)
        pygame.draw.circle(surface, blue, (x, y), 10)

        for d in [25, 50, 100, 200]:
            pos = self.vehicles[0].get_green_line_ahead(d)
            x, y = self.track_px_to_window_px(pos)  # (x, y)
            pygame.draw.circle(surface, green, (x, y), 10)

        for d in [-10, 10]:
            pos = self.vehicles[0].get_green_line_ahead(d)
            x, y = self.track_px_to_window_px(pos)  # (x, y)
            pygame.draw.circle(surface, yellow, (x, y), 10)

        # pos = self.vehicles[0].get_forward(150)
        # x, y = self.track_px_to_window_px(pos)  # (x, y)
        # pygame.draw.circle(surface, red, (x, y), 10)

        self.screen.blit(surface, (0,  0))

    def track_px_to_window_px(self, coord: np.array) -> np.array:
        """
        Converts a track pixel coordinate to the window coordinate
        Args:
            coord (y, x)
        Returns:
            window coordinate (x, y): np.array
        """
        diff_track = coord - self.camera
        diff_window = diff_track[::-1] * self.zoom
        window_coord = diff_window + self.window_size / 2
        return window_coord
