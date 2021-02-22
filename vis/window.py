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
        self.window_size = np.array([1536, 864])  # (width, height)
        self.zoom = 2
        self.camera = np.array([7264, 6356])  # (y, x)

        # Init pygame
        pygame.init()
        pygame.display.set_mode(self.window_size)
        self.screen: Surface = pygame.display.get_surface()
        self.text_font = pygame.font.SysFont('Arial', 24)
        self.assets = {}
        self.clock = pygame.time.Clock()
        self.track_start_y = 0
        self.track_start_x = 0

        # Status
        self.vehicles: List[Car] = []
        self.data: dict = {}
        self.running = True

        # Start main loop
        self.loop()

    def update_loop(self):
        while self.running:
            self.data = next(self.update)
            self.vehicles = self.data['vehicles']

    def loop(self):
        update_thread = Thread(target=self.update_loop)
        update_thread.start()

        # main loop
        while self.running:
            self.clock.tick(30)
            # event handling, gets all event from the event queue
            for event in pygame.event.get():
                # only do something if the event is of type QUIT
                if event.type == pygame.QUIT:
                    # change the value to False, to exit the main loop
                    self.running = False

                # Zoom in and out
                elif event.type == pygame.MOUSEWHEEL:
                    self.zoom *= 1.1 ** event.y
                    self.zoom = min(5, max(0.05, self.zoom))
                # Pan the camera
                elif event.type == pygame.MOUSEMOTION:
                    if event.buttons[0] == 1:
                        new_camera = self.camera - np.array((event.rel[1], event.rel[0])) / self.zoom
                        new_camera[0] = max(min(self.track.image.shape[1], new_camera[0]), 0)
                        new_camera[1] = max(min(self.track.image.shape[0], new_camera[1]), 0)
                        self.set_camera(new_camera)

            self.draw()

    def draw(self):
        """Draws all the elements on the screen"""
        self.draw_track()
        self.draw_vehicles()
        self.draw_hud()

        pygame.display.flip()

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

    def draw_vehicles(self):
        # Load the asset if not loaded yet
        if 'f1' not in self.assets:
            self.assets['f1'] = pygame.image.load('./vis/assets/f1.png').convert_alpha()

        img: Surface = self.assets['f1']
        car_size = 2 * np.array([5.7, 2])
        image = pygame.transform.scale(img, (self.zoom * car_size / self.track.scale).astype('int'))

        for vehicle in self.vehicles:
            track_location = self.track.vehicle_pos_to_px(vehicle.position)  # (y, x)
            window_location = self.track_px_to_window_px(track_location)  # (x, y)

            img_rot = pygame.transform.rotate(image, -np.rad2deg(vehicle.heading))
            rect_rot = img_rot.get_rect(center=img.get_rect(center=window_location).center)

            self.screen.blit(img_rot, rect_rot.topleft)

    def set_camera(self, center: np.array):
        """
        Sets the center of the camera at the given position.
        Position is the pixel coordinate of the track image.
        """
        self.camera = center

    def draw_hud(self):
        start_y = 0
        text_to_print = [
            f'FPS: {self.clock.get_fps():.0f}',
            f'zoom: {self.zoom:.2f}',
            f'time: {self.data["time"]:.1f}',
            f'generation: {self.data["generation"]}'
        ]
        for text in text_to_print:
            text_surface = self.text_font.render(text, True, (0, 0, 0))

            hud_bg = pygame.Surface(text_surface.get_size())
            hud_bg.set_alpha(128)
            hud_bg.fill((255, 255, 255))
            self.screen.blit(hud_bg, (0, start_y))
            self.screen.blit(text_surface, (0, start_y))
            start_y += text_surface.get_size()[1]

    def draw_track(self):
        width, height = self.window_size[0] // self.zoom, self.window_size[1] // self.zoom
        width, height = int((width // 2) * 2), int((height // 2) * 2)

        self.track_start_y, track_end_y = int(self.camera[0] - height // 2), int(self.camera[0] + height // 2)
        self.track_start_x, track_end_x = int(self.camera[1] - width // 2), int(self.camera[1] + width // 2)

        draw_pos_y = 0 if self.track_start_y >= 0 else - self.track_start_y * self.zoom
        draw_pos_x = 0 if self.track_start_x >= 0 else - self.track_start_x * self.zoom

        crop_start_y = max(0, self.track_start_y)
        crop_start_x = max(0, self.track_start_x)
        crop_end_y = min(self.track.image.shape[1], track_end_y)
        crop_end_x = min(self.track.image.shape[0], track_end_x)

        crop_skip = int(max(1, np.ceil(width / self.window_size[0])))

        cropped = self.track.image[crop_start_y:crop_end_y:crop_skip, crop_start_x:crop_end_x:crop_skip, :]
        im = Image.fromarray(cropped)
        arr = bytearray(im.tobytes('raw', 'RGB'))
        image = pygame.image.frombuffer(arr, (cropped.shape[1], cropped.shape[0]), "RGB")
        image = pygame.transform.scale(image, (int(self.zoom * (crop_end_x - crop_start_x+1)), int(self.zoom * (crop_end_y - crop_start_y+1))))
        self.screen.fill((0, 0, 0))
        self.screen.blit(image, (draw_pos_x, draw_pos_y))
