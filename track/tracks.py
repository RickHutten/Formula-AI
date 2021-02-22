import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000


class Track:
    def __init__(self, file_path: str, track_length_m: int):
        self.image = np.asarray(Image.open(file_path))
        self.image = self.image[:, :, :3]

        self.track_length_m = track_length_m

        self.grids = np.array(list(zip(*np.where(np.all(self.image == (255, 0, 0), axis=-1)))))
        self.start = np.array(list(zip(*np.where(np.all(self.image == (0, 255, 255), axis=-1))))[0])

        self.green_line, self.track_length_px = self.__compute_green_line()
        self.scale = self.track_length_m / self.track_length_px  # meter per pixel

    def is_outside(self, position: np.array) -> bool:
        """Returns whether the position is outside the track"""
        position = tuple(self.vehicle_pos_to_px(position))
        return np.all(self.image[position] == (255, 255, 255))

    def vehicle_pos_to_px(self, position: np.array) -> np.array:
        return (position / self.scale).astype('int')

    def get_lap_progress(self, position: np.array) -> float:
        """Get the lap position"""
        position = self.vehicle_pos_to_px(position)
        min = np.inf
        min_pos = ()
        for y, x in self.green_line:
            dist_sq = (y - position[0])**2 + (x - position[1])**2
            if dist_sq < min:
                min = dist_sq
                min_pos = (y, x)

        index = self.green_line.index(min_pos)
        return index / len(self.green_line)

    def get_spawn_point_and_heading(self):
        """Return the spawn point (in meters) and the starting heading in radians"""
        return self.scale * self.start, np.pi

    def is_on_green(self, position: tuple):
        """Returns whether the position is on the green center line"""
        return np.all(self.image[position] == (0, 255, 0)) or np.all(position == self.start)

    def __compute_green_line(self):
        """Get a list of all green pixels in order of the lap. """
        position = tuple(self.start)
        green_line = [position, (position[0], position[1] - 1)]
        track_length = 0
        while green_line[-1] != tuple(self.start):
            last_pos = green_line[-1]
            for dx, dy in ((dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]):
                if dx == dy == 0:
                    continue
                new_pos = last_pos[0] + dy, last_pos[1] + dx
                if new_pos == green_line[-2]:
                    continue
                if not self.is_on_green(new_pos):
                    continue
                green_line.append(new_pos)
                if dx == 0 or dy == 0:
                    track_length += 1
                else:
                    track_length += 1.4142
                break
        return green_line, track_length


monza = Track("./track/img/monza.png", track_length_m=5793)
