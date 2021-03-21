import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000


class Track:
    def __init__(self, file_path: str, track_length_m: int):
        self.image = np.array(Image.open(file_path))
        self.name = file_path.split("/")[-1].split("\\")[-1].split(".")[0]
        self.image = self.image[:, :, :3]
        self.green_mask = np.zeros(self.image.shape[:-1]).astype('bool')
        self.red_mask = np.zeros(self.image.shape[:-1]).astype('bool')

        self.track_length_m = track_length_m

        self.grids = np.array(list(zip(*np.where(np.all(self.image == (255, 0, 0), axis=-1)))))
        self.start = np.array(list(zip(*np.where(np.all(self.image == (0, 255, 255), axis=-1))))[0])

        self.green_line, self.track_length_px = self.__compute_green_line()
        self.scale = self.track_length_m / self.track_length_px  # meter per pixel

    def is_outside(self, position: np.array) -> bool:
        """Returns whether the position is outside the track"""
        position = tuple(self.vehicle_pos_to_px(position))
        outside = np.all(self.image[position] == (255, 255, 255))
        return outside

    def vehicle_pos_to_px(self, position: np.array) -> np.array:
        return (position / self.scale).astype('int')

    def get_closest_green_line_index(self, position: np.array) -> int:
        """
        Get the closest point on the green line.
        Returns the index in the green_line list.
        """
        position = self.vehicle_pos_to_px(position)
        min = np.inf
        min_pos = ()

        i = 0
        step = 1
        while i < len(self.green_line):
            y, x = self.green_line[i]
            dist_sq = (y - position[0])**2 + (x - position[1])**2

            if dist_sq > 1000**2:
                step = 50
            elif dist_sq > 1000:
                step = 20
            else:
                step = 1

            if dist_sq < min:
                min = dist_sq
                min_pos = (y, x)

            i += step

        return self.green_line.index(min_pos)

    def get_lap_progress(self, position: np.array) -> float:
        """Get the lap position"""
        index = self.get_closest_green_line_index(position)
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
test_track = Track("./track/img/test_track.png", track_length_m=3784)
