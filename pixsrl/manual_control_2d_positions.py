#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.
"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""

from __future__ import print_function

import argparse
import logging
import random
import time
import math
import colorsys

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
    from numpy.linalg import pinv, inv
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.transform import Transform

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 640
MINI_WINDOW_WIDTH = 400
MINI_WINDOW_HEIGHT = 200

WINDOW_WIDTH_HALF = WINDOW_WIDTH / 2
WINDOW_HEIGHT_HALF = WINDOW_HEIGHT / 2

MAX_DEPTH = 16777215.0

# np.set_printoptions(precision=2, suppress=True)
np.set_printoptions(suppress=True)


def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel
    """
    try:
        array = image_converter.to_bgra_array(image)
        array = array.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        raw_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
        #normalized_depth = raw_depth / 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        return raw_depth
    except:
        return None


def point_in_canvas(pos):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < WINDOW_HEIGHT) and (pos[1] >= 0) and (
            pos[1] < WINDOW_WIDTH):
        return True
    return False


def draw_rect(array, pos, size, color=(255, 0, 255)):
    """Draws a rect"""
    point_0 = (pos[0] - size / 2, pos[1] - size / 2)
    point_1 = (pos[0] + size / 2, pos[1] + size / 2)
    if point_in_canvas(point_0) and point_in_canvas(point_1):
        for i in range(size):
            for j in range(size):
                array[int(point_0[0] + i), int(point_0[1] + j)] = color


def rand_color(seed):
    """Return random color based on a seed"""
    random.seed(seed)
    col = colorsys.hls_to_rgb(random.random(), random.uniform(.2, .8), 1.0)
    return (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))


def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=35,
        NumberOfPedestrians=15,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()

    camera_location = (2.0, 0.0, 1.4)

    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(*camera_location)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)

    cam_data_depth = sensor.Camera('DataCameraDepth', PostProcessing='Depth')
    cam_data_depth.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    cam_data_depth.set_position(*camera_location)
    cam_data_depth.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(cam_data_depth)

    camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera1.set_position(*camera_location)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)

    camera2 = sensor.Camera(
        'CameraSemSeg', PostProcessing='SemanticSegmentation')
    camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera2.set_position(*camera_location)
    camera2.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera2)

    if args.lidar:
        lidar = sensor.Lidar('Lidar32')
        lidar.set_position(0, 0, 2.5)
        lidar.set_rotation(0, 0, 0)
        lidar.set(
            Channels=32,
            Range=50,
            PointsPerSecond=100000,
            RotationFrequency=10,
            UpperFovLimit=10,
            LowerFovLimit=-30)
        settings.add_sensor(lidar)

    # (Intrinsic) K Matrix
    k = np.identity(3)
    k[0, 2] = WINDOW_WIDTH_HALF
    k[1, 2] = WINDOW_HEIGHT_HALF
    k[0, 0] = k[1, 1] = WINDOW_WIDTH / \
        (2.0 * math.tan(90.0 * math.pi / 360.0))

    camera_to_car_transform = camera0.get_unreal_transform()
    # camera_to_car_transform = camera0.get_transform()

    return settings, k, camera_to_car_transform


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step -
                     self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings, self._intrinsic, self._camera_to_car_transform = make_carla_settings(
            args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 16.43,
                             50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(
            WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None

        self._measurements = None
        self._extrinsic = None

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        # if self._city_name is not None:
        #     self._display = pygame.display.set_mode(
        #         (WINDOW_WIDTH + int(
        #             (WINDOW_HEIGHT / float(self._map.map_image.shape[0])) *
        #             self._map.map_image.shape[1]), WINDOW_HEIGHT),
        #         pygame.HWSURFACE | pygame.DOUBLEBUF)
        # else:
        #     self._display = pygame.display.set_mode(
        #         (WINDOW_WIDTH, WINDOW_HEIGHT),
        #         pygame.HWSURFACE | pygame.DOUBLEBUF)

        self._display = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT + MINI_WINDOW_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()

        # (Extrinsic) Rt Matrix
        # (Camera) local 3d to world 3d.
        # Get the transform from the player protobuf transformation.
        world_transform = Transform(measurements.player_measurements.transform)
        # Compute the final transformation matrix.
        self._extrinsic = world_transform * self._camera_to_car_transform

        self._measurements = measurements
        self._main_image = sensor_data.get('CameraRGB', None)
        self._mini_view_image1 = sensor_data.get('CameraDepth', None)
        self._mini_view_image2 = sensor_data.get('CameraSemSeg', None)
        self._lidar_measurement = sensor_data.get('Lidar32', None)
        self._data_cam_depth = sensor_data.get('DataCameraDepth', None)

        # Print measurements every second.
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z
                ])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z
                ])

                self._print_player_measurements_map(
                    measurements.player_measurements, map_position,
                    lane_orientation)
            else:
                self._print_player_measurements(
                    measurements.player_measurements)

            # Plot position on the map as well.

            self._timer.lap()

        control = self._get_keyboard_control(pygame.key.get_pressed())
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z
            ])
            self._agent_positions = measurements.non_player_agents

        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(
                measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot
        control.reverse = self._is_on_reverse
        return control

    def _print_player_measurements_map(self, player_measurements, map_position,
                                       lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _on_render(self):
        gap_x = (
            WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH - MINI_WINDOW_HEIGHT) // 3
        mini_image_y = WINDOW_HEIGHT

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            array.setflags(write=1)

            inv_extrinsic = inv(self._extrinsic.matrix)

            # https://github.com/carla-simulator/carla/issues/314#issuecomment-379328792
            def to_image_plane(x, y, z, depth_map=None):
                # Affine transformation with homogeneous coordinates
                pos_vec = [[x], [y], [z], [1.0]]
                # Discard the trailing 1.0 component
                pos_camera_centered = np.dot(inv_extrinsic, pos_vec)[:-1]
                pre_pos2d = np.dot(self._intrinsic, pos_camera_centered)
                pos2d = np.array([
                    pre_pos2d[0] / pre_pos2d[2], pre_pos2d[1] / pre_pos2d[2],
                    pre_pos2d[2]
                ])

                x_img = WINDOW_WIDTH - pos2d[0]
                y_img = WINDOW_HEIGHT - pos2d[1]

                if (y_img < 0) or (y_img >= WINDOW_HEIGHT) or \
                   (x_img < 0) or (x_img >= WINDOW_WIDTH):
                    return None, None

                cull_z = MAX_DEPTH
                if depth_map is not None:
                    # Adjust the size of the vehicle?
                    cull_z = depth_map[int(y_img)][int(x_img)] + 10

                if pos2d[2] <= 0 or pos2d[2] > cull_z:
                    return None, None

                return y_img, x_img

            def add_vec(v1, v2):
                x = v1.x + v2.x
                y = v1.y + v2.y
                z = v1.z + v2.z
                return x, y, z

            depth_map = depth_to_array(self._data_cam_depth)

            for agent in self._measurements.non_player_agents:
                if agent.HasField('vehicle'):
                    # TODO: with center, rotation and extent, find corner box coordinates
                    box_center = agent.vehicle.bounding_box.transform.location
                    #box_rot = agent.vehicle.bounding_box.transform.rotation
                    #box_extent = agent.vehicle.bounding_box.transform.extent
                    px, py, pz = add_vec(agent.vehicle.transform.location,
                                         box_center)
                    y_img, x_img = to_image_plane(px, py, pz, depth_map)

                    # TODO: cull points outside the scene by depth map
                    if y_img is not None:
                        # TODO: draw the actual 3D shape projected to image plane
                        draw_rect(array, (y_img, x_img), 10,
                                  rand_color(agent.id))

            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))  # starting position

        # Draw depth image to the left
        if self._mini_view_image1 is not None:
            array = image_converter.depth_to_logarithmic_grayscale(
                self._mini_view_image1)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, mini_image_y))

        # Draw semantic segmantation to the right
        if self._mini_view_image2 is not None:
            array = image_converter.labels_to_cityscapes_palette(
                self._mini_view_image2)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            self._display.blit(
                surface, (WINDOW_WIDTH - MINI_WINDOW_WIDTH, mini_image_y))

        # Lidar is a square that sits in the middle
        if self._lidar_measurement is not None:
            lidar_data = np.array(self._lidar_measurement.data[:, :2])
            lidar_data *= 2.0
            lidar_data += 100.0
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            # LIDAR always wants a square image, thus height x height
            lidar_img_size = (MINI_WINDOW_HEIGHT, MINI_WINDOW_HEIGHT, 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            surface = pygame.surfarray.make_surface(lidar_img)
            self._display.blit(surface,
                               (gap_x + MINI_WINDOW_WIDTH, mini_image_y))

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0] *
                        (float(WINDOW_HEIGHT) / float(self._map_shape[0])))
            h_pos = int(self._position[1] *
                        (new_window_width / float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z
                    ])

                    w_pos = int(
                        agent_position[0] *
                        (float(WINDOW_HEIGHT) / float(self._map_shape[0])))
                    h_pos = int(agent_position[1] *
                                (new_window_width / float(self._map_shape[1])))

                    pygame.draw.circle(surface, [255, 0, 255, 255],
                                       (w_pos, h_pos), 4, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))

        pygame.display.flip()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p',
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a',
        '--autopilot',
        action='store_true',
        default=True,
        help='enable autopilot')
    argparser.add_argument(
        '-l',
        '--lidar',
        action='store_true',
        default=True,
        help='enable Lidar')
    argparser.add_argument(
        '-q',
        '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help=
        'graphics quality level, a lower level makes the simulation run considerably faster.'
    )
    argparser.add_argument(
        '-m',
        '--map-name',
        metavar='M',
        default='Town02',
        help='plot the map of the current city (needs to match active map in '
        'server, options: Town01 or Town02)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
