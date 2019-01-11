from __future__ import print_function

from carla.client import make_carla_client, CarlaClient
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line, make_connection

settings = CarlaSettings()
settings.set(
    SynchronousMode=True,
    SendNonPlayerAgentsInfo=True,
    NumberOfVehicles=20,
    NumberOfPedestrians=40,
    WeatherId=7,
    QualityLevel='Epic')
settings.randomize_seeds()

# The default camera captures RGB images of the scene.
camera_scene = Camera('CameraRGB')
# Set image resolution in pixels.
camera_scene.set_image_size(800, 600)
# Set its position relative to the car in meters.
camera_scene.set_position(0.30, 0, 1.30)
settings.add_sensor(camera_scene)

# Let's add another camera producing ground-truth depth.
camera_depth = Camera('CameraDepth', PostProcessing='Depth')
camera_depth.set_image_size(800, 600)
camera_depth.set_position(0.30, 0, 1.30)
settings.add_sensor(camera_depth)

lidar = Lidar('Lidar32')
lidar.set_position(0, 0, 2.50)
lidar.set_rotation(0, 0, 0)
lidar.set(
    Channels=32,
    Range=50,
    PointsPerSecond=100000,
    RotationFrequency=10,
    UpperFovLimit=10,
    LowerFovLimit=-30)
settings.add_sensor(lidar)


client = CarlaClient('10.0.0.142', 2000, timeout=15)
client.connect()

# A protocol buffer object with all the informations
scene = client.load_settings(settings)

# Choose one player start at random.
number_of_player_starts = len(scene.player_start_spots)
player_start = 7

# Notify the server that we want to start the episode at the
# player_start index. This function blocks until the server is ready
# to start the episode.
print('Starting new episode at %r...' % scene.map_name)
client.start_episode(player_start)

measurements, sensor_data = client.read_data()
