#!/bin/bash

set -eu -o pipefail

CONTAINER=carla-simulator-server

docker rm -f "${CONTAINER}" &>/dev/null || true

function launch_server_with_display {

    docker run -it \
	   -p 2000-2002:2000-2002 \
	   --runtime=nvidia \
	   --ipc=host \
	   --privileged \
	   -v $PWD/carla_settings.ini:/home/carla/CarlaSettings.ini \
	   -v /tmp/.X11-unix:/tmp/.X11-unix \
	   -e DISPLAY="${DISPLAY}" \
	   -e SDL_VIDEODRIVER=x11 \
	   -e NVIDIA_VISIBLE_DEVICES=1 \
	   --name "${CONTAINER}" \
	   carla-simulator:latest \
	   ./CarlaUE4.sh -carla-settings=CarlaSettings.ini -server $@

}

function launch_server_sans_display {

    docker run -it \
	   -p 2000-2002:2000-2002 \
	   --runtime=nvidia \
	   --ipc=host \
	   --privileged \
	   -v $PWD/carla_settings.ini:/home/carla/CarlaSettings.ini \
	   -e SDL_VIDEODRIVER=offscreen \
	   -e NVIDIA_VISIBLE_DEVICES=1 \
	   --name "${CONTAINER}" \
	   carla-simulator:latest \
	   ./CarlaUE4.sh -carla-settings=CarlaSettings.ini -server $@

}

if [[ "yes" == "${DISPLAY:-no}" ]]; then
    launch_server_with_display
else
    launch_server_sans_display
fi
