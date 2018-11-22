#!/bin/bash

set -eux -o pipefail

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function install_system_kvm2_driver {
    # Install libvirt and qemu-kvm on your system
    sudo apt-get install libvirt-clients libvirt-daemon-system qemu-kvm

    cat <<'_LIBVIRT_GRP_EOF_'
    # Add yourself to the libvirt group so you don't need to sudo
    # NOTE: For older Debian/Ubuntu versions change the group to 'libvirtd'
    sudo usermod -aG libvirt $(whoami)

    # Update your current session for the group change to take effect
    newgrp libvirt
_LIBVIRT_GRP_EOF_

    curl -fSL \
         -O https://storage.googleapis.com/minikube/releases/latest/docker-machine-driver-kvm2
    sudo install docker-machine-driver-kvm2 /usr/local/bin/
}

function get_or_update_system_minikube {
    curl -fSL \
         -o minikube -O \
         https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    chmod +x minikube && sudo cp minikube /usr/local/bin/ && rm -f minikube
}

function get_or_update_system_kubectl {
    local release="$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)"
    curl -fSL \
         -o kubectl \
         -O "https://storage.googleapis.com/kubernetes-release/release/${release}/bin/linux/amd64/kubectl"
    chmod +x kubectl && sudo cp kubectl /usr/local/bin/ && rm -f kubectl
}

export MINIKUBE_WANTUPDATENOTIFICATION=false
export MINIKUBE_WANTREPORTERRORPROMPT=false
export MINIKUBE_HOME="${HOME}"
export CHANGE_MINIKUBE_NONE_USER=true

mkdir -p "${MINIKUBE_HOME}" && pushd $_
mkdir -p .kube
mkdir -p .minikube
touch .kube/config
which docker-machine-driver-kvm2 \
    || install_system_kvm2_driver
which minikube &>/dev/null \
    || get_or_update_system_minikube
which kubectl &>/dev/null \
    || get_or_update_system_kubectl
popd

export KUBECONFIG=${MINIKUBE_HOME}/.kube/config
sudo -E minikube stop || true
sudo -E minikube delete || true
# sudo -E minikube start --vm-driver=kvm2
# minikube start --vm-driver kvm2 --gpu

sudo rm -rf /var/lib/minikube/ || true
sudo rm -rf /etc/kubernetes || true

sudo -E minikube start \
     --vm-driver=none \
     --apiserver-ips 127.0.0.1 \
     --apiserver-name localhost \
     --kubernetes-version=v1.12.2

kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v1.12/nvidia-device-plugin.yml

# this for loop waits until kubectl can access the api server that Minikube has created
for i in {1..150}; do # timeout for 5 minutes
    kubectl get po &> /dev/null
    if [ $? -ne 1 ]; then
        break
    fi
    sleep 2
done

ln -fsn ${MINIKUBE_HOME}/.minikube ~/.minikube

kubectl run hello-minikube --image=k8s.gcr.io/echoserver:1.4 --port=8080
kubectl get nodes -ojson | jq .items[].status.capacity
