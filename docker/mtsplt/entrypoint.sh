#!/bin/bash

apt-get update
msfdb init
exec msfconsole $@
