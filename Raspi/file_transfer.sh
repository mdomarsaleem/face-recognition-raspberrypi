#!/bin/bash

sshpass -p(password) rsync -avP /home/pi/Desktop/Image_db/* (username)@(server ip address):/home/mckc/Raspberry

ping -c 1 (server ip) &> /dev/null && rm -r /home/pi/Desktop/Image_db/* || echo fail
