#!/bin/bash

sshpass -ppassword rsync -avP /home/pi/Desktop/Image_db/* mckc@192.168.100.94:/home/mckc/Raspberry

ping -c 1 192.168.100.94 &> /dev/null && rm -r /home/pi/Desktop/Image_db/* || echo fail
