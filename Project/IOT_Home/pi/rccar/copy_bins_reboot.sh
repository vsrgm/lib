#sshpass -p "pi1234"
scp server.c pi@192.168.0.104:/home/pi/Project/scripts
#sshpass -p "pi1234"
ssh pi@192.168.0.104 "cd /home/pi/Project/scripts; gcc server.c -o server.elf;sync;sudo reboot"
