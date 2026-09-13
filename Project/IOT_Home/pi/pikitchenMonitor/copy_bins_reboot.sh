#sshpass -p "pi123"
scp -r server.c pi@192.168.0.113:/home/pi/
#sshpass -p "pi123"
#ssh pi@192.168.0.113 "cd /home/pi/; gcc server.c -o server.elf -lpthread;sync;sudo reboot"
