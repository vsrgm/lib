#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <errno.h>

int set_interface_attribs(int fd, int speed)
{
	struct termios tty;

	if (tcgetattr(fd, &tty) < 0) {
		printf("Error from tcgetattr: %s\n", strerror(errno));
		return -1;
	}

	cfsetospeed(&tty, (speed_t)speed);
	cfsetispeed(&tty, (speed_t)speed);

	tty.c_cflag |= (CLOCAL | CREAD);    /* ignore modem controls */
	tty.c_cflag &= ~CSIZE;
	tty.c_cflag |= CS8;         /* 8-bit characters */
	tty.c_cflag &= ~PARENB;     /* no parity bit */
	tty.c_cflag &= ~CSTOPB;     /* only need 1 stop bit */
	tty.c_cflag &= ~CRTSCTS;    /* no hardware flowcontrol */

	/* setup for non-canonical mode */
	tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
	tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
	tty.c_oflag &= ~OPOST;

	/* fetch bytes as they become available */
	tty.c_cc[VMIN] = 1;
	tty.c_cc[VTIME] = 1;

	if (tcsetattr(fd, TCSANOW, &tty) != 0) {
		printf("Error from tcsetattr: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

int open_port(int *fd, char *port)
{
	struct termios newtio;
			
	*fd = open(port, O_RDWR | O_NOCTTY | O_SYNC); //O_NDELAY |
	if(*fd < 0) {
		return *fd;
	}

	return 0;
}

int printstring(unsigned char *string, unsigned int string_length)
{
	unsigned int pos;
	printf("\n--------------------------------------------STARTED-------------------------------------------------\n");

	for(pos=0;pos<string_length;pos++)
	{
		printf("0x%02x-%c ",string[pos],(string[pos] >= 32)?((string[pos] <= 126)?string[pos]:' '):' ');
		if((((pos+1)%20)==0))
			printf("\n");
			
	}
		
	printf("\n--------------------------------------------ENDED---------------------------------------------------\n");
	return 0;
}

int get_readbuf(int fd, unsigned char *rbuf, int expected)
{
	int rlen = 0;
	int ret;
	while(1) {
		ret = read (fd, &rbuf[rlen], (expected <= 0)?1:expected);
		if (ret > 0) {
			expected -= ret;
			rlen += ret;
			if (expected <= 0)
				break;
		}else {
			return ret;
		}
	}
	return rlen;
}

int main (int argc, char **argv)
{
	int fd, ret;
	unsigned char wbuf[100] = {"Hello"};
	unsigned char rbuf[100];

	ret = open_port(&fd, "/dev/ttyUSB0");
	if (ret < 0) {
		printf("Failed to open serial port \n");
	}
	ret = write (fd, wbuf, strlen(wbuf));
	get_readbuf(fd, rbuf, 5);
	printstring(rbuf, 5);
	close(fd);
}
