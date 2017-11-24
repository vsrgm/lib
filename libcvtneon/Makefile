SRC=$(wildcard *.c)
OBJS=$(join $(addsuffix , $(dir $(SRC))), $(notdir $(SRC:.c=.o)))
#/usr/local/gcc-linaro-arm-linux-gnueabihf-4.8-2014.03_linux/bin/arm-linux-gnueabihf-
#-I/usr/local/gcc-linaro-arm-linux-gnueabihf-4.8-2014.03_linux/arm-linux-gnueabihf/libc/usr/include
SYSROOT:=/

%.o : %.c
	$(CROSS_COMPILE)-gcc --sysroot=${SYSROOT} -mfloat-abi=hard -mfpu=neon -c $< -o $@

libcvtneon.a: $(OBJS)
	$(CROSS_COMPILE)-ar -cvq $@ $^
	$(CROSS_COMPILE)-gcc -shared -o libcvtneon.so $^

clean:
	rm -rf *.o *.a *.so*
