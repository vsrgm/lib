SRC=$(wildcard *.c)
OBJS=$(join $(addsuffix , $(dir $(SRC))), $(notdir $(SRC:.c=.elf)))

#/usr/local/gcc-linaro-arm-linux-gnueabihf-4.8-2014.03_linux/bin/arm-linux-gnueabihf-
#-I/usr/local/gcc-linaro-arm-linux-gnueabihf-4.8-2014.03_linux/arm-linux-gnueabihf/libc/usr/include
%.elf : %.c
	$(CROSS_COMPILE)-gcc --sysroot=${SYSROOT} --static -mfloat-abi=hard -mfpu=neon $< ../libcvtneon.a -o $@

all: $(OBJS)
	@echo "done"
clean:
	rm -rf *.elf


