SRC=$(wildcard *.c)
OBJS=$(join $(addsuffix , $(dir $(SRC))), $(notdir $(SRC:.c=.elf)))

%.elf : %.c
	gcc  $< -I/usr/include/SDL/ -o $@ -lSDL

all: $(OBJS)

clean:
	rm -rf *.elf
