SRC=$(wildcard *.c)
OBJS=$(join $(addsuffix , $(dir $(SRC))), $(notdir $(SRC:.c=.o)))
OPENGLLIBS=-lglut -lGL -lGLU
%.o : %.c
	gcc -c $< -o $@

objloaderelephant.elf: objloaderelephant.o
	gcc $< -o $@ $(OPENGLLIBS)

clean:
	rm -rf *.elf *.o
