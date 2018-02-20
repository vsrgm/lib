SRC=$(wildcard *.c)
OBJS=$(join $(addsuffix , $(dir $(SRC))), $(notdir $(SRC:.c=.o)))
OPENGLLIBS=-lglut -lGL
%.o : %.c
	gcc -c $< -o $@

#dinoshade.elf: dinoshade.o
#	gcc dinoshade.o -o dinoshade.elf

triangle.elf: triangle.o
	gcc triangle.o -o triangle.elf $(OPENGLLIBS)

clean:
	rm -rf *.elf *.o
