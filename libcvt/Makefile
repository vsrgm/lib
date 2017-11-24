SRC=$(wildcard *.c)
OBJS=$(join $(addsuffix , $(dir $(SRC))), $(notdir $(SRC:.c=.o)))

%.o : %.c
	gcc -c $< -o $@

libcvt.a: $(OBJS)
	ar -cvq $@ $^
#	gcc -shared -o libcvt.so $^

clean:
	rm -rf *.o *.a *.so*
