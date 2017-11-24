SRC=$(wildcard *.c)
OBJS=$(join $(addsuffix , $(dir $(SRC))), $(notdir $(SRC:.c=.elf)))

%.elf : %.c ../libcvt.a
	gcc  $< ../libcvt.a -o $@

all: $(OBJS) ../libcvt.a
	@echo "done"
clean:
	rm -rf *.elf


