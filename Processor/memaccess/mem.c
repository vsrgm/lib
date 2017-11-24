#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Usage: %s <phys_addr> <offset>\n", argv[0]);
		return 0;
	}

	off_t offset = strtoul(argv[1], NULL, 0);
	size_t len = strtoul(argv[2], NULL, 0);

	// Truncate offset to a multiple of the page size, or mmap will fail.
	size_t pagesize = sysconf(_SC_PAGE_SIZE);
	off_t page_base = (offset / pagesize) * pagesize;
	off_t page_offset = offset - page_base;

	int fd = open("/dev/mem", O_SYNC);
	printf("page_base = %x \n", page_base);
	unsigned char *mem = mmap(NULL, page_offset + len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, page_base);
	if (mem == MAP_FAILED) {
		perror("Can't map memory");
		return -1;
	}

	size_t i;
	char *addr = page_base;
	int start_offset = offset - page_base;
	printf("start_offset = 0x%x Mapped size 0x%x\n", start_offset,(page_offset + len));
	for (i = start_offset; i < (page_offset + len); ++i) {
	//	printf("0x%x => %02x \n", addr++, (int)mem[page_offset + i]);
		mem[page_offset + i] = 0xFF;
	}

	return 0;
}
