#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL)
    {
        printf("File not found\n");
        exit(0);
    }

    fseek(fp, 0L, SEEK_END);
    unsigned int length = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    unsigned int ch;
    while (1)
    {
        fscanf(fp, "0x%x ", &ch);
        printf("%c", (char) ch);
        if (ftell(fp) == length)
            break;
    }

    return 0;
}

