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
    unsigned char ch;
    unsigned int count = 0;
    while (1)
    {
        fscanf(fp, "%c", &ch);
        printf("0x%x ", ch);
        count++;
        if (count%20 == 0)
            printf("\n");

        if (ftell(fp) == length)
            break;
    }

    return 0;
}

