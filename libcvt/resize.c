#include <stdio.h>
#include <stdlib.h>

void resize_fast(int* input, int* output, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight) 
{    
    int x, y, index;
    float x_ratio = ((float)(sourceWidth - 1)) / targetWidth;
    float y_ratio = ((float)(sourceHeight - 1)) / targetHeight;
    int offset = 0 ;
    int i, j;

    for (i = 0; i < targetHeight; i++) 
    {
        for (j = 0; j < targetWidth/2; j++) 
        {
            x = (int)(x_ratio * j) ;
            y = (int)(y_ratio * i) ;
            index = (y * sourceWidth/2 + x) ;                
            output [offset++] = input[index];    
        }
    }
}

void resize(int* input, int* output, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight) 
{    
    int a, b, c, d, x, y, index;
    float x_ratio = ((float)(sourceWidth - 1)) / targetWidth;
    float y_ratio = ((float)(sourceHeight - 1)) / targetHeight;
    float x_diff, y_diff, Y1, U, Y2, V;
    int offset = 0 ;
    int i, j;

    for (i = 0; i < targetHeight; i++) 
    {
        for (j = 0; j < targetWidth/2; j++) 
        {
            x = (int)(x_ratio * j) ;
            y = (int)(y_ratio * i) ;
            x_diff = (x_ratio * j) - x ;
            y_diff = (y_ratio * i) - y ;
            index = (y * sourceWidth/2 + x) ;                
            a = input[index] ;
            b = input[index + 1] ;
            c = input[index + sourceWidth/2] ;
            d = input[index + sourceWidth/2 + 1] ;

            // Y element
            Y1 = (a&0xff)*(1-x_diff)*(1-y_diff) + (b&0xff)*(x_diff)*(1-y_diff) +
                   (c&0xff)*(y_diff)*(1-x_diff)   + (d&0xff)*(x_diff*y_diff);

            // U element
            U = ((a>>8)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>8)&0xff)*(x_diff)*(1-y_diff) +
                    ((c>>8)&0xff)*(y_diff)*(1-x_diff)   + ((d>>8)&0xff)*(x_diff*y_diff);

            // Y element
            Y2 = ((a>>16)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>16)&0xff)*(x_diff)*(1-y_diff) +
                  ((c>>16)&0xff)*(y_diff)*(1-x_diff)   + ((d>>16)&0xff)*(x_diff*y_diff);

            // V element
            V = ((a>>24)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>24)&0xff)*(x_diff)*(1-y_diff) +
                  ((c>>24)&0xff)*(y_diff)*(1-x_diff)   + ((d>>24)&0xff)*(x_diff*y_diff);

            output [offset++] = 
                    ((int)Y1) & 0x000000ff | 
                    ((((int)V)   << 24)&0xff000000) |
                    ((((int)Y2) << 16)&0xff0000) |
                    ((((int)U)  << 8)&0xff00);
		    
        }
    }
}
