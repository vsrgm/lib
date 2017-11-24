/* Includes */
#include <SDL.h>

/* Globals */
SDL_Surface *demo_screen;

/* Main */
int main(int argn,char **argv)
{
	SDL_Event ev;
	int active;
	SDL_Overlay* overlay=NULL;
	SDL_Overlay* overlay1=NULL;
	SDL_Overlay* overlay2=NULL;
	unsigned char *p = NULL;
	unsigned char a=0;
	SDL_Rect drect;
	int read_info;

	unsigned char *buffer;
	static struct timeval cur_tv, prev_tv;
	static int old_stream_count, stream_count, fps;
	int width = atoi(argv[1]);
	int height = atoi(argv[2]);

	buffer = malloc(width * height *2);

	/* Initialize SDL */
	if(SDL_Init(SDL_INIT_VIDEO) != 0)
		fprintf(stderr,"Could not initialize SDL: %s\n",SDL_GetError());
	/* Open main window */
	demo_screen = SDL_SetVideoMode(width,height,0,SDL_HWSURFACE|SDL_DOUBLEBUF);
	if(!demo_screen)
		fprintf(stderr,"Could not set video mode: %s\n",SDL_GetError());

        overlay = SDL_CreateYUVOverlay(width, height, SDL_YUY2_OVERLAY, demo_screen);
        overlay1 = SDL_CreateYUVOverlay(width, height, SDL_YUY2_OVERLAY, demo_screen);
        overlay2 = SDL_CreateYUVOverlay(width, height, SDL_YUY2_OVERLAY, demo_screen);
	if (overlay) {
		p = (unsigned char *) overlay->pixels[0];
		drect.x = 0;
		drect.y = 0;
		drect.w = demo_screen->w;
		drect.h = demo_screen->h;
	}else {
		printf("Failed to create overlay\n");
	}
	p = (unsigned char *) overlay->pixels[0];
	memset(p,0x00, width*height*2);
	p = (unsigned char *) overlay1->pixels[0];
	memset(p,0x80, width*height*2);
	p = (unsigned char *) overlay2->pixels[0];
	memset(p,0xA0, width*height*2);

	/* Main loop */
	active = 1;
	while(active)
	{
		gettimeofday(&cur_tv, NULL);
		printf("PATTERN 1 %06d.%06d\n",cur_tv.tv_sec,cur_tv.tv_usec);
//		SDL_DisplayYUVOverlay(overlay, &drect);

		gettimeofday(&cur_tv, NULL);
		printf("PATTERN 2 %06d.%06d\n",cur_tv.tv_sec,cur_tv.tv_usec);
    		SDL_DisplayYUVOverlay(overlay1, &drect);

		gettimeofday(&cur_tv, NULL);
		printf("PATTERN 3 %06d.%06d\n",cur_tv.tv_sec,cur_tv.tv_usec);
//    		SDL_DisplayYUVOverlay(overlay2, &drect);

		gettimeofday(&cur_tv, NULL);
		if(cur_tv.tv_sec > prev_tv.tv_sec) {
			prev_tv	= cur_tv;
			fps	= stream_count - old_stream_count;
			old_stream_count = stream_count;
			printf("fps %06d count %06d \n", fps, stream_count);
		}
		stream_count++;

		/* Handle events */
		while(SDL_PollEvent(&ev))
		{
			if(ev.type == SDL_QUIT) {
				printf("Going to Quit\n");
				active = 0; /* End */
			}
		}
	}
	printf("fps %06d count %06d \n", fps, stream_count);
	SDL_FreeYUVOverlay(overlay);
	SDL_FreeYUVOverlay(overlay1);
	SDL_FreeYUVOverlay(overlay2);
	/* Exit */
	SDL_Quit();
	return 0;
}
