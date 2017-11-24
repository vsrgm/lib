/* Includes */
#include "sdl.h"

struct _sdl_info sdl = {
	.init   = sdl_init,
	.render = sdl_render,
	.exit   = sdl_exit,
};

int sdl_init(int width, int height) 
{
	
	/* Initialize SDL */
	if(SDL_Init(SDL_INIT_VIDEO) != 0) {
		fprintf(stderr,"Could not initialize SDL: %s\n",SDL_GetError());
		return -1;
	}

	/* Open main window */
	sdl.screen = SDL_SetVideoMode(width, height, 0, SDL_HWSURFACE|SDL_DOUBLEBUF);
	if(!sdl.screen) {
		fprintf(stderr,"Could not set video mode: %s\n",SDL_GetError());
		return -1;
	}

        sdl.overlay_ping = SDL_CreateYUVOverlay(width, height, SDL_UYVY_OVERLAY, sdl.screen);//SDL_YUY2_OVERLAY
	if (sdl.overlay_ping) {
		sdl.olay_buf_ping = (unsigned char *) sdl.overlay_ping->pixels[0];
		sdl.drect.x = 0;
		sdl.drect.y = 0;
		sdl.drect.w = sdl.screen->w;
		sdl.drect.h = sdl.screen->h;
	}else {
		printf("Failed to create overlay_ping \n");
		return -1;
	}

        sdl.overlay_pong = SDL_CreateYUVOverlay(width, height, SDL_UYVY_OVERLAY, sdl.screen);//SDL_YUY2_OVERLAY
	if (sdl.overlay_ping) {
		sdl.olay_buf_pong = (unsigned char *) sdl.overlay_pong->pixels[0];
		sdl.drect.x = 0;
		sdl.drect.y = 0;
		sdl.drect.w = sdl.screen->w;
		sdl.drect.h = sdl.screen->h;
	}else {
		printf("Failed to create overlay_ping \n");
		return -1;
	}
	sdl.width = width;
	sdl.height = height;

	return 0;
}

int sdl_render(unsigned char *buf)
{
	static int ping;
	ping = ping?0:1;

	SDL_Event ev;
	if (SDL_PollEvent(&ev)) 
	{
		if(ev.type == SDL_QUIT) {
			sdl_exit();
		
		}		
	}else {
		SDL_LockYUVOverlay(ping?sdl.overlay_ping:sdl.overlay_pong);
		memcpy(ping?sdl.olay_buf_ping:sdl.olay_buf_pong , buf , sdl.width*sdl.height*2);
		SDL_UnlockYUVOverlay(ping?sdl.overlay_ping:sdl.overlay_pong);
		SDL_DisplayYUVOverlay(ping?sdl.overlay_pong:sdl.overlay_ping, &sdl.drect);
	}
	return 0;
}

int sdl_exit()
{
	SDL_FreeYUVOverlay(sdl.overlay_ping);
	SDL_FreeYUVOverlay(sdl.overlay_pong);
	SDL_Quit();	
}

int register_display(int **disp)
{
	int ret;
	int type;
	ret = get_type_info("SDL", &type);
	if (ret < 0) {
		printf("%s %s %d : Failed to obtain get_type_info \n", __FILE__, __func__, __LINE__);
	}
	sdl.type = type;
	*disp = (int *)&sdl;
}
