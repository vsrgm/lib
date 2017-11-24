#include <SDL.h>

int sdl_init(int width, int height);
int sdl_render(unsigned char *buf);
int sdl_exit(void);

struct _sdl_info {
	int type;
	SDL_Surface *screen;
	SDL_Overlay *overlay_ping;
	SDL_Overlay *overlay_pong;
	SDL_Rect drect;
	unsigned char *olay_buf_ping;
	unsigned char *olay_buf_pong;

	int width;
	int height;

	int (*init) (int width, int height);
	int (*render) (unsigned char *buf);
	int (*exit) (void);
};
