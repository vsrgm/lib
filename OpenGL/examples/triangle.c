// Programmer: Mihalis Tsoukalos
// Date: Wednesday 04 June 2014
//
// A simple OpenGL program that draws a triangle.

#include "GL/freeglut.h"
#include "GL/gl.h"

void drawTriangle()
{
	//    glClearColor(0.4, 0.4, 0.4, 0.4);
	glClear(GL_COLOR_BUFFER_BIT);

	//    glColor3f(1.0, 1.0, 1.0);
	//    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	glBegin(GL_POLYGON);
	glVertex3f( 0.0f, 0.0f, 0.0f);
	glVertex3f( 100.0f, 0.0f, 0.0f);
	glVertex3f( 100.0f,100.0f, 0.0f);
	glVertex3f(0, 100, 0);
//	glVertex3f(0.7, 0.7, 0);
///	glVertex3f(1, -1, 0);
	glEnd();

	glFlush();
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("OpenGL - Creating a triangle");
	glutDisplayFunc(drawTriangle);
	glutMainLoop();
	return 0;
}
