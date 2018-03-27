//header

#include<GL/gl.h>
#include<GL/glut.h>
#include<stdio.h>

//globals
struct {
	float x;
	float y;
	float z;
}vertex[] = {
	#include "vertex.txt"
};

struct {
	float x;
	float y;
	float z;
}vertexn[] = {
	#include "vertexnormal.txt"
};

struct {
	unsigned int pt1v, pt1vt, pt1vn;
	unsigned int pt2v, pt2vt, pt2vn;
	unsigned int pt3v, pt3vt, pt3vn;
}face[] = {
	#include "face.txt"
};
	

GLuint elephant;
float elephantrot;
char ch = '1';

//other functions and main

//.obj loader code begins

void loadObj(char *fname)
{
	FILE *fp;
	int read;
	GLfloat x, y, z;
	char ch;
	unsigned int i = 0, total_vertex = (sizeof(vertex)/sizeof(vertex[0]));
	unsigned int total_face = (sizeof(face)/sizeof(face[0]));
	elephant = glGenLists(1);
	fp = fopen(fname, "r");
	if(!fp) 
	{
		printf("can't open file %s\n", fname);
		exit(1);
	}
	glPointSize(2.0);
	glNewList(elephant, GL_COMPILE);
	{
		glPushMatrix();
#if 0
		glBegin(GL_POINTS);
#if 0
		while(!(feof(fp)))
		{
			read = fscanf(fp, "%c %f %f %f", &ch, &x, &y, &z);
			if(read == 4&&ch == 'v')
			{
				glVertex3f(x, y, z);
			}
		}
#else
		printf("total_vertex = %d \n", total_vertex);
		while (i < total_vertex) {
			printf("%f %f %f \n",vertex[i].x, vertex[i].y, vertex[i].z);
			glVertex3f(vertex[i].x, vertex[i].y, vertex[i].z);
			i++;
		}
			
#endif
		glEnd();
#else
	    while (i < total_face-2) {
		    // printf("Triangle count %d \n", i);
		    glBegin(GL_LINES);
		    //glBegin(GL_TRIANGLES);
		    glColor3f(.5, .5, .5);
		    glVertex3f(vertex[face[i].pt1v-1].x, vertex[face[i].pt1v-1].y, vertex[face[i].pt1v-1].z);
		    //glNormal3f(vertexn[face[i].pt1vn-1].x, vertexn[face[i].pt1vn-1].y, vertexn[face[i].pt1vn-1].z);
		    glVertex3f(vertex[face[i].pt2v-1].x, vertex[face[i].pt2v-1].y, vertex[face[i].pt2v-1].z);
		    glEnd();

		    glBegin(GL_LINES);
		    glVertex3f(vertex[face[i].pt2v-1].x, vertex[face[i].pt2v-1].y, vertex[face[i].pt2v-1].z);
		    // glVertex3f(vertexn[face[i].pt2vn-1].x, vertexn[face[i].pt2vn-1].y, vertexn[face[i].pt2vn-1].z);

		    glVertex3f(vertex[face[i].pt3v-1].x, vertex[face[i].pt3v-1].y, vertex[face[i].pt3v-1].z);
		    // glVertex3f(vertexn[face[i].pt3vn-1].x, vertexn[face[i].pt3vn-1].y, vertexn[face[i].pt3vn-1].z);
		    glEnd();

		    glBegin(GL_LINES);
		    glVertex3f(vertex[face[i].pt1v-1].x, vertex[face[i].pt1v-1].y, vertex[face[i].pt1v-1].z);
		    glVertex3f(vertex[face[i].pt3v-1].x, vertex[face[i].pt3v-1].y, vertex[face[i].pt3v-1].z);
		    glEnd();
		    i++;
	    }
#endif
	}
	glPopMatrix();
	glEndList();
	fclose(fp);
}

//.obj loader code ends here

void reshape(int w, int h)
{    
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60,(GLfloat)w /(GLfloat)h, 0.1, 1000.0);
	//glOrtho(-25, 25, -2, 2, 0.1, 100);	
	glMatrixMode(GL_MODELVIEW);
}

void drawCar()
{
	glPushMatrix();
	glTranslatef(0, -40.00, -105);
	glColor3f(1.0, 0.23, 0.27);
	glScalef(0.1, 0.1, 0.1);
	glRotatef(elephantrot, 0, 1, 0);
	glCallList(elephant);
	glPopMatrix();
	elephantrot = elephantrot+0.6;
	if(elephantrot>360)elephantrot = elephantrot-360;
}

void display(void)
{  
	glClearColor(0.0, 0.0, 0.0, 1.0); 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	drawCar();
	glutSwapBuffers(); //swap the buffers

}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);
	glutInitWindowSize(800, 450);
	glutInitWindowPosition(20, 20);
	glutCreateWindow("ObjLoader");
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutIdleFunc(display);
	loadObj("data/elepham.obj");//replace porsche.obj with radar.obj or any other .obj to display it
	glutMainLoop();
	return 0;
}
