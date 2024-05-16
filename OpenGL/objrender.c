#include<GL/gl.h>
#include<GL/glut.h>
#include<stdio.h>
#include <string.h>

struct vertex
{
    float x;
    float y;
    float z;
}*vp, *vnp;

struct face
{
    unsigned int pt1v, pt1vt, pt1vn;
    unsigned int pt2v, pt2vt, pt2vn;
    unsigned int pt3v, pt3vt, pt3vn;
}*fp;

unsigned int vn_count = 0, v_count = 0, f_count = 0;
GLuint obj;
float objrot;

int loadfrmfileobj(char *filename)
{
    FILE *fileptr = fopen(filename, "r");
    char ch[5];
    unsigned int length = 0;
    if (fileptr == NULL)
    {
        return -1;
    }

    fseek(fileptr, 0L, SEEK_END);
    length = ftell(fileptr);
    fseek(fileptr, 0L, SEEK_SET);

    while (ftell(fileptr) < length)
    {
        fscanf(fileptr, "%s", ch);

        if (strncmp(ch, "vn", strlen("vn")) == 0)
        {
            vn_count++;
        }else if (strncmp(ch, "v", strlen("v")) == 0)
        {
            v_count++;
        }else if (strncmp(ch, "f", strlen("f")) == 0)
        {
            f_count++;
        }else if (strncmp(ch, "s", strlen("s")) == 0)
        {
        }
    }
    vp = malloc(sizeof(struct vertex) * v_count);
    fp = malloc(sizeof(struct face) * f_count);
    vnp = malloc(sizeof(struct vertex) * vn_count);

    fseek(fileptr, 0L, SEEK_SET);
    vn_count = f_count = v_count = 0;

    char lineread[1024];
    while (ftell(fileptr) < length-2)
    {
        fscanf(fileptr, "%s", ch);

        if ((strncmp(ch, "vn", strlen("vn")) == 0) && (strlen(ch) == 2))
        {
            fscanf(fileptr, "%f %f %f", &vnp[vn_count].x, &vnp[vn_count].y, &vnp[vn_count].z);
            vn_count++;
        }else if ((strncmp(ch, "v", strlen("v")) == 0) && (strlen(ch) == 1))
        {
            fscanf(fileptr, "%f %f %f", &vp[v_count].x, &vp[v_count].y, &vp[v_count].z);
            v_count++;
        }else if ((strncmp(ch, "f", strlen("f")) == 0) && (strlen(ch) == 1))
        {
            fscanf(fileptr, "%s",lineread);
            char* token1 = strtok(lineread, "/");
            char* token2 = strtok(NULL, "/");
            char* token3 = strtok(NULL, "/");
            if (token3 == NULL)
            {
                fp[f_count].pt1v = atoi(token1);
                fp[f_count].pt1vt = 0;
                fp[f_count].pt1vn = atoi(token2);
            }else
            {
                fp[f_count].pt1v = atoi(token1);
                fp[f_count].pt1vt = atoi(token2);
                fp[f_count].pt1vn = atoi(token3);
            }

            fscanf(fileptr, "%s",lineread);
            token1 = strtok(lineread, "/");
            token2 = strtok(NULL, "/");
            token3 = strtok(NULL, "/");
            if (token3 == NULL)
            {
                fp[f_count].pt2v = atoi(token1);
                fp[f_count].pt2vt = 0;
                fp[f_count].pt2vn = atoi(token2);
            }else
            {
                fp[f_count].pt2v = atoi(token1);
                fp[f_count].pt2vt = atoi(token2);
                fp[f_count].pt2vn = atoi(token3);
            }

            fscanf(fileptr, "%s",lineread);
            token1 = strtok(lineread, "/");
            token2 = strtok(NULL, "/");
            token3 = strtok(NULL, "/");
            if (token3 == NULL)
            {
                fp[f_count].pt3v = atoi(token1);
                fp[f_count].pt3vt = 0;
                fp[f_count].pt3vn = atoi(token2);
            }else
            {
                fp[f_count].pt3v = atoi(token1);
                fp[f_count].pt3vt = atoi(token2);
                fp[f_count].pt3vn = atoi(token3);
            }

            f_count++;
        }else if (strncmp(ch, "s", strlen("s")) == 0)
        {
        }
    }
    fclose(fileptr);
    return 0;
}


void reshape(int w, int h)
{    
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60,(GLfloat)w /(GLfloat)h, 0.1, 1000.0);
    glMatrixMode(GL_MODELVIEW);
}

void drawCar()
{
    glPushMatrix();
    glTranslatef(0, -30, -105);
    glColor3f(1.0, 0.23, 0.27);
    glScalef(.1, .1, .1);
    glRotatef(objrot, 0, 1, 0);
    glCallList(obj);
    glPopMatrix();
    objrot = objrot + 0.01;
    if(objrot>360) objrot = objrot-360;
}

void display(void)
{  
    glClearColor(0.0, 0.0, 0.0, 1.0); 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    drawCar();
    glutSwapBuffers();

}
void loadObj()
{
    unsigned int i = 0;
    unsigned int total_face = f_count;
    obj = glGenLists(1);

    glPointSize(2.0);
    glNewList(obj, GL_COMPILE);
    {
        glPushMatrix();
        while (i < total_face-2)
        {
            glBegin(GL_TRIANGLES);
            glColor3f(.0, .5, .0);
            glVertex3f(vp[fp[i].pt1v-1].x, vp[fp[i].pt1v-1].y, vp[fp[i].pt1v-1].z);
            //glColor3f(.5, .0, .0);
            glVertex3f(vp[fp[i].pt2v-1].x, vp[fp[i].pt2v-1].y, vp[fp[i].pt2v-1].z);
            //glColor3f(.0, .0, .5);
            glVertex3f(vp[fp[i].pt3v-1].x, vp[fp[i].pt3v-1].y, vp[fp[i].pt3v-1].z);
            glEnd();
            i++;
        }
    }
    glPopMatrix();
    glEndList();
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
    loadfrmfileobj(argv[1]);
    loadObj();
    glutMainLoop();
    return 0;
}    
