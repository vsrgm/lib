#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setWindowTitle("IoT Home Control Center (x86 Desktop)");
    w.resize(1280, 800);
    w.show();
    return a.exec();
}
