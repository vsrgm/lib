#include <QApplication>
#include "convert.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    convert w;
    w.show();
    return a.exec();
}
