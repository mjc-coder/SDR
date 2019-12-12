#include "hackrf_transmitter.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Hackrf_Transmitter w;
    w.show();
    return a.exec();
}
