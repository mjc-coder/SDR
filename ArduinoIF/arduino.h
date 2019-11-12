#ifndef ARDUINO_H
#define ARDUINO_H

#include <QMainWindow>
#include "../RadioNode/src/libraries/arduino_if/Arduino_IF.h"
#include <QTimer>

QT_BEGIN_NAMESPACE
namespace Ui { class Arduino; }
QT_END_NAMESPACE

class Arduino : public QMainWindow
{
    Q_OBJECT

public:
    Arduino(QWidget *parent = nullptr);
    ~Arduino();

    void update();

private:
    Ui::Arduino *ui;
    Arduino_IF m_arduino;
    QTimer timer;
};
#endif // ARDUINO_H
