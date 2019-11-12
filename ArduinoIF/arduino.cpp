#include "arduino.h"
#include "ui_arduino.h"

Arduino::Arduino(QWidget *parent)
: QMainWindow(parent)
, ui(new Ui::Arduino)
, m_arduino("/dev/cu.usbmodem1442301")
, timer(this)
{
    ui->setupUi(this);
    timer.setInterval(static_cast<int>(250));
    timer.start();
    connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
}

Arduino::~Arduino()
{
    delete ui;
}


void Arduino::update()
{
    ui->m_analog_0->setText(QString::number(m_arduino.Analog_0()));
    ui->m_analog_1->setText(QString::number(m_arduino.Analog_1()));
    ui->m_analog_2->setText(QString::number(m_arduino.Analog_2()));
    ui->m_analog_3->setText(QString::number(m_arduino.Analog_3()));
    ui->m_analog_4->setText(QString::number(m_arduino.Analog_4()));
    ui->m_analog_5->setText(QString::number(m_arduino.Analog_5()));
}
