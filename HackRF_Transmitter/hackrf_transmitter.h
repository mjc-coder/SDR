#ifndef HACKRF_TRANSMITTER_H
#define HACKRF_TRANSMITTER_H

#include <QMainWindow>
#include <hackrf/HackRF_radio.h>
#include <QTimer>

QT_BEGIN_NAMESPACE
namespace Ui { class Hackrf_Transmitter; }
QT_END_NAMESPACE

class Hackrf_Transmitter : public QMainWindow
{
    Q_OBJECT

public:
    enum DATA_TYPE
    {
        USER = 0,
        NUMS = 1,
        LETS = 2
    };

public:
    Hackrf_Transmitter(QWidget *parent = nullptr);
    ~Hackrf_Transmitter();

private slots:
    void on_m_rf_gain_clicked();

    void on_m_txfreq_valueChanged(double arg1);

    void on_m_bps_valueChanged(double arg1);

    void on_m_lna_gain_valueChanged(double arg1);

    void on_m_StartTransmit_clicked();

    void updateDataSlot();

    void on_m_encoding_scheme_currentIndexChanged(const QString &arg1);

    void on_m_DataSelection_currentIndexChanged(const QString &arg1);

private:
    Ui::Hackrf_Transmitter *ui;
    HackRF_radio* m_radio;
    hackrf_device_list_t* m_list;
    bool agc;
    double freq;
    double bps;
    double lna;
    HackRF_radio::HACKRF_MODE m_mode;
    QTimer m_qtimer;
    DATA_TYPE m_data_type;
};
#endif // HACKRF_TRANSMITTER_H
