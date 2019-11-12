#include "hackrf_transmitter.h"
#include "ui_hackrf_transmitter.h"

Hackrf_Transmitter::Hackrf_Transmitter(QWidget *parent)
: QMainWindow(parent)
, ui(new Ui::Hackrf_Transmitter)
, m_radio(nullptr)
, m_list(0)
, agc(false)
, freq(800)
, bps(100000)
, lna(10)
, m_mode(HackRF_radio::FM)
, m_data_type(USER)
{
    ui->setupUi(this);
    // initialize them if you got them
    (void)hackrf_init();
    ui->m_data_message->setText("Hello World");

    m_list = hackrf_device_list();

    if(m_list->devicecount > 0)
    {
        ui->m_radio_list->clear();
    }

    // capture radios
    for(int i = 0; i < m_list->devicecount; i++)
    {
        cout << "[" << i << "] " << hackrf_usb_board_id_name(m_list->usb_board_ids[i]) << "  (" << m_list->serial_numbers[i] << ")" << std::endl;

        QString temp;
        temp += "[";
        temp += QString::number(i);
        temp += "] ";
        temp += hackrf_usb_board_id_name(m_list->usb_board_ids[i]);
        temp += "  (";
        temp += m_list->serial_numbers[i];
        temp += ")";

        ui->m_radio_list->addItem(temp);
        ui->statusbar->showMessage("Radio Found", 1000);
    }
    ui->m_radio_list->setCurrentIndex(0);

    // setup a timer that repeatedly calls MainWindow::realtimeDataSlot:
    m_qtimer.setInterval(200);
    m_qtimer.start();
    connect(&m_qtimer, SIGNAL(timeout()), this, SLOT(updateDataSlot()));
}

Hackrf_Transmitter::~Hackrf_Transmitter()
{
    hackrf_device_list_free(m_list);
    delete m_radio;
    delete ui;
}

void Hackrf_Transmitter::updateDataSlot()
{
    if(!m_radio)
    {
        return;
    }
    if(m_radio->hackrf_found())
    {
        ui->m_RadioFound->setOn(true);
    }
    else
    {
        ui->m_RadioFound->setOn(false);
    }

    if(m_radio->radio_active())
    {
        ui->m_TransmitterActive->setOn(true);
    }
    else
    {
        ui->m_TransmitterActive->setOn(false);
    }

    if(m_radio->hackrf_is_streaming())
    {
        ui->m_is_streaming->setOn(true);
    }
    else
    {
        ui->m_is_streaming->setOn(false);
    }
}

void Hackrf_Transmitter::on_m_rf_gain_clicked()
{
    if(m_radio)
    {
        if(m_radio->get_rf_gain())
        {
            agc = false;
            ui->m_rf_gain->setText("Disabled");
        }
        else
        {
            agc = true;
            ui->m_rf_gain->setText("Enabled");
        }
    }
}


void Hackrf_Transmitter::on_m_txfreq_valueChanged(double arg1)
{
    freq = arg1;
}

void Hackrf_Transmitter::on_m_bps_valueChanged(double arg1)
{
    bps = arg1;
}

void Hackrf_Transmitter::on_m_lna_gain_valueChanged(double arg1)
{
    lna = arg1;
}

void Hackrf_Transmitter::on_m_StartTransmit_clicked()
{
    if(m_radio != nullptr)
    {
        if(m_radio->hackrf_found())
        {

            m_radio->transmit_enabled(false);
            m_radio->set_rf_gain(agc);
            m_radio->set_freq(static_cast<unsigned int>(MHZ_TO_HZ(freq)));
            m_radio->set_baud_rate(static_cast<unsigned long>(bps));
            m_radio->set_txvga_gain(static_cast<unsigned int>(lna));
            m_radio->set_encoding_mode(m_mode);
            m_radio->set_data_message((ui->m_data_message->toPlainText()).toUtf8().constData(), m_data_type);
            m_radio->transmit_enabled(true);
        }
    }
    else
    {
        ui->m_StartTransmit->setText("Configure Transmitter");
        m_radio = new HackRF_radio(m_list, ui->m_radio_list->currentIndex());

        if(m_radio)
        {
            m_radio->set_rf_gain(agc);
            m_radio->set_freq(static_cast<unsigned int>(MHZ_TO_HZ(freq)));
            m_radio->set_baud_rate(static_cast<unsigned long>(bps));
            m_radio->set_txvga_gain(static_cast<unsigned int>(lna));
            m_radio->set_encoding_mode(m_mode);
            m_radio->set_data_message((ui->m_data_message->toPlainText()).toUtf8().constData(), m_data_type);
            m_radio->transmit_enabled(true);
        }
    }
}



void Hackrf_Transmitter::on_m_encoding_scheme_currentIndexChanged(const QString &arg1)
{
    if(arg1 == "All Ones")
    {
        m_mode = HackRF_radio::ONES;
    }
    else if(arg1 == "All Zeros")
    {
        m_mode = HackRF_radio::ZEROS;
    }
    else if(arg1 == "Tone@50000Hz")
    {
        m_mode = HackRF_radio::TONE50000HZ;
    }
    else if(arg1 == "Tone@10000Hz")
    {
        m_mode = HackRF_radio::TONE10000HZ;
    }
    else if(arg1 == "Tone@20000Hz")
    {
        m_mode = HackRF_radio::TONE20000HZ;
    }
    else if(arg1 == "AM")
    {
        m_mode = HackRF_radio::AM;
    }
    else if(arg1 == "FM")
    {
        m_mode = HackRF_radio::FM;
    }
    else if(arg1 == "BPSK")
    {
        m_mode = HackRF_radio::BPSK;
    }
    else if(arg1 == "QPSK")
    {
        m_mode = HackRF_radio::QPSK;
    }
    else
    {
        m_mode = HackRF_radio::ZEROS;
    }
}



void Hackrf_Transmitter::on_m_DataSelection_currentIndexChanged(const QString &arg1)
{
    if(arg1 == "User Data")
    {
        m_data_type = USER;
    }
    else if(arg1 == "Incrementing Numbers")
    {
        m_data_type = NUMS;
    }
    else if(arg1 == "Incrementing Letters")
    {
        m_data_type = LETS;
    }
}
