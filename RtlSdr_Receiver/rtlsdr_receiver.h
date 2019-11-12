#ifndef RTLSDR_RECEIVER_H
#define RTLSDR_RECEIVER_H

#include <QMainWindow>
#include <rtl_sdr/rtlsdr_radio.h>
#include <QVector>
#include <QMutex>
#include <QTimer>
#include <qcustomplot.h>
#include <streams/AM.h>
#include <streams/FM.h>
#include <streams/BPSK.h>
#include <streams/QPSK.h>
#include <common/confidence_counter.h>
#include <network/PacketFramer.h>
#include <common/RingBuffer.h>
#include <hackrf/HackRF_radio.h>
#include <DigitalSignalProcessing/LowPassFilter.h>
#include <QUdpSocket>
#include <cpu_usage.h>
#include <system_usage.h>

#define MAX_NUM_RADOIOS 6

QT_BEGIN_NAMESPACE
namespace Ui { class RtlSdr_Receiver; }
QT_END_NAMESPACE

class RtlSdr_Receiver : public QMainWindow
{
    Q_OBJECT
public:
    enum PLOT_VIEW
    {
      TIME_REAL = 0,
      TIME_IMAG = 1,
      CONST = 2,
      DEBUG = 3,
      TEMPERATURE = 4,
      POWER = 5,
      PROC_USAGE = 6,
      GPU = 7,
      RAM_USAGE = 8
    };

    static uint64_t unique_word[];
    static uint64_t bpsk_unique_word[];
public:
    RtlSdr_Receiver(QWidget *parent = nullptr);
    ~RtlSdr_Receiver();

    static void callbackRoutine(unsigned char*, uint32_t, int RADIO_ID);

private slots:
    void on_pushButton_clicked();

    void updateDataSlot();

    void ShowContextMenu(const QPoint &pos);

    void StatusUpdateSlot();

    void bpsUpdateSlot();

    void updateDataMsg(bool validPacket, bool validCRC, bool validHeader, int bad_bits, int good_bits);

    void UpdatePlots();

    void radio_search();
    
    void on_pushButton_2_clicked();

    void processPendingSystemMonitorDatagrams();

    void demodulateData(int RADIO_ID);

    void deserializeData(int RADIO_ID);

    void LogData();

    void PerformanceTests(int TestID);

    void FinishTest();
signals:
    void messageReceived(bool validPacket, bool validCRC, bool validHeader, int bad_bits, int good_bits);

    void updatePlotsSingal();

    void triggerDemodulation(int RADIO_ID);

    void triggerDeserialize(int RADIO_ID);

    void PerformanceTestSignal(int TestID);

private:
    void ConfigureRadio(int RADIO_ID);

    void TearDownRadios();

    void CreateRadio(int RADIO_ID);

private:
     void setMode();

     void Initialize();

     static void AM_OverProcess_Callback(size_t thread_id, size_t samples_per_bit, int RADIO_ID);
    
private:
    Ui::RtlSdr_Receiver *ui;
    Radio_RTLSDR* m_radio_rtlsdr[MAX_NUM_RADOIOS];
    size_t m_RefreshRate;
    size_t m_SampleRate;
    size_t m_NumPoints;
    size_t m_decimation;
    RADIO_DATA_TYPE m_delta_t;
    RADIO_DATA_TYPE m_delta_f;
    bool m_check_the_crc;
    size_t m_num_validCRCs;
    size_t m_num_validHeaders;
    size_t m_bad_bits;
    size_t m_good_bits;
    bool m_cpu_vs_gpu;
    size_t m_N_Overprocess;

    // Q Objects
    QTimer m_qtimer;
    QTimer m_logtimer;
    QTimer m_bpstimer;
    QTimer m_performanceTest;

    // Standard Types
    uint32_t m_block_index[MAX_NUM_RADOIOS];
    bool auto_gain;
    size_t plotVar;
    bool ready;
    RADIO_DATA_TYPE m_VCO_time;
    bool m_Normalize;
    bool m_ABS;
    bool m_AM;
    RADIO_DATA_TYPE m_rx_time;
    bool m_DisablePlots;
    size_t m_prev_pkt_cnt;
    size_t m_pkt_cnt;
    size_t m_actual_samples_per_second;
    size_t m_actual_bits_per_second;
    Radio_RTLSDR::RTLSDR_MODE m_mode;
    size_t m_bps;
    bool m_validPacket;
    bool m_validCRC;
    PLOT_VIEW m_left_plot;
    PLOT_VIEW m_right_plot;


    // Plot Variables (Pointers)
    QVector<RADIO_DATA_TYPE>* m_plot_td_vector_x;
    QVector<RADIO_DATA_TYPE>* m_plot_td_vector_y_real;
    QVector<RADIO_DATA_TYPE>* m_plot_td_vector_y_imag;
    RADIO_DATA_TYPE* m_temp_real[MAX_NUM_RADOIOS];
    RADIO_DATA_TYPE* m_temp_imag[MAX_NUM_RADOIOS];

    // Demodulation data
    QVector<RADIO_DATA_TYPE>* m_block_x[MAX_NUM_RADOIOS];
    QVector<RADIO_DATA_TYPE>* m_block_y[MAX_NUM_RADOIOS];
    QVector<RADIO_DATA_TYPE>* m_block_demod_x[MAX_NUM_RADOIOS];
    QVector<RADIO_DATA_TYPE>* m_block_demod_y[MAX_NUM_RADOIOS];
    QVector<RADIO_DATA_TYPE>* m_block_output_x[MAX_NUM_RADOIOS];
    QVector<RADIO_DATA_TYPE>* m_block_output_y[MAX_NUM_RADOIOS];
    uint8_t* m_data_buffer[MAX_NUM_RADOIOS];
    RingBuffer<uint8_t>* m_ring_buffer[MAX_NUM_RADOIOS];
    uint8_t* m_pkt[MAX_NUM_RADOIOS];
    uint8_t* m_temp_pkt[MAX_NUM_RADOIOS];

    // QCP Axis
    QCPAxisRect* leftAxisRect;
    QCPAxisRect* rightAxisRect;
    QCPLegend *left_arLegend;
    QCPLegend *right_arLegend;

    confidence_counter* m_ones_zeros[MAX_NUM_RADOIOS];
    confidence_counter* m_packet_confidence[MAX_NUM_RADOIOS];
    PacketFramer<uint8_t>* m_framer[MAX_NUM_RADOIOS];
    ::AM<RADIO_DATA_TYPE>* m_am[MAX_NUM_RADOIOS];
    ::FM<RADIO_DATA_TYPE>* m_fm[MAX_NUM_RADOIOS];
    ::BPSK* m_bpsk[MAX_NUM_RADOIOS];
    ::QPSK* m_qpsk[MAX_NUM_RADOIOS];


    // multicast receiver
    enum VectorIndexes
    {
        RTL0 = 0,
        RTL1,
        RTL2,
        RTL3,
        RTL4,
        RTL5,
        HackRF,
        SystemController,
        MAX
    };
    QVector<double>* m_plot_temps[VectorIndexes::MAX];
    QVector<double>* m_plot_currents[VectorIndexes::MAX];
    QVector<double>* m_plot_xvector;
    QUdpSocket udpSocket4;
    QHostAddress groupAddress4;
#ifndef GPU_ENABLED
    CPU_Usage m_linux_usage;
#else
    System_Usage m_nvidia_usage;
#endif

    // threads last
    std::mutex m_block_lock[MAX_NUM_RADOIOS];
    std::fstream fout;
    std::fstream debug;
    uint32_t m_num_activate_Radios;
    LowPassFilter filter_Real;
    LowPassFilter filter_Imag;
    size_t currentTestID;
    size_t MaxTestID;
    double Y_Val_Real;
    double Y_Val_Imag;
    double alpha;
};


#endif // RTLSDR_RECEIVER_H
