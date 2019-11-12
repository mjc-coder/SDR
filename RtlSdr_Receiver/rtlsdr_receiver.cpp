#include "rtlsdr_receiver.h"
#include "ui_rtlsdr_receiver.h"
#include <RTLSDR/include/rtl-sdr.h>
#include <RTLSDR/include/rtl-sdr_export.h>
#include <DigitalSignalProcessing/FFT.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <arpa/inet.h>
using namespace std;

#define REFRESH_RATE            1000.0
#define SAMPLE_RATE             2000000.0
#define BUFFER_SIZE             SAMPLE_RATE/2
#define SCALE_FACTOR            2.0
#define NUM_POINTS              10000
#define DELTA_T                 1.0/SAMPLE_RATE
#define DECIMATE_FACTOR         10000
#define DELTA_F                 SAMPLE_RATE / (512000/DECIMATE_FACTOR * 2)
#define SIGNAL_FREQ             40000.0 * 1/(2.0 * PI)
#define MAX_PKT_SIZE            13000


#define OVERBUFFER 10


const char* LOGHEADER =  "Timestamp," \
                        "Num Active Radios,"\
                        "Block Multiplier,"\
                        "Demod HW,"\
                        "Receiver Freq,"\
                        "BPS,"\
                        "Gain,"\
                        "Decoding Scheme,"\
                        "Check CRC,"\
                        "Num Overprocess,"\
                        "Modem Lock,"\
                        "CRC Lock,"\
                        "Headers Received,"\
                        "Valid CRCs,"\
                        "Samples Per Second,"\
                        "Num. Valid Pkts,"\
                        "Total Bits,"\
                        "Error Bits,"\
                        "BER,"\
                        "Valid Bits,"\
                        "CPU Avg,"\
                        "CPU0,"\
                        "CPU1,"\
                        "CPU2,"\
                        "CPU3,"\
                        "CPU4,"\
                        "CPU5,"\
                        "Temp R0,"\
                        "Temp R1,"\
                        "Temp R2,"\
                        "Temp R3,"\
                        "Temp R4,"\
                        "Temp R5,"\
                        "Temp HackRF,"\
                        "Ram,"\
                        "GPU,";


enum PLOT_ENUM
{
    LEFT_PLOT_0  = 0,
    LEFT_PLOT_1,
    LEFT_PLOT_2,
    LEFT_PLOT_3,
    LEFT_PLOT_4,
    LEFT_PLOT_5,
    LEFT_PLOT_6,
    RIGHT_PLOT_0,
    RIGHT_PLOT_1,
    RIGHT_PLOT_2,
    RIGHT_PLOT_3,
    RIGHT_PLOT_4,
    RIGHT_PLOT_5,
    RIGHT_PLOT_6
};


#ifdef GPU_ENABLED
    extern "C"
    size_t am_gpu_demodulation(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, uint8_t* output, size_t number_of_points, size_t ds);

    extern "C"
    int32_t ones_zeros_demodulation(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, size_t number_of_points);
#endif

static RtlSdr_Receiver* m_GlobalPtr;

void RtlSdr_Receiver::AM_OverProcess_Callback(size_t thread_id, size_t samples_per_bit, int RADIO_ID)
{
    RADIO_DATA_TYPE* block_x = new RADIO_DATA_TYPE[static_cast<int>(m_GlobalPtr->m_SampleRate)];
    RADIO_DATA_TYPE* block_y = new RADIO_DATA_TYPE[static_cast<int>(m_GlobalPtr->m_SampleRate)];
    uint8_t* data_buffer = new uint8_t[static_cast<unsigned long>(BUFFER_SIZE)];
    memset(data_buffer, 0, static_cast<unsigned long>(BUFFER_SIZE));
    size_t read_number_of_bits = 0;

    m_GlobalPtr->m_block_lock[RADIO_ID].lock();
    // load data
    memcpy(block_x, m_GlobalPtr->m_block_x[RADIO_ID]->data(), m_GlobalPtr->m_SampleRate*sizeof(RADIO_DATA_TYPE));
    memcpy(block_y, m_GlobalPtr->m_block_y[RADIO_ID]->data(), m_GlobalPtr->m_SampleRate*sizeof(RADIO_DATA_TYPE));


    m_GlobalPtr->m_block_lock[RADIO_ID].unlock();


    if(!m_GlobalPtr->m_cpu_vs_gpu) // CPU demodulation
    {
        read_number_of_bits = m_GlobalPtr->m_am[RADIO_ID]->demodulate(block_x, block_y, data_buffer, m_GlobalPtr->m_SampleRate/2, samples_per_bit, m_GlobalPtr->m_bps);
    }
    else
    {
#ifdef GPU_ENABLED
        read_number_of_bits = am_gpu_demodulation(block_x, block_y, data_buffer, m_GlobalPtr->m_SampleRate/2, samples_per_bit);
#endif
    }

    if(thread_id == 0) // only save the data for the first thread
    {
        size_t numbits_added = m_GlobalPtr->m_ring_buffer[RADIO_ID]->append(m_GlobalPtr->m_data_buffer[RADIO_ID], read_number_of_bits);
        if(numbits_added != read_number_of_bits)
        {
            std::cout << "WARNING WERE SAMPLING MORE THAN WE CAN HANDLE. LOSING ALOT OF BITS: " << read_number_of_bits - numbits_added << std::endl;
        }
    }

    delete[] block_x;
    delete[] block_y;
    delete[] data_buffer;
}






RtlSdr_Receiver::RtlSdr_Receiver(QWidget *parent)
: QMainWindow(parent)
, ui(new Ui::RtlSdr_Receiver)
, m_RefreshRate(500)
, m_SampleRate(SAMPLE_RATE*OVERBUFFER)
, m_NumPoints(NUM_POINTS)
, m_decimation(200)
, m_delta_t(0)
, m_delta_f(0)
, m_check_the_crc(false)
, m_num_validCRCs(0)
, m_num_validHeaders(0)
, m_bad_bits(0)
, m_good_bits(0)
, m_cpu_vs_gpu(false)
, m_N_Overprocess(1)
, auto_gain(false)
, plotVar(0)
, ready(false)
, m_VCO_time(0)
, m_Normalize(false)
, m_ABS(false)
, m_AM(false)
, m_rx_time(0)
, m_DisablePlots(true)
, m_prev_pkt_cnt(0)
, m_pkt_cnt(0)
, m_actual_samples_per_second(0)
, m_actual_bits_per_second(0)
, m_mode(Radio_RTLSDR::ZEROS)
, m_bps(100000)
, m_validPacket(false)
, m_validCRC(false)
, m_left_plot(TIME_REAL)
, m_right_plot(TIME_IMAG)
, m_plot_td_vector_x(nullptr)
, m_plot_td_vector_y_real(nullptr)
, m_plot_td_vector_y_imag(nullptr)
, leftAxisRect(nullptr)
, rightAxisRect(nullptr)
, left_arLegend(nullptr)
, right_arLegend(nullptr)
, groupAddress4(QStringLiteral("239.255.43.21")) // hardcode multicast on both ends
, fout("/tmp/rtlsdr_logfile.csv", ios::out | ios::trunc)
, debug("rtlsdr.csv", ios::out | ios::trunc)
, m_num_activate_Radios(0)
, filter_Real(2000, 1.0/2000000.0)
, filter_Imag(2000, 1.0/2000000.0)
, currentTestID(0)
, MaxTestID(2*rtlsdr_get_device_count())
, Y_Val_Real(0)
, Y_Val_Imag(0)
{
    for(int i = 0; i < MAX_NUM_RADOIOS; i++)
    {
        m_radio_rtlsdr[i] = nullptr;
    }
    ui->setupUi(this);
    m_GlobalPtr = this;

    fout << LOGHEADER << std::endl;

    radio_search();
}

void RtlSdr_Receiver::Initialize()
{

    std::cout << "Initializing RTLSDR " << std::endl;
    // sample rate = 2014000 / 2 for i/q's
    m_SampleRate = static_cast<size_t>(1000000.0/(m_RefreshRate/1000.0)*ui->m_sample_block_multiplier->value());  // number of points that we actually get per second time * Oversample amount
    m_delta_t = (m_RefreshRate/1000.0)/static_cast<RADIO_DATA_TYPE>(m_SampleRate);
    m_delta_f = static_cast<RADIO_DATA_TYPE>(m_SampleRate) / static_cast<RADIO_DATA_TYPE>(m_NumPoints);
    m_decimation = m_SampleRate/m_NumPoints;
    ui->m_gain->setRange(0, 47);
    std::cout << "...." << m_SampleRate << "  " << m_NumPoints << std::endl;


    // Allocate all of the news
    try
    {
        // Plot values
        m_plot_td_vector_x = new QVector<RADIO_DATA_TYPE>(static_cast<int>(m_NumPoints));
        m_plot_td_vector_y_real = new QVector<RADIO_DATA_TYPE>(static_cast<int>(m_NumPoints));
        m_plot_td_vector_y_imag = new QVector<RADIO_DATA_TYPE>(static_cast<int>(m_NumPoints));

        // data values
        for(int i = 0; i < MAX_NUM_RADOIOS; i++)
        {
            m_block_index[i] = 0;
            m_temp_real[i] = new RADIO_DATA_TYPE[m_SampleRate];
            m_temp_imag[i] = new RADIO_DATA_TYPE[m_SampleRate];
            m_block_x[i] = new QVector<RADIO_DATA_TYPE>(static_cast<int>(m_SampleRate));
            m_block_y[i] = new QVector<RADIO_DATA_TYPE>(static_cast<int>(m_SampleRate));
            m_block_demod_x[i] = new QVector<RADIO_DATA_TYPE>(static_cast<int>(m_SampleRate));
            m_block_demod_y[i] = new QVector<RADIO_DATA_TYPE>(static_cast<int>(m_SampleRate));
            m_block_output_x[i] = new QVector<RADIO_DATA_TYPE>(static_cast<int>(m_SampleRate));
            m_block_output_y[i] = new QVector<RADIO_DATA_TYPE>(static_cast<int>(m_SampleRate));

            // packets and rings
            m_data_buffer[i] = new uint8_t[static_cast<unsigned long>(BUFFER_SIZE)];
            memset(m_data_buffer[i], 0, static_cast<unsigned long>(BUFFER_SIZE));
            m_ring_buffer[i] = new RingBuffer<uint8_t>(static_cast<int>(m_SampleRate*2));
            m_pkt[i] = new uint8_t[MAX_PKT_SIZE];
            memset(m_pkt[i], 0, MAX_PKT_SIZE);
            m_temp_pkt[i] = new uint8_t[MAX_PKT_SIZE];
            memset(m_pkt[i], 0, MAX_PKT_SIZE);

            m_ones_zeros[i] = new confidence_counter(500000, 50000, 400000);
            m_packet_confidence[i] = new confidence_counter(10, 3, 7);
            m_framer[i] = new PacketFramer<uint8_t>();
            m_am[i] = new ::AM<RADIO_DATA_TYPE>();
            m_fm[i] = new ::FM<RADIO_DATA_TYPE>(10000000,2000000);
            m_bpsk[i] = new ::BPSK(0xDEADBEEF600DF00D, 64);
            m_qpsk[i] = new ::QPSK();
        }

        cout << "Memory allocation Successful" << endl;
    }
    catch (const bad_alloc& e)
    {
        cout << "Allocation failed: " << e.what() << '\n';
        //handle error
    }

    // configure the Plots
    ui->m_CustomPlot->plotLayout()->clear(); // let's start from scratch and remove the default axis rect
    // add the first axis rect in second row (row index 1):
    leftAxisRect = new QCPAxisRect(ui->m_CustomPlot);
    left_arLegend = new QCPLegend;
    leftAxisRect->insetLayout()->addElement(left_arLegend, Qt::AlignTop|Qt::AlignRight);
    left_arLegend->setLayer("legend");


    rightAxisRect = new QCPAxisRect(ui->m_CustomPlot);
    right_arLegend = new QCPLegend;
    rightAxisRect->insetLayout()->addElement(right_arLegend, Qt::AlignTop|Qt::AlignRight);
    right_arLegend->setLayer("legend");
    ui->m_CustomPlot->setAutoAddPlottableToLegend(false);

    ui->m_CustomPlot->plotLayout()->addElement(0, 0, leftAxisRect);
    ui->m_CustomPlot->plotLayout()->addElement(0, 1, rightAxisRect);

    // since we've created the axis rects and axes from scratch, we need to place them on
    // according layers, if we don't want the grid to be drawn above the axes etc.
    // place the axis on "axes" layer and grids on the "grid" layer, which is below "axes":
    QList<QCPAxis*> allAxes;
    allAxes << leftAxisRect->axes() << rightAxisRect->axes();
    foreach (QCPAxis *axis, allAxes)
    {
        axis->setLayer("axes");
        axis->grid()->setLayer("grid");
    }

    for (size_t i=0; i< m_NumPoints; ++i)
    {
        m_plot_td_vector_x->operator[](static_cast<int>(i)) = i*m_delta_t*m_decimation;
        m_plot_td_vector_y_real->operator[](static_cast<int>(i)) = 0;
        m_plot_td_vector_y_imag->operator[](static_cast<int>(i)) = 0;
    }

    // left
    ui->m_CustomPlot->addGraph(leftAxisRect->axis(QCPAxis::atBottom), leftAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(leftAxisRect->axis(QCPAxis::atBottom), leftAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(leftAxisRect->axis(QCPAxis::atBottom), leftAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(leftAxisRect->axis(QCPAxis::atBottom), leftAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(leftAxisRect->axis(QCPAxis::atBottom), leftAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(leftAxisRect->axis(QCPAxis::atBottom), leftAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(leftAxisRect->axis(QCPAxis::atBottom), leftAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(rightAxisRect->axis(QCPAxis::atBottom), rightAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(rightAxisRect->axis(QCPAxis::atBottom), rightAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(rightAxisRect->axis(QCPAxis::atBottom), rightAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(rightAxisRect->axis(QCPAxis::atBottom), rightAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(rightAxisRect->axis(QCPAxis::atBottom), rightAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(rightAxisRect->axis(QCPAxis::atBottom), rightAxisRect->axis(QCPAxis::atLeft));
    ui->m_CustomPlot->addGraph(rightAxisRect->axis(QCPAxis::atBottom), rightAxisRect->axis(QCPAxis::atLeft));


    ui->m_CustomPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    ui->m_CustomPlot->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->m_CustomPlot, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(ShowContextMenu(const QPoint &)));

    UpdatePlots();
    ui->m_CustomPlot->replot();


    // Connect Timers to Signals
    m_qtimer.setInterval(static_cast<int>(m_RefreshRate));
    m_qtimer.start();
    connect(&m_qtimer, SIGNAL(timeout()), this, SLOT(updateDataSlot()));
    connect(&m_qtimer, SIGNAL(timeout()), this, SLOT(StatusUpdateSlot()));
    connect(&m_performanceTest, SIGNAL(timeout()), this, SLOT(FinishTest()));


    // Initialize Logger by writing the header.
    m_logtimer.setInterval(static_cast<int>(200));
    m_logtimer.start();
    connect(&m_logtimer, SIGNAL(timeout()), this, SLOT(LogData()));

    m_bpstimer.setInterval(1000); // 1 second refresh rate
    m_bpstimer.start();
    connect(&m_bpstimer, SIGNAL(timeout()), this, SLOT(bpsUpdateSlot()));

    connect(this, &RtlSdr_Receiver::triggerDeserialize, this, &RtlSdr_Receiver::deserializeData);
    connect(this, &RtlSdr_Receiver::triggerDemodulation, this, &RtlSdr_Receiver::demodulateData);
    connect(this, &RtlSdr_Receiver::messageReceived, this, &RtlSdr_Receiver::updateDataMsg);
    ui->m_ReceiverActive->setOn(QtLight::BAD);

    double delta = 0.60;
    size_t numPoints = 100.0/delta;

    for(size_t i = 0; i < VectorIndexes::MAX; i++)
    {
        m_plot_temps[i] = new QVector<double>(numPoints);
        m_plot_currents[i] = new QVector<double>(numPoints);

        for(size_t j = 0; j < numPoints; j++)
        {
            m_plot_temps[i]->data()[j] = 0;
            m_plot_currents[i]->data()[j] = 0;
        }
    }

    m_plot_xvector = new QVector<double>(numPoints);
    for(size_t j = 0; j < numPoints; j++)
    {
        m_plot_xvector->data()[j] = j*delta;
    }

    // connect system monitor multicast receiver
    udpSocket4.bind(8888);

    // connect udp signals
    connect(&udpSocket4, SIGNAL(readyRead()), this, SLOT(processPendingSystemMonitorDatagrams()));
    connect(this, &RtlSdr_Receiver::PerformanceTestSignal, this, &RtlSdr_Receiver::PerformanceTests);
}


RtlSdr_Receiver::~RtlSdr_Receiver()
{
    for(int i = 0; i < MAX_NUM_RADOIOS; i++)
    {
        if(m_radio_rtlsdr[i])
        {
            m_radio_rtlsdr[i]->set_active(false);
        }
    }

    // disable timers
    m_qtimer.stop();
    m_logtimer.stop();
    m_bpstimer.stop();

    for(int i = 0; i < MAX_NUM_RADOIOS; i++)
    {

        if(m_qpsk[i])
        {
            delete m_qpsk[i];
        }

        if(m_bpsk[i])
        {
            delete m_bpsk[i];
        }

        if(m_am[i])
        {
            delete m_am[i];
        }
        if(m_fm[i])
        {
            delete m_fm[i];
        }

        if(m_framer[i])
        {
            delete m_framer[i];
        }

        if(m_ones_zeros[i])
        {
            delete m_ones_zeros[i];
        }
        if(m_packet_confidence[i])
        {
            delete m_packet_confidence[i];
        }
    }

    if(leftAxisRect)
    {
        delete leftAxisRect;
    }
    if(rightAxisRect)
    {
        delete rightAxisRect;
    }

    for(int i = 0; i < MAX_NUM_RADOIOS; i++)
    {
        if(m_pkt[i])
        {
            delete[] m_pkt[i];
        }
        if(m_temp_pkt[i])
        {
            delete[] m_temp_pkt[i];
        }
        if(m_ring_buffer[i])
        {
            delete m_ring_buffer[i];
        }
        if(m_data_buffer[i])
        {
            delete[] m_data_buffer[i];
        }
        if(m_block_output_x[i])
        {
            delete m_block_output_x[i];
        }
        if(m_block_output_y[i])
        {
            delete m_block_output_y[i];
        }
        if(m_block_demod_x[i])
        {
            delete m_block_demod_x[i];
        }
        if(m_block_demod_y[i])
        {
            delete m_block_demod_y[i];
        }
        if(m_block_x[i])
        {
            delete m_block_x[i];
        }
        if(m_block_y[i])
        {
            delete m_block_y[i];
        }
        if(m_temp_real[i])
        {
            delete m_temp_real[i];
        }
        if(m_temp_imag[i])
        {
            delete m_temp_imag[i];
        }
    }
    if(m_plot_td_vector_y_real)
    {
        delete m_plot_td_vector_y_real;
    }
    if(m_plot_td_vector_y_imag)
    {
        delete m_plot_td_vector_y_imag;
    }
    if(m_plot_td_vector_x)
    {
        delete m_plot_td_vector_x;
    }
    for(int i = 0; i < MAX_NUM_RADOIOS; i++)
    {
        if(m_radio_rtlsdr[i])
        {
            delete m_radio_rtlsdr[i];
        }
    }

    for(size_t i = 0; i < VectorIndexes::MAX; i++)
    {
        if(m_plot_temps[i])
        {
            delete m_plot_temps[i];
        }
        if(m_plot_currents[i])
        {
            delete m_plot_currents[i];
        }
    }

    delete m_plot_xvector;

    if(ui)
    {
        delete ui;
    }

}

void RtlSdr_Receiver::ShowContextMenu(const QPoint& /*pos*/)
{
   QMenu contextMenu(tr("Context menu"), this);

   QAction action1((m_Normalize) ? QString("Actual") : QString("Normalize"), this);
   connect(&action1, &QAction::triggered, this, [this]{ m_Normalize = !m_Normalize; emit UpdatePlots();});
   contextMenu.addAction(&action1);

   QAction actionABS((!m_ABS) ? QString("Enable ABS") : QString("Disable ABS"), this);
   connect(&actionABS, &QAction::triggered, this, [this]{ m_ABS = !m_ABS; emit UpdatePlots();});
   contextMenu.addAction(&actionABS);

   QAction actionAM((!m_AM) ? QString("Enable AM Debug") : QString("Disable AM Debug"), this);
   connect(&actionAM, &QAction::triggered, this, [this]{ m_AM = !m_AM; emit UpdatePlots();});
   contextMenu.addAction(&actionAM);

   QAction action2((m_DisablePlots) ? QString("Disable Plots") : QString("Enable Plots"), this);
   connect(&action2,  &QAction::triggered,  this, [this]{ m_DisablePlots = !m_DisablePlots; /*Flip the Bit*/ });
   contextMenu.addAction(&action2);

   QMenu* subMenu_Left = contextMenu.addMenu(tr("Left"));
   QMenu* subMenu_Right = contextMenu.addMenu(tr("Right"));

   QAction left_td_real_action("TD Real", this);
   QAction left_td_imag_action("TD Imag", this);
   QAction left_const_action("Constellation", this);
   QAction left_debug_action("Debug", this);
   QAction left_temp_action("Temperatures (*F)", this);
   QAction left_power_action("Power (Watts)", this);
   QAction left_proc_action("Processor (%Utilization)", this);
   QAction left_gpu_action("GPU (%Utilization)", this);
   QAction left_ram_action("Ram Utilization", this);
   QAction right_td_real_action("TD Real", this);
   QAction right_td_imag_action("TD Imag", this);
   QAction right_const_action("Constellation", this);
   QAction right_debug_action("Debug", this);
   QAction right_temp_action("Temperatures (*F)", this);
   QAction right_power_action("Power (Watts)", this);
   QAction right_proc_action("Processor (%Utilization)", this);
   QAction right_gpu_action("GPU (%Utilization)", this);
   QAction right_ram_action("Ram Utilization", this);

   connect(&left_td_real_action,  &QAction::triggered,  this, [this]{ m_left_plot = TIME_REAL; emit UpdatePlots();});
   connect(&left_td_imag_action,  &QAction::triggered,  this, [this]{ m_left_plot = TIME_IMAG; emit UpdatePlots();});
   connect(&left_const_action,    &QAction::triggered,  this, [this]{ m_left_plot = CONST; emit UpdatePlots();});
   connect(&left_debug_action,    &QAction::triggered,  this, [this]{ m_left_plot = DEBUG; emit UpdatePlots();});
   connect(&left_temp_action,     &QAction::triggered,  this, [this]{ m_left_plot = TEMPERATURE; emit UpdatePlots();});
   connect(&left_power_action,    &QAction::triggered,  this, [this]{ m_left_plot = POWER; emit UpdatePlots();});
   connect(&left_proc_action,     &QAction::triggered,  this, [this]{ m_left_plot = PROC_USAGE; emit UpdatePlots();});
   connect(&left_gpu_action,      &QAction::triggered,  this, [this]{ m_left_plot = GPU; emit UpdatePlots();});
   connect(&left_ram_action,      &QAction::triggered,  this, [this]{ m_left_plot = RAM_USAGE; emit UpdatePlots();});

   connect(&right_td_real_action, &QAction::triggered,  this, [this]{ m_right_plot = TIME_REAL; emit UpdatePlots();});
   connect(&right_td_imag_action, &QAction::triggered,  this, [this]{ m_right_plot = TIME_IMAG; emit UpdatePlots();});
   connect(&right_const_action,   &QAction::triggered,  this, [this]{ m_right_plot = CONST; emit UpdatePlots();});
   connect(&right_debug_action,   &QAction::triggered,  this, [this]{ m_right_plot = DEBUG; emit UpdatePlots();});
   connect(&right_temp_action,    &QAction::triggered,  this, [this]{ m_right_plot = TEMPERATURE; emit UpdatePlots();});
   connect(&right_power_action,   &QAction::triggered,  this, [this]{ m_right_plot = POWER; emit UpdatePlots();});
   connect(&right_proc_action,    &QAction::triggered,  this, [this]{ m_right_plot = PROC_USAGE; emit UpdatePlots();});
   connect(&right_gpu_action,     &QAction::triggered,  this, [this]{ m_right_plot = GPU; emit UpdatePlots();});
   connect(&right_ram_action,     &QAction::triggered,  this, [this]{ m_right_plot = RAM_USAGE; emit UpdatePlots();});

   subMenu_Left->addAction(&left_td_real_action);
   subMenu_Left->addAction(&left_td_imag_action);
   subMenu_Left->addAction(&left_const_action);
   subMenu_Left->addAction(&left_debug_action);
   subMenu_Left->addAction(&left_temp_action);
   subMenu_Left->addAction(&left_power_action);
   subMenu_Left->addAction(&left_proc_action);
   subMenu_Left->addAction(&left_gpu_action);
   subMenu_Left->addAction(&left_ram_action);
   subMenu_Right->addAction(&right_td_real_action);
   subMenu_Right->addAction(&right_td_imag_action);
   subMenu_Right->addAction(&right_const_action);
   subMenu_Right->addAction(&right_debug_action);
   subMenu_Right->addAction(&right_temp_action);
   subMenu_Right->addAction(&right_power_action);
   subMenu_Right->addAction(&right_proc_action);
   subMenu_Right->addAction(&right_gpu_action);
   subMenu_Right->addAction(&right_ram_action);

   contextMenu.exec(QCursor::pos());
}

void RtlSdr_Receiver::UpdatePlots()
{
    RADIO_DATA_TYPE timeRange = static_cast<RADIO_DATA_TYPE>(m_RefreshRate) / 1000.0;

    // update graphs
    left_arLegend->remove(left_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(LEFT_PLOT_0)));
    left_arLegend->remove(left_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(LEFT_PLOT_1)));
    left_arLegend->remove(left_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(LEFT_PLOT_2)));
    left_arLegend->remove(left_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(LEFT_PLOT_3)));
    left_arLegend->remove(left_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(LEFT_PLOT_4)));
    left_arLegend->remove(left_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(LEFT_PLOT_5)));
    left_arLegend->remove(left_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(LEFT_PLOT_6)));
    right_arLegend->remove(right_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(RIGHT_PLOT_0)));
    right_arLegend->remove(right_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(RIGHT_PLOT_1)));
    right_arLegend->remove(right_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(RIGHT_PLOT_2)));
    right_arLegend->remove(right_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(RIGHT_PLOT_3)));
    right_arLegend->remove(right_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(RIGHT_PLOT_4)));
    right_arLegend->remove(right_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(RIGHT_PLOT_5)));
    right_arLegend->remove(right_arLegend->itemWithPlottable(ui->m_CustomPlot->graph(RIGHT_PLOT_6)));


    if(m_Normalize)
    {
        switch(m_left_plot)
        {
        case DEBUG:
        case TIME_REAL:
        case TIME_IMAG:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-1.5,1.5);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(0.0,timeRange);
            break;
        case CONST:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-1.5,1.5);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(-1.5,1.5);
            break;
        case TEMPERATURE:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-10, 150);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case POWER:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 1000);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case PROC_USAGE:
        case GPU:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 110);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case RAM_USAGE:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 110);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        }
        switch(m_right_plot)
        {
        case DEBUG:
        case TIME_REAL:
        case TIME_IMAG:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-1.5,1.5);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(0.0,timeRange);
            break;
        case CONST:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-1.5,1.5);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(-1.5,1.5);
            break;
        case TEMPERATURE:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-10, 150);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case POWER:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 1000);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case PROC_USAGE:
        case GPU:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 110);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case RAM_USAGE:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 110);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        }
    }
    else
    {
        switch(m_left_plot)
        {
        case DEBUG:
        case TIME_REAL:
        case TIME_IMAG:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-150,150);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(0.0,timeRange);
            break;
        case CONST:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-150,150);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(-150,150);
            break;
        case TEMPERATURE:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-10, 150);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case POWER:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 1000);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case GPU:
        case PROC_USAGE:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 110);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case RAM_USAGE:
            leftAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 110);
            leftAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        }
        switch(m_right_plot)
        {
        case DEBUG:
        case TIME_REAL:
        case TIME_IMAG:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-150,150);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(0.0,timeRange);
            break;
        case CONST:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-150,150);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(-150,150);
            break;
        case TEMPERATURE:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-10, 150);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case POWER:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 1000);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case GPU:
        case PROC_USAGE:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 110);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        case RAM_USAGE:
            rightAxisRect->axis(QCPAxis::atLeft)->setRange(-1, 110);
            rightAxisRect->axis(QCPAxis::atBottom)->setRange(0,100);
            break;
        }
    }

    switch(m_left_plot)
    {
    case TIME_REAL:
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        leftAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        leftAxisRect->axis(QCPAxis::atLeft)->setLabel("Amplitude");
        leftAxisRect->axis(QCPAxis::atTop)->setLabel("Real [TD]");
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setName("Time Domain [Real]");
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_0)));
        break;
    case TIME_IMAG:
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        leftAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        leftAxisRect->axis(QCPAxis::atLeft)->setLabel("Amplitude");
        leftAxisRect->axis(QCPAxis::atTop)->setLabel("Imag [TD]");
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setName("Time Domain [Imag]");
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_0)));
        break;
    case CONST:
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsNone);
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4));
        leftAxisRect->axis(QCPAxis::atBottom)->setLabel("In-Phase");
        leftAxisRect->axis(QCPAxis::atLeft)->setLabel("Quadrature");
        leftAxisRect->axis(QCPAxis::atTop)->setLabel("Constellation");
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setName("Constellation");
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_0)));
        break;
    case DEBUG:
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsNone);
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4));
        leftAxisRect->axis(QCPAxis::atBottom)->setLabel("In-Phase");
        leftAxisRect->axis(QCPAxis::atLeft)->setLabel("Quadrature");
        leftAxisRect->axis(QCPAxis::atTop)->setLabel("Debug");
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setName("Debug");
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_0)));
        break;
    case TEMPERATURE:
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(LEFT_PLOT_1)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(LEFT_PLOT_2)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(LEFT_PLOT_3)->setPen(QPen(Qt::cyan));
        ui->m_CustomPlot->graph(LEFT_PLOT_4)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(LEFT_PLOT_5)->setPen(QPen(Qt::black));
        ui->m_CustomPlot->graph(LEFT_PLOT_6)->setPen(QPen(Qt::gray));
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        leftAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        leftAxisRect->axis(QCPAxis::atLeft)->setLabel("*F");
        leftAxisRect->axis(QCPAxis::atTop)->setLabel("Temperature");

        // Set Plot Names
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setName("RTL_SDR [0]");
        ui->m_CustomPlot->graph(LEFT_PLOT_1)->setName("RTL_SDR [1]");
        ui->m_CustomPlot->graph(LEFT_PLOT_2)->setName("RTL_SDR [2]");
        ui->m_CustomPlot->graph(LEFT_PLOT_3)->setName("RTL_SDR [3]");
        ui->m_CustomPlot->graph(LEFT_PLOT_4)->setName("RTL_SDR [4]");
        ui->m_CustomPlot->graph(LEFT_PLOT_5)->setName("RTL_SDR [5]");
        ui->m_CustomPlot->graph(LEFT_PLOT_6)->setName("HackRF One");
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_0)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_1)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_2)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_3)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_4)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_5)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_6)));
        break;
    case POWER:
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(LEFT_PLOT_1)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(LEFT_PLOT_2)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(LEFT_PLOT_3)->setPen(QPen(Qt::cyan));
        ui->m_CustomPlot->graph(LEFT_PLOT_4)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(LEFT_PLOT_5)->setPen(QPen(Qt::black));
        ui->m_CustomPlot->graph(LEFT_PLOT_6)->setPen(QPen(Qt::gray));
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        leftAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        leftAxisRect->axis(QCPAxis::atLeft)->setLabel("milli-Watts");
        leftAxisRect->axis(QCPAxis::atTop)->setLabel("Power Usage");
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setName("RTL_SDR [0]");
        ui->m_CustomPlot->graph(LEFT_PLOT_1)->setName("RTL_SDR [1]");
        ui->m_CustomPlot->graph(LEFT_PLOT_2)->setName("RTL_SDR [2]");
        ui->m_CustomPlot->graph(LEFT_PLOT_3)->setName("RTL_SDR [3]");
        ui->m_CustomPlot->graph(LEFT_PLOT_4)->setName("RTL_SDR [4]");
        ui->m_CustomPlot->graph(LEFT_PLOT_5)->setName("RTL_SDR [5]");
        ui->m_CustomPlot->graph(LEFT_PLOT_6)->setName("HackRF One");
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_0)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_1)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_2)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_3)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_4)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_5)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_6)));
        break;
    case PROC_USAGE:
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(LEFT_PLOT_1)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(LEFT_PLOT_2)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(LEFT_PLOT_3)->setPen(QPen(Qt::cyan));
        ui->m_CustomPlot->graph(LEFT_PLOT_4)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(LEFT_PLOT_5)->setPen(QPen(Qt::black));
        ui->m_CustomPlot->graph(LEFT_PLOT_6)->setPen(QPen(Qt::gray));
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        leftAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        leftAxisRect->axis(QCPAxis::atLeft)->setLabel("Percent Utilization");
        leftAxisRect->axis(QCPAxis::atTop)->setLabel("% Usage");

        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setName("Average");
        ui->m_CustomPlot->graph(LEFT_PLOT_1)->setName("Core 0");
        ui->m_CustomPlot->graph(LEFT_PLOT_2)->setName("Core 1");
        ui->m_CustomPlot->graph(LEFT_PLOT_3)->setName("Core 2");
        ui->m_CustomPlot->graph(LEFT_PLOT_4)->setName("Core 3");
        ui->m_CustomPlot->graph(LEFT_PLOT_5)->setName("Core 4");
        ui->m_CustomPlot->graph(LEFT_PLOT_6)->setName("Core 5");
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_0)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_1)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_2)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_3)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_4)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_5)));
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_6)));
        break;
    case GPU:
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(LEFT_PLOT_1)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(LEFT_PLOT_2)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(LEFT_PLOT_3)->setPen(QPen(Qt::cyan));
        ui->m_CustomPlot->graph(LEFT_PLOT_4)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(LEFT_PLOT_5)->setPen(QPen(Qt::black));
        ui->m_CustomPlot->graph(LEFT_PLOT_6)->setPen(QPen(Qt::gray));
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        leftAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        leftAxisRect->axis(QCPAxis::atLeft)->setLabel("Percent Utilization");
        leftAxisRect->axis(QCPAxis::atTop)->setLabel("% Usage");
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setName("GPU");
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_0)));
        break;
    case RAM_USAGE:
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(LEFT_PLOT_1)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(LEFT_PLOT_2)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(LEFT_PLOT_3)->setPen(QPen(Qt::cyan));
        ui->m_CustomPlot->graph(LEFT_PLOT_4)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(LEFT_PLOT_5)->setPen(QPen(Qt::black));
        ui->m_CustomPlot->graph(LEFT_PLOT_6)->setPen(QPen(Qt::gray));
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        leftAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        leftAxisRect->axis(QCPAxis::atLeft)->setLabel("Percent Utilization");
        leftAxisRect->axis(QCPAxis::atTop)->setLabel("RAM Usage");
        ui->m_CustomPlot->graph(LEFT_PLOT_0)->setName("Ram");
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(LEFT_PLOT_0)));
        break;
    }

    switch(m_right_plot)
    {
    case TIME_REAL:
    {
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        rightAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        rightAxisRect->axis(QCPAxis::atLeft)->setLabel("Amplitude");
        rightAxisRect->axis(QCPAxis::atTop)->setLabel("Real [TD]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setName("Time Domain [Real]");
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_0)));
    }
        break;
    case TIME_IMAG:
    {
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        rightAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        rightAxisRect->axis(QCPAxis::atLeft)->setLabel("Amplitude");
        rightAxisRect->axis(QCPAxis::atTop)->setLabel("Imag [TD]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setName("Time Domain [Imag]");
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_0)));
        break;
    }
    case CONST:
    {
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsNone);
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4));
        rightAxisRect->axis(QCPAxis::atBottom)->setLabel("In-Phase");
        rightAxisRect->axis(QCPAxis::atLeft)->setLabel("Quadrature");
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setName("Constellation");
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_0)));
        break;
    }
    case DEBUG:
    {
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsNone);
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4));
        rightAxisRect->axis(QCPAxis::atBottom)->setLabel("In-Phase");
        rightAxisRect->axis(QCPAxis::atLeft)->setLabel("Quadrature");
        rightAxisRect->axis(QCPAxis::atTop)->setLabel("Debug");
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setName("Debug");
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_0)));
        break;
        break;
    }
    case TEMPERATURE:
    {
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setPen(QPen(Qt::cyan));
        ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setPen(QPen(Qt::black));
        ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setPen(QPen(Qt::gray));
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        rightAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        rightAxisRect->axis(QCPAxis::atLeft)->setLabel("*F");
        rightAxisRect->axis(QCPAxis::atTop)->setLabel("Temperature");


        // Set Plot Names
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setName("RTL_SDR [0]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setName("RTL_SDR [1]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setName("RTL_SDR [2]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setName("RTL_SDR [3]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setName("RTL_SDR [4]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setName("RTL_SDR [5]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setName("HackRF One");
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_0)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_1)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_2)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_3)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_4)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_5)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_6)));

        break;
    }
    case POWER:
    {
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setPen(QPen(Qt::cyan));
        ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setPen(QPen(Qt::black));
        ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setPen(QPen(Qt::gray));
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        rightAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        rightAxisRect->axis(QCPAxis::atLeft)->setLabel("milli-Watts");
        rightAxisRect->axis(QCPAxis::atTop)->setLabel("Power Usage");

        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setName("RTL_SDR [0]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setName("RTL_SDR [1]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setName("RTL_SDR [2]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setName("RTL_SDR [3]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setName("RTL_SDR [4]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setName("RTL_SDR [5]");
        ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setName("HackRF One");
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_0)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_1)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_2)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_3)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_4)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_5)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_6)));

        break;
    }
    case PROC_USAGE:
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setPen(QPen(Qt::cyan));
        ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setPen(QPen(Qt::black));
        ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setPen(QPen(Qt::gray));
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        rightAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        rightAxisRect->axis(QCPAxis::atLeft)->setLabel("Percent Utilization");
        rightAxisRect->axis(QCPAxis::atTop)->setLabel("% Usage");

        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setName("Average");
        ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setName("Core 0");
        ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setName("Core 1");
        ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setName("Core 2");
        ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setName("Core 3");
        ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setName("Core 4");
        ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setName("Core 5");
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_0)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_1)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_2)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_3)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_4)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_5)));
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_6)));
        break;
    case GPU:
    {
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setPen(QPen(Qt::cyan));
        ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setPen(QPen(Qt::black));
        ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setPen(QPen(Qt::gray));
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        rightAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        rightAxisRect->axis(QCPAxis::atLeft)->setLabel("Percent Utilization");
        rightAxisRect->axis(QCPAxis::atTop)->setLabel("% Usage");
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setName("GPU");
        right_arLegend->addItem(new QCPPlottableLegendItem(right_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_0)));
        break;
    }
    case RAM_USAGE:
    {
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setPen(QPen(Qt::red));
        ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setPen(QPen(Qt::green));
        ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setPen(QPen(Qt::blue));
        ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setPen(QPen(Qt::cyan));
        ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setPen(QPen(Qt::magenta));
        ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setPen(QPen(Qt::black));
        ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setPen(QPen(Qt::gray));
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setLineStyle(QCPGraph::LineStyle::lsLine);
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, 4));
        rightAxisRect->axis(QCPAxis::atBottom)->setLabel("Time");
        rightAxisRect->axis(QCPAxis::atLeft)->setLabel("Percent Utilization");
        rightAxisRect->axis(QCPAxis::atTop)->setLabel("RAM Usage");
        ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setName("Ram");
        left_arLegend->addItem(new QCPPlottableLegendItem(left_arLegend, ui->m_CustomPlot->graph(RIGHT_PLOT_0)));

        break;
    }
    }
}


void RtlSdr_Receiver::updateDataMsg(bool validPacket, bool validCRC, bool validHeader, int bad_bits, int good_bits)
{
    m_bad_bits += static_cast<size_t>(bad_bits);
    m_good_bits += static_cast<size_t>(good_bits);

    // load packet into guid
    if(!m_check_the_crc)
    {
        ui->m_ValidCRC->setOn(QtLight::NONE);   // no CRC
        if(validPacket)
        {
            ui->m_data_msg->clear();
            m_prev_pkt_cnt = m_pkt_cnt;
            m_pkt_cnt++;
            (*m_packet_confidence)[0]++;
        }
        else
        {
            // got a problem here
            (*m_packet_confidence)[0]--;
        }
    }
    else
    {
        if(validHeader && validCRC)
        {
            ui->m_data_msg->clear();
            ui->m_data_msg->appendPlainText(QString((const char*)m_pkt[0]));
            m_prev_pkt_cnt = m_pkt_cnt;
            m_pkt_cnt++;
            (*m_packet_confidence)[0]++;
            ui->m_ValidCRC->setOn(QtLight::GOOD);   // no CRC
        }
        else if(validHeader && validCRC)
        {
            // got a problem here
            (*m_packet_confidence)[0]--;
            ui->m_ValidCRC->setOn(QtLight::BAD);   // no CRC
        }
    }

    if(m_check_the_crc && validCRC)
    {
        m_num_validCRCs++;
    }
    if(validHeader)
    {
        m_num_validHeaders++;
    }
}

void RtlSdr_Receiver::updateDataSlot()
{
    if(!m_DisablePlots || ui == nullptr)
    {
        return;
    }

    // calculate two new data points:
    if((m_radio_rtlsdr[0] && m_radio_rtlsdr[0]->get_is_active()))
    {
        switch(m_left_plot)
        {
        case TIME_REAL:
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->setData(*m_plot_td_vector_x, *m_plot_td_vector_y_real);
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->data()->clear();
            break;
        case TIME_IMAG:
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->setData(*m_plot_td_vector_x, *m_plot_td_vector_y_imag);
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->data()->clear();
            break;
        case CONST:
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->setData(*m_plot_td_vector_y_real, *m_plot_td_vector_y_imag);
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->data()->clear();
            break;
        case DEBUG:
            //ui->m_CustomPlot->graph(LEFT_PLOT_0)->setData(*m_plot_fd_vector_x, *m_fd_vector_y);
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->data()->clear();
            break;
        case TEMPERATURE:
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL0]);
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL1]);
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL2]);
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL3]);
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL4]);
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL5]);
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::HackRF]);
            break;

        case POWER:
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL0]);
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL1]);
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL2]);
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL3]);
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL4]);
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL5]);
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::HackRF]);
            break;
        case PROC_USAGE:
#ifdef GPU_ENABLED
            // GPU version
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu_filt_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu0_filt_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu1_filt_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu2_filt_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu3_filt_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu4_filt_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu5_filt_load);
#else
// MAC Version
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->setData(m_linux_usage.m_xvector, m_linux_usage.m_cpu_avg_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->setData(m_linux_usage.m_xvector, m_linux_usage.m_cpu0_avg_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->setData(m_linux_usage.m_xvector, m_linux_usage.m_cpu1_avg_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->setData(m_linux_usage.m_xvector, m_linux_usage.m_cpu2_avg_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->setData(m_linux_usage.m_xvector, m_linux_usage.m_cpu3_avg_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->data()->clear();
#endif
            break;
        case GPU:
#ifdef GPU_ENABLED
            // GPU version
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_gpu_filt_load);
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->data()->clear();
#else
// MAC Version
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->data()->clear();
#endif

            break;
        case RAM_USAGE:
#ifdef GPU_ENABLED
            // GPU version
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_usedRam_vec);
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->data()->clear();
#else
// MAC Version
            // GPU version
            ui->m_CustomPlot->graph(LEFT_PLOT_0)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(LEFT_PLOT_6)->data()->clear();
#endif
        }



        switch(m_right_plot)
        {
        case TIME_REAL:
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setData(*m_plot_td_vector_x, *m_plot_td_vector_y_real);
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->data()->clear();
            break;
        case TIME_IMAG:
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setData(*m_plot_td_vector_x, *m_plot_td_vector_y_imag);
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->data()->clear();
            break;
        case CONST:
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setData(*m_plot_td_vector_y_real, *m_plot_td_vector_y_imag);
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->data()->clear();
            break;
        case DEBUG:
           // ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setData(*m_plot_fd_vector_x, *m_fd_vector_y);
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->data()->clear();
            break;
        case TEMPERATURE:
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL0]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL1]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL2]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL3]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL4]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::RTL5]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setData(*m_plot_xvector, *m_plot_temps[VectorIndexes::HackRF]);
            break;

        case POWER:
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL0]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL1]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL2]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL3]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL4]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::RTL5]);
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setData(*m_plot_xvector, *m_plot_currents[VectorIndexes::HackRF]);
            break;
        case PROC_USAGE:
#ifdef GPU_ENABLED
// GPU version
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu_filt_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu0_filt_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu1_filt_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu2_filt_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu3_filt_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu4_filt_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_cpu5_filt_load);

#else
// MAC Version
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setData(m_linux_usage.m_xvector, m_linux_usage.m_cpu_avg_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->setData(m_linux_usage.m_xvector, m_linux_usage.m_cpu0_avg_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->setData(m_linux_usage.m_xvector, m_linux_usage.m_cpu1_avg_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->setData(m_linux_usage.m_xvector, m_linux_usage.m_cpu2_avg_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->setData(m_linux_usage.m_xvector, m_linux_usage.m_cpu3_avg_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->data()->clear();
#endif
            break;
        case GPU:
#ifdef GPU_ENABLED
            // GPU version
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_gpu_filt_load);
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->data()->clear();
#else
// MAC Version
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->data()->clear();
#endif
            break;
        case RAM_USAGE:
#ifdef GPU_ENABLED
            // GPU version
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->setData(m_nvidia_usage.m_xvector, m_nvidia_usage.m_usedRam_vec);
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->data()->clear();
#else
// MAC Version
            // GPU version
            ui->m_CustomPlot->graph(RIGHT_PLOT_0)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_1)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_2)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_3)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_4)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_5)->data()->clear();
            ui->m_CustomPlot->graph(RIGHT_PLOT_6)->data()->clear();
#endif
            break;
        }
    }


    // Update Processor Usage
#ifdef GPU_ENABLED
    ui->m_cpu_average->setText(QString::number(m_nvidia_usage.m_cpu_filt_load.back(), 'f', 2));
    ui->m_core_0->setText(QString::number(m_nvidia_usage.m_cpu0_filt_load.back(), 'f', 2));
    ui->m_core_1->setText(QString::number(m_nvidia_usage.m_cpu1_filt_load.back(), 'f', 2));
    ui->m_core_2->setText(QString::number(m_nvidia_usage.m_cpu2_filt_load.back(), 'f', 2));
    ui->m_core_3->setText(QString::number(m_nvidia_usage.m_cpu3_filt_load.back(), 'f', 2));
    ui->m_core_4->setText(QString::number(m_nvidia_usage.m_cpu4_filt_load.back(), 'f', 2));
    ui->m_core_5->setText(QString::number(m_nvidia_usage.m_cpu5_filt_load.back(), 'f', 2));
    ui->m_gpu_average->setText(QString::number(m_nvidia_usage.m_gpu_filt_load.back(), 'f', 2));
#else
    ui->m_cpu_average->setText("N/A");
    ui->m_core_0->setText("N/A");
    ui->m_core_1->setText("N/A");
    ui->m_core_2->setText("N/A");
    ui->m_core_3->setText("N/A");
    ui->m_core_4->setText("N/A");
    ui->m_core_5->setText("N/A");
    ui->m_gpu_average->setText("N/A");
#endif
    ui->m_CustomPlot->replot();
}

void RtlSdr_Receiver::StatusUpdateSlot()
{
    if(ui == nullptr || m_ones_zeros == nullptr)
    {
        return;
    }

    if(m_radio_rtlsdr[0] && m_radio_rtlsdr[0]->get_is_active())
    {
        if(m_mode == Radio_RTLSDR::ZEROS)
        {
            if(m_ones_zeros[0]->low_confidence()) // Low means lots of zeros and lock
            {
                ui->m_ModemLock->setOn(QtLight::GOOD);
            }
            else if(m_ones_zeros[0]->high_confidence())
            {
                ui->m_ModemLock->setOn(QtLight::BAD);
            }
            else
            {
                ui->m_ModemLock->setOn(QtLight::WARN);
            }

            ui->m_ValidCRC->setOn(QtLight::NONE);
        }
        else if(m_mode == Radio_RTLSDR::ONES)
        {
            if(m_ones_zeros[0]->high_confidence()) // high means lots of ones and lock
            {
                ui->m_ModemLock->setOn(QtLight::GOOD);
            }
            else if(m_ones_zeros[0]->low_confidence())
            {
                ui->m_ModemLock->setOn(QtLight::BAD);
            }
            else
            {
                ui->m_ModemLock->setOn(QtLight::WARN);
            }
            ui->m_ValidCRC->setOn(QtLight::NONE);
        }
        else if(m_mode == Radio_RTLSDR::AM || m_mode == Radio_RTLSDR::FM || m_mode == Radio_RTLSDR::BPSK || m_mode == Radio_RTLSDR::QPSK)
        {
            if(m_pkt_cnt > m_prev_pkt_cnt && m_packet_confidence[0]->low_confidence())
            {
               m_packet_confidence[0]->reset();
            }

            if(m_packet_confidence[0]->low_confidence())
            {
                ui->m_ModemLock->setOn(QtLight::BAD);
            }
            else if(m_packet_confidence[0]->high_confidence())
            {
                ui->m_ModemLock->setOn(QtLight::GOOD);
            }
            else
            {
                ui->m_ModemLock->setOn(QtLight::WARN);
            }
        }
        else
        {
            ui->m_ModemLock->setOn(QtLight::BAD);
            ui->m_ValidCRC->setOn(QtLight::BAD);
        }
    }
    else
    {
        ui->m_ModemLock->setOn(QtLight::BAD);
        ui->m_ValidCRC->setOn(QtLight::BAD);
    }

    ui->m_data_msg->clear();
    ui->m_data_msg->appendPlainText(QString((const char*)m_pkt[0]));           // Packet Message
    ui->m_num_pkts_recieved->setNum(static_cast<int>(m_pkt_cnt));           // Pkts Received
    ui->m_valid_CRCs->setNum(static_cast<int>(m_num_validCRCs));            // Valid CRCs
    ui->m_HeadersReceived->setNum(static_cast<int>(m_num_validHeaders));    // Valid Headers
    ui->m_invalid_bits->setNum(static_cast<int>(m_bad_bits));               // Valid Headers
    if((m_good_bits + m_bad_bits) > 0)
    {
        ui->m_ber->setNum(static_cast<RADIO_DATA_TYPE>(static_cast<RADIO_DATA_TYPE>(m_bad_bits)/static_cast<RADIO_DATA_TYPE>(m_good_bits+m_bad_bits)));
    }
    else
    {
        ui->m_ber->setNum(0.0);
    }
    ui->m_good_bits->setNum(static_cast<int>(m_good_bits));

   // ui->statusbar->showMessage("Buffer  NumOfBits[" + QString::number(m_ring_buffer[0]->count()) + "/" + QString::number(m_ring_buffer[0]->max()) + "]", 1000);
}

 void RtlSdr_Receiver::bpsUpdateSlot()
 {
     if(ui)
     {
         // Set fields
         ui->m_actual_samples_per_second->setText(QString::number(m_actual_samples_per_second/2));
         ui->m_actual_bits_per_second->setText(QString::number((m_good_bits+m_bad_bits)));
     }
     // clear values
     m_actual_samples_per_second = 0;
 }



 void RtlSdr_Receiver::callbackRoutine(unsigned char* buffer, uint32_t /*length*/, int RADIO_ID)
 {
     if(m_GlobalPtr == nullptr ||
       (!(m_GlobalPtr->m_radio_rtlsdr[RADIO_ID] && m_GlobalPtr->m_radio_rtlsdr[RADIO_ID]->get_is_active())))
     {
         return;
     }
     else
     {
         if(m_GlobalPtr->m_radio_rtlsdr[RADIO_ID] && m_GlobalPtr->m_radio_rtlsdr[RADIO_ID]->get_is_active())
         {
             // Sample Rate is 2000000 even though the buffers are coming in with 2048000 samples
            RADIO_DATA_TYPE real = 0;
            RADIO_DATA_TYPE imag = 0;
            RADIO_DATA_TYPE real_low = 0;
            RADIO_DATA_TYPE imag_low = 0;
            RADIO_DATA_TYPE real_high = 0;
            RADIO_DATA_TYPE imag_high = 0;

            m_GlobalPtr->m_block_lock[RADIO_ID].lock();
            for(uint32_t i = 0; i < 2000000; i+=2)
            {
                real = (buffer[i]  -127);
                imag = (buffer[i+1]-127);

                // only plot the first radio
                if((RADIO_ID == 0) && (m_GlobalPtr->plotVar < m_GlobalPtr->m_NumPoints) && ((i % m_GlobalPtr->m_decimation) == 0)) // Decimate the array for plotting
                {
                    if(m_GlobalPtr->m_Normalize)
                    {
                        real/=127.0;
                        imag/=127.0;
                    }


                    if(m_GlobalPtr->m_AM) // now its FM
                    {
                        real = sqrt(real_low*real_low + imag_low*imag_low)*100;
                        imag = sqrt(real_high*real_high + imag_high*imag_high)*100;
                    }

                    if(m_GlobalPtr->m_ABS)
                    {
                        real = abs(real);
                        imag = abs(imag);
                    }

                    (*m_GlobalPtr->m_plot_td_vector_y_real)[m_GlobalPtr->plotVar] = real;
                    (*m_GlobalPtr->m_plot_td_vector_y_imag)[m_GlobalPtr->plotVar] = imag;
                    m_GlobalPtr->plotVar++;
                    imag = 0;
                }

                // Fully Load Block -- All points go in here - adjusted but raw
                (*(m_GlobalPtr->m_block_x[RADIO_ID]))[m_GlobalPtr->m_block_index[RADIO_ID]] = (static_cast<RADIO_DATA_TYPE>(buffer[i]   - 127));
                (*(m_GlobalPtr->m_block_y[RADIO_ID]))[m_GlobalPtr->m_block_index[RADIO_ID]] = (static_cast<RADIO_DATA_TYPE>(buffer[i+1] - 127));
                m_GlobalPtr->m_block_index[RADIO_ID]++;
                m_GlobalPtr->m_actual_samples_per_second++; // I/Q samples are 2 for each sample

                // Reset Sample Rate
                m_GlobalPtr->m_block_index[RADIO_ID] = m_GlobalPtr->m_block_index[RADIO_ID] % (m_GlobalPtr->m_SampleRate);

                if(m_GlobalPtr->m_block_index[RADIO_ID] % (m_GlobalPtr->m_SampleRate) == 0)
                {
                    m_GlobalPtr->m_block_index[RADIO_ID] = 0;
                    emit m_GlobalPtr->triggerDemodulation(RADIO_ID);    // Trigger demodulation with Radio ID
                    m_GlobalPtr->m_radio_rtlsdr[RADIO_ID]->set_active(false);
                }

                // Reset Plot Variables
                if(RADIO_ID == 0 && (m_GlobalPtr->plotVar % m_GlobalPtr->m_NumPoints == 0))
                {
                    m_GlobalPtr->plotVar = 0;
                }
            }


            int8_t a = 0;
            int8_t b = 0;
            uint8_t ONE = 1;
            uint8_t ZERO = 0;
            for(size_t i = 0; i < 2000000; i++)
            {
                //fout << f.do_sample(buffer[i]-127) << "," << std::endl;
               // fout << (int8_t)(buffer[i]-127) << "," << std::endl;
                a = buffer[i] - 127;
                b = buffer[i+2] - 127;

             //   std::cout << (int)a << "  " << (int)b << "  " << std::endl;
                m_GlobalPtr->debug << a;
//                if( (a < 0 and b > 0) || (a > 0 and b < 0))
//                {
//              //      std::cout << "HIGH" << std::endl;
//                  //    fout << a;
//                }
//                else
//                {
//                 //   fout << b;
//                  //  fout << (uint8_t)0 << std::endl;
//                //    std::cout << "LOW" << std::endl;
//                }
            }


            m_GlobalPtr->m_block_lock[RADIO_ID].unlock();
         }
     }
 }

 void RtlSdr_Receiver::setMode()
 {
     std::cout << "[RTLSDR] Setting Mode to: " << ui->m_decoding_scheme->currentText().toUtf8().constData() << "[" << ui->m_decoding_scheme->currentIndex() << "]" << std::endl;
     if(ui->m_decoding_scheme->currentText() == "ZEROS")
     {
         m_mode = Radio_RTLSDR::ZEROS;
     }
     else if(ui->m_decoding_scheme->currentText() == "ONES")
     {
         m_mode = Radio_RTLSDR::ONES;
     }
     else if(ui->m_decoding_scheme->currentText() == "AM")
     {
         m_mode = Radio_RTLSDR::AM;
     }
     else if(ui->m_decoding_scheme->currentText() == "FM")
     {
         m_mode = Radio_RTLSDR::FM;
     }
     else if(ui->m_decoding_scheme->currentText() == "BPSK")
     {
         m_mode = Radio_RTLSDR::BPSK;
     }
     else
     {
         m_mode = Radio_RTLSDR::QPSK;
     }
 }

void RtlSdr_Receiver::on_pushButton_clicked()
{
    if(ui == nullptr)
    {
        return;
    }

    if(rtlsdr_get_device_count() > 0)
    {
        for(int i = 0; i < MAX_NUM_RADOIOS; i++)
        {
            if(m_radio_rtlsdr[i] != nullptr)
            {
                m_radio_rtlsdr[i]->set_active(false);
                ui->m_ReceiverActive->setOn(QtLight::BAD);
                m_bps = static_cast<size_t>(ui->m_bps->value());
                std::cout << "BPS configured to " << m_bps << std::endl;
                setMode();
                m_radio_rtlsdr[i]->set_agc_enabled(false); // Leave it always disabled
                m_radio_rtlsdr[i]->set_gain_agc(static_cast<int>(ui->m_gain->value()*10));
                m_radio_rtlsdr[i]->set_center_freq(static_cast<unsigned int>(MHZ_TO_HZ(ui->m_freq->value())));
                m_radio_rtlsdr[i]->set_sample_rate(2000000); // Samples per second 1/5 of the hackrf
                ui->m_ReceiverActive->setOn(QtLight::GOOD);
              //  ui->statusbar->showMessage("Radio Configured", 1000);
                m_framer[i]->checkCRC(ui->m_check_crc->isChecked());
                m_check_the_crc = ui->m_check_crc->isChecked();

                // reset stuff
                ui->m_data_msg->clear();
                ui->m_ValidCRC->setOn(QtLight::BAD);
                ui->m_ModemLock->setOn(QtLight::BAD);
                m_ones_zeros[i]->reset();
                m_ring_buffer[i]->reset(); // flush buffers
                m_radio_rtlsdr[i]->set_active(true);
            }


            if(m_radio_rtlsdr[0] == nullptr)
            {
                ui->statusbar->showMessage("Please connect to a radio before configuring", 0);
            }
        }
    }

    // Reset the counters
    m_good_bits = 0;
    m_bad_bits = 0;
    m_actual_bits_per_second = 0;
    m_num_validCRCs=0;
    m_num_validHeaders=0;



    if(ui->m_rb_CPU->isChecked())
    {
        // Configure CPU Demodulation
        m_cpu_vs_gpu = false;
    }
    else
    {
        // Configure GPU Demodulation
        m_cpu_vs_gpu = true;
    }
}


void RtlSdr_Receiver::radio_search()
{
    ui->m_radio_list->clear();
    ui->m_radio_list->addItem(QString("None Detected"));

    ui->m_gain->setValue(14);
    // RTLSDR
    if(rtlsdr_get_device_count() > 0)
    {
        ui->m_radio_list->clear();
    }

    // capture number of radios
    for(unsigned int i = 0; i < rtlsdr_get_device_count(); i++)
    {
        char manufact[255] = {0};
        char product[255] = {0};
        char serial[255] = {0};
        rtlsdr_get_device_usb_strings(i, manufact, product, serial);
        cout << "Device [" << i << "] " << rtlsdr_get_device_name(i)
                  << "\n\t\t Manufac ... " << manufact
                  << "\n\t\t Product ... " << product
                  << "\n\t\t Serial .... " << serial
                  << endl;

        QString temp = "";
        temp += "[";
        temp += QString::number(i);
        temp += "] ";
        temp += rtlsdr_get_device_name(i);
        temp += " ";
        temp += manufact;
        temp += " ";
        temp += product;
        temp += " ";
        temp += serial;

        ui->m_radio_list->addItem(temp);
        ui->statusbar->showMessage("Radio Found", 1000);
    }
    ui->m_num_radios_to_activate->setRange(1, rtlsdr_get_device_count());
    ui->m_num_radios_to_activate->setValue(rtlsdr_get_device_count());
    ui->m_radio_list->addItem(QString("Initialize N Radios"));
    ui->m_radio_list->addItem(QString("Initialize All Radios"));
    ui->m_radio_list->addItem(QString("Run Performance Tests"));


    ui->m_decoding_scheme->setCurrentIndex(0);
}

void RtlSdr_Receiver::on_pushButton_2_clicked()
{
    if(ui == nullptr)
    {
        return;
    }

    Initialize(); // only trigger once


    ui->m_ReceiverActive->setOn(QtLight::BAD);
    m_bps = static_cast<size_t>(ui->m_bps->value());
    ui->m_ReceiverActive->setOn(QtLight::GOOD);
    m_check_the_crc = ui->m_check_crc->isChecked();
    m_pkt_cnt = 0;
    m_prev_pkt_cnt = 0;
    m_cpu_vs_gpu = ui->m_rb_GPU->isChecked(); // GPU true CPU false


    for (size_t i = 0; i < m_NumPoints; ++i)
    {
        m_plot_td_vector_x->operator[](i) = i*m_delta_t*m_decimation;
        m_plot_td_vector_y_real->operator[](i) = 0;
        m_plot_td_vector_y_imag->operator[](i) = 0;
    }

    // reset stuff
    ui->m_data_msg->clear();
    ui->m_ValidCRC->setOn(QtLight::BAD);
    ui->m_ModemLock->setOn(QtLight::BAD);
    ui->m_sample_block_multiplier->setEnabled(false);
    ui->m_radio_list->setEnabled(false);
    ui->pushButton_2->setEnabled(false);

    if(rtlsdr_get_device_count() > 0)
    {
        if(ui->m_radio_list->currentText() == QString("Run Performance Tests"))
        {
            currentTestID = 0;
            emit PerformanceTestSignal(currentTestID);
        }
        else if(ui->m_radio_list->currentText() == QString("Initialize All Radios"))
        {
            Initialize(); // only trigger once

            for(size_t i = 0; i < rtlsdr_get_device_count() && i < MAX_NUM_RADOIOS; i++)
            {
                CreateRadio(static_cast<int>(i));
            }
        }
        else if(ui->m_radio_list->currentText() == QString("Initialize N Radios"))
        {
            for(size_t i = 0; i < ui->m_num_radios_to_activate->value() && i < MAX_NUM_RADOIOS; i++)
            {
                CreateRadio(static_cast<int>(i));
            }
        }
        else
        {
            CreateRadio(0);
        }
    }

    if(m_radio_rtlsdr[0] != nullptr)
    {
        ui->m_ReceiverConnected->setOn(QtLight::GOOD);
    }
    else
    {
        ui->m_ReceiverConnected->setOn(QtLight::BAD);
    }


    if(ui->m_rb_CPU->isChecked())
    {
        // Configure CPU Demodulation
        m_cpu_vs_gpu = false;
    }
    else
    {
        // Configure GPU Demodulation
        m_cpu_vs_gpu = true;
    }
}


void RtlSdr_Receiver::processPendingSystemMonitorDatagrams()
{
    QByteArray datagram;
    struct Message
    {
        short HEADER;
        short TMP35_rtlsdr_0;
        short TMP35_rtlsdr_1;
        short TMP35_rtlsdr_2;
        short TMP35_rtlsdr_3;
        short TMP35_rtlsdr_4;
        short TMP35_rtlsdr_5;
        short TMP35_hackrf;
        short INA219_0_shuntvoltage;
        short INA219_0_busvoltage;
        short INA219_0_current_mA;
        short INA219_0_loadvoltage;
        short INA219_0_power_mW;
        short INA219_1_shuntvoltage;
        short INA219_1_busvoltage;
        short INA219_1_current_mA;
        short INA219_1_loadvoltage;
        short INA219_1_power_mW;
        short INA219_2_shuntvoltage;
        short INA219_2_busvoltage;
        short INA219_2_current_mA;
        short INA219_2_loadvoltage;
        short INA219_2_power_mW;
        short INA219_3_shuntvoltage;
        short INA219_3_busvoltage;
        short INA219_3_current_mA;
        short INA219_3_loadvoltage;
        short INA219_3_power_mW;
        short INA219_4_shuntvoltage;
        short INA219_4_busvoltage;
        short INA219_4_current_mA;
        short INA219_4_loadvoltage;
        short INA219_4_power_mW;
        short INA219_5_shuntvoltage;
        short INA219_5_busvoltage;
        short INA219_5_current_mA;
        short INA219_5_loadvoltage;
        short INA219_5_power_mW;
        short INA219_6_shuntvoltage;
        short INA219_6_busvoltage;
        short INA219_6_current_mA;
        short INA219_6_loadvoltage;
        short INA219_6_power_mW;
        short INA219_7_shuntvoltage;
        short INA219_7_busvoltage;
        short INA219_7_current_mA;
        short INA219_7_loadvoltage;
        short INA219_7_power_mW;
    };

    const Message* msg;


   // using QUdpSocket::readDatagram (API since Qt 4)
   while (udpSocket4.hasPendingDatagrams())
   {
       datagram.resize(int(udpSocket4.pendingDatagramSize()));
       udpSocket4.readDatagram(datagram.data(), datagram.size());
       msg = reinterpret_cast<const Message*>(datagram.constData());

       // step through and push each value onto the appropriate vector
       m_plot_temps[VectorIndexes::RTL0]->append((msg->TMP35_rtlsdr_0)/100.0);
       m_plot_temps[VectorIndexes::RTL0]->pop_front();
       m_plot_temps[VectorIndexes::RTL1]->append((msg->TMP35_rtlsdr_1)/100.0);
       m_plot_temps[VectorIndexes::RTL1]->pop_front();
       m_plot_temps[VectorIndexes::RTL2]->append((msg->TMP35_rtlsdr_2)/100.0);
       m_plot_temps[VectorIndexes::RTL2]->pop_front();
       m_plot_temps[VectorIndexes::RTL3]->append((msg->TMP35_rtlsdr_3)/100.0);
       m_plot_temps[VectorIndexes::RTL3]->pop_front();
       m_plot_temps[VectorIndexes::RTL4]->append((msg->TMP35_rtlsdr_4)/100.0);
       m_plot_temps[VectorIndexes::RTL4]->pop_front();
       m_plot_temps[VectorIndexes::RTL5]->append((msg->TMP35_rtlsdr_5)/100.0);
       m_plot_temps[VectorIndexes::RTL5]->pop_front();
       m_plot_temps[VectorIndexes::HackRF]->append((msg->TMP35_hackrf)/100.0);
       m_plot_temps[VectorIndexes::HackRF]->pop_front();

       m_plot_currents[VectorIndexes::RTL0]->append(abs(msg->INA219_0_current_mA)/100.0);
       m_plot_currents[VectorIndexes::RTL0]->pop_front();
       m_plot_currents[VectorIndexes::RTL1]->append(abs(msg->INA219_1_current_mA)/100.0);
       m_plot_currents[VectorIndexes::RTL1]->pop_front();
       m_plot_currents[VectorIndexes::RTL2]->append(abs(msg->INA219_2_current_mA)/100.0);
       m_plot_currents[VectorIndexes::RTL2]->pop_front();
       m_plot_currents[VectorIndexes::RTL3]->append(abs(msg->INA219_3_current_mA)/100.0);
       m_plot_currents[VectorIndexes::RTL3]->pop_front();
       m_plot_currents[VectorIndexes::RTL4]->append(abs(msg->INA219_4_current_mA)/100.0);
       m_plot_currents[VectorIndexes::RTL4]->pop_front();
       m_plot_currents[VectorIndexes::RTL5]->append(abs(msg->INA219_5_current_mA)/100.0);
       m_plot_currents[VectorIndexes::RTL5]->pop_front();
       m_plot_currents[VectorIndexes::HackRF]->append(abs(msg->INA219_6_current_mA)/100.0);
       m_plot_currents[VectorIndexes::HackRF]->pop_front();

       // Set GUI Temperatures
       ui->m_temp_rtlsdr_0->setText(QString::number(m_plot_temps[VectorIndexes::RTL0]->back(), 'f', 2));
       ui->m_temp_rtlsdr_1->setText(QString::number(m_plot_temps[VectorIndexes::RTL1]->back(), 'f', 2));
       ui->m_temp_rtlsdr_2->setText(QString::number(m_plot_temps[VectorIndexes::RTL2]->back(), 'f', 2));
       ui->m_temp_rtlsdr_3->setText(QString::number(m_plot_temps[VectorIndexes::RTL3]->back(), 'f', 2));
       ui->m_temp_rtlsdr_4->setText(QString::number(m_plot_temps[VectorIndexes::RTL4]->back(), 'f', 2));
       ui->m_temp_rtlsdr_5->setText(QString::number(m_plot_temps[VectorIndexes::RTL5]->back(), 'f', 2));
       ui->m_temp_hackrf->setText(QString::number(m_plot_temps[VectorIndexes::HackRF]->back(), 'f', 2));

       // Set GUI Power
       ui->m_power_rtlsdr_0->setText(QString::number(m_plot_currents[VectorIndexes::RTL0]->back(), 'f', 2));
       ui->m_power_rtlsdr_1->setText(QString::number(m_plot_currents[VectorIndexes::RTL1]->back(), 'f', 2));
       ui->m_power_rtlsdr_2->setText(QString::number(m_plot_currents[VectorIndexes::RTL2]->back(), 'f', 2));
       ui->m_power_rtlsdr_3->setText(QString::number(m_plot_currents[VectorIndexes::RTL3]->back(), 'f', 2));
       ui->m_power_rtlsdr_4->setText(QString::number(m_plot_currents[VectorIndexes::RTL4]->back(), 'f', 2));
       ui->m_power_rtlsdr_5->setText(QString::number(m_plot_currents[VectorIndexes::RTL5]->back(), 'f', 2));
       ui->m_power_hackrf->setText(QString::number(m_plot_currents[VectorIndexes::HackRF]->back(), 'f', 2));
   }
}


void RtlSdr_Receiver::demodulateData(int RADIO_ID)
{
    //////////////////////////////////
    // Call Demodulator
    size_t samples_per_bit = 0;
    if(m_radio_rtlsdr[RADIO_ID])
    {
        // decimate real data down to original -- already downsampled by 5 (1000000 to 2000000)
        // now decimate by the BPS
        samples_per_bit = static_cast<size_t>(floor(10000000/(m_bps*5)));
    }
    else
    {
        std::cout << "Invalid RADIO ???" << std::endl;
        return; // invalid radio if null
    }


    size_t read_number_of_bits = 0;

    if( (m_radio_rtlsdr[RADIO_ID]) )
    {
        if(m_mode == Radio_RTLSDR::AM)
        {
            if(!m_cpu_vs_gpu) // CPU demodulation
            {
                m_block_lock[RADIO_ID].lock();
                read_number_of_bits = m_GlobalPtr->m_am[RADIO_ID]->demodulate(m_block_x[RADIO_ID]->data(), m_block_y[RADIO_ID]->data(), m_data_buffer[RADIO_ID], m_GlobalPtr->m_SampleRate/2, samples_per_bit, m_GlobalPtr->m_bps);
                m_block_lock[RADIO_ID].unlock();
            }
            else
            {

#ifdef GPU_ENABLED
               m_block_lock[RADIO_ID].lock();
               read_number_of_bits = am_gpu_demodulation(m_block_x[RADIO_ID]->data(), m_block_y[RADIO_ID]->data(), m_data_buffer[RADIO_ID], m_SampleRate/2, samples_per_bit);
               m_block_lock[RADIO_ID].unlock();
#endif
            }

            size_t numbits_added = m_GlobalPtr->m_ring_buffer[RADIO_ID]->append(m_GlobalPtr->m_data_buffer[RADIO_ID], read_number_of_bits);
            if(numbits_added != read_number_of_bits)
            {
                std::cout << "WARNING WERE SAMPLING MORE THAN WE CAN HANDLE. LOSING ALOT OF BITS: " << read_number_of_bits - numbits_added << std::endl;
            }

            emit triggerDeserialize(RADIO_ID);
        }
        else if(m_mode == Radio_RTLSDR::FM)
        {
            std::cout << "FM Demodulation" << std::endl;
            if(!m_cpu_vs_gpu) // CPU demodulation
            {
                std::cout << "Got Here " << samples_per_bit << std::endl;
                m_block_lock[RADIO_ID].lock();
                std::cout << "SAMPLES PER BIT " << samples_per_bit << std::endl;
                read_number_of_bits = m_fm[RADIO_ID]->demodulate(m_block_x[RADIO_ID]->data(), m_block_y[RADIO_ID]->data(), m_data_buffer[RADIO_ID], m_SampleRate/2, samples_per_bit);

                m_block_lock[RADIO_ID].unlock();

            }
            else
            {
                m_block_lock[RADIO_ID].lock();
#ifdef GPU_ENABLED
               // TODO read_number_of_bits = am_gpu_demodulation(m_block_x, m_block_y, m_data_buffer, m_SampleRate/2, samples_per_bit);
#endif
                m_block_lock[RADIO_ID].unlock();
            }
            size_t numbits_added = m_ring_buffer[RADIO_ID]->append(m_data_buffer[RADIO_ID], read_number_of_bits);
            if(numbits_added != read_number_of_bits)
            {
                std::cout << "DUDE WERE LOSING ALOT OF BITS: " << read_number_of_bits - numbits_added << std::endl;
            }

          //  emit triggerDeserialize(RADIO_ID);
        }
        else if(m_mode == Radio_RTLSDR::ZEROS || m_mode == Radio_RTLSDR::ONES)
        {
            if(!m_cpu_vs_gpu) // CPU demodulation
            {
                m_block_lock[RADIO_ID].lock();
                for(size_t i = 0; i < m_SampleRate/2; i++)
                {
                    // magnitude should always be between 0 - 127, so decision is based on 63.5 middle ground
                    if( sqrt(m_block_x[RADIO_ID]->operator[](i)*m_block_x[RADIO_ID]->operator[](i) + m_block_y[RADIO_ID]->operator[](i)*m_block_y[RADIO_ID]->operator[](i)) >= AM<RADIO_DATA_TYPE>::THRESHOLD )
                    {
                        m_ones_zeros[RADIO_ID]->increment(1); // got a high
                    }
                    else
                    {
                        m_ones_zeros[RADIO_ID]->decrement(1); // Got a low
                    }
                }
                m_block_lock[RADIO_ID].unlock();
            }
            else
            {

#ifdef GPU_ENABLED
               m_block_lock[RADIO_ID].lock();
               // positive / negative increments are returned by the ones_zeros demoduation
               size_t numOfOnes = ones_zeros_demodulation(m_block_x[RADIO_ID]->data(), m_block_y[RADIO_ID]->data(), m_SampleRate/2);

               m_ones_zeros[RADIO_ID]->increment(numOfOnes);
               m_ones_zeros[RADIO_ID]->decrement(m_SampleRate/2 - numOfOnes);

               m_block_lock[RADIO_ID].unlock();
#endif
            }

            read_number_of_bits+= m_SampleRate/2;
        }
        else if(m_mode == Radio_RTLSDR::BPSK)
        {
            //read_number_of_bits = m_bpsk->demodulate(*m_block_demod, m_data_buffer, m_SampleRate/2, samples_per_bit, m_temp_real, m_temp_imag);
            //m_ring_buffer->append(m_data_buffer, read_number_of_bits);
        }
        else if(m_mode == Radio_RTLSDR::QPSK)
        {
          //  read_number_of_bits = m_qpsk->demodulate(*m_block_demod, m_data_buffer, 512000*2, samples_per_bit);
          //  m_ring_buffer->append(m_data_buffer, read_number_of_bits);
        }
        else
        {
            // Do NOthing
        }
    }

    m_actual_bits_per_second += read_number_of_bits;
}


void RtlSdr_Receiver::LogData()
{
    if(m_radio_rtlsdr[0] == nullptr)
    {
        return;
    }

    // logs data at 2hz
    // csv file format
    fout << std::time(0) << ",";                                // timestamp
    fout << m_num_activate_Radios << ",";                       // num activate radios
    fout << ui->m_sample_block_multiplier->value() << ",";      // Block multiplier
    fout << m_mode << ",";
    fout << m_radio_rtlsdr[0]->get_center_freq() << ",";           // center frequency
    fout << m_bps << ",";
    fout << m_radio_rtlsdr[0]->get_gain_agc()/10.0 << ",";
    fout << m_mode << ",";
    fout << m_check_the_crc << ",";
    fout << m_N_Overprocess << ",";
    fout << ui->m_ModemLock->getState() << ",";
    fout << ui->m_ValidCRC->getState() << ",";
    fout << m_num_validHeaders << ",";
    fout << m_num_validCRCs << ",";
    fout << m_actual_samples_per_second << ",";
    fout << m_pkt_cnt << ",";
    fout << m_bad_bits + m_good_bits << ",";
    fout << m_bad_bits << ",";
    fout << static_cast<double>(m_bad_bits)/static_cast<double>(m_good_bits+m_bad_bits) << ",";
    fout << m_good_bits << ",";

#ifdef GPU_ENABLED
    fout << m_nvidia_usage.m_cpu_filt_load.back() << ",";
    fout << m_nvidia_usage.m_cpu0_filt_load.back() << ",";
    fout << m_nvidia_usage.m_cpu1_filt_load.back() << ",";
    fout << m_nvidia_usage.m_cpu2_filt_load.back() << ",";
    fout << m_nvidia_usage.m_cpu3_filt_load.back() << ",";
    fout << m_nvidia_usage.m_cpu4_filt_load.back() << ",";
    fout << m_nvidia_usage.m_cpu5_filt_load.back() << ",";
#else
    fout << 0 << ",";
    fout << 0 << ",";
    fout << 0 << ",";
    fout << 0 << ",";
    fout << 0 << ",";
    fout << 0 << ",";
    fout << 0 << ",";
#endif

    for(int i = 0; i < VectorIndexes::MAX-1; i++)
    {
        fout << m_plot_temps[i]->back() << ",";
    }

#ifdef GPU_ENABLED
    fout << m_nvidia_usage.m_usedRam_vec.back() << ",";
    fout << m_nvidia_usage.m_gpu_filt_load.back() << ",";
#else
    fout << 0 << ",";
    fout << 0 << ",";
#endif
    fout << std::endl;
}



void RtlSdr_Receiver::deserializeData(int RADIO_ID)
{
    bool validPacket = false;
    bool validCRC = false;
    bool validHeader = false;
    int bad_bits = 0;

    if(RADIO_ID != 0)
    {
        // Only deserialize the first radios data
        return;
    }


    while(m_ring_buffer[RADIO_ID] && (m_ring_buffer[RADIO_ID]->count() > PacketFramer<RADIO_DATA_TYPE>::MAX_PKT_SIZE_BITS))
    {
        int bytes_read = 0;

        if(m_mode == Radio_RTLSDR::AM || m_mode == Radio_RTLSDR::FM || m_mode == Radio_RTLSDR::BPSK || m_mode == Radio_RTLSDR::QPSK)
        {
            bytes_read = m_framer[RADIO_ID]->deserialize(*m_ring_buffer[RADIO_ID], m_temp_pkt[RADIO_ID], MAX_PKT_SIZE, validPacket, validCRC, validHeader, bad_bits);
        }
        else
        {
            break; // shouldnt get here
        }

        // process packet
        if(bytes_read > 0)
        {
            m_ring_buffer[RADIO_ID]->remove(static_cast<size_t>(bytes_read*8)); // Pop good packet off the front now
            memcpy(m_pkt[RADIO_ID], m_temp_pkt[RADIO_ID], MAX_PKT_SIZE);
            emit messageReceived(validPacket, validCRC, validHeader, bad_bits, bytes_read*8);
        }
        else if(bytes_read == 0)
        {
            emit messageReceived(false, false, false, bad_bits, 0);
        }
        else if(bytes_read == -1) // remove a single bit
        {
            m_ring_buffer[RADIO_ID]->remove();
            emit messageReceived(false, validCRC, validHeader, 1, 0);
        }
        else // were negative...
        {
            emit messageReceived(false, false, false, bad_bits, 0);
        }
    }
}


void RtlSdr_Receiver::PerformanceTests(int testID)
{
    std::cout << "Executing Performance Test " << testID << std::endl;
    const int MAX_BLOCK_MULTIPLIER = 10;

    size_t num_radios = testID/2;

    for(int i = 0; i <= num_radios; i++)
    {
        CreateRadio(i);
    }

    // CPU Version
    m_logtimer.stop(); // stop the logger for a bit
    fout.close();
    string filename;
    if(testID%2 == 0)
    {
        filename += "CPU_";
    }
    else
    {
        filename += "GPU_";
    }
    filename += std::to_string((size_t)ui->m_bps->value());
    filename += "_BLOCKSIZE_";
    filename += std::to_string((size_t)ui->m_sample_block_multiplier->value());
    filename += "_Radios_";
    filename += std::to_string(num_radios+1);
    filename += "_logfile.csv";

    fout.open(filename, ios::out | ios::trunc);

    // configure the test
    if(testID %2 == 0)
    {
        ui->m_rb_CPU->setChecked(true);
    }
    else
    {
        ui->m_rb_CPU->setChecked(false);
    }

    for(int j = 0; j <= num_radios; j++)
    {
        ConfigureRadio(j);
    }

    // disable all editable fields
    ui->m_radio_list->setEnabled(false);
    ui->m_num_radios_to_activate->setEnabled(false);
    ui->m_sample_block_multiplier->setEnabled(false);
    ui->pushButton_2->setEnabled(false);
    ui->pushButton->setEnabled(false);
    ui->m_rb_GPU->setCheckable(false);
    ui->m_rb_CPU->setCheckable(false);
    ui->m_freq->setEnabled(false);
    ui->m_bps->setEnabled(false);
    ui->m_gain->setEnabled(false);
    ui->m_decoding_scheme->setEnabled(false);
    ui->m_check_crc->setEnabled(false);


    m_logtimer.start(); // start logger to record data
    m_performanceTest.setSingleShot(true);
    m_performanceTest.setInterval(3000); // 120 seconds of data
    m_performanceTest.start();
    ui->statusbar->showMessage(QString("Performance Test In Progress [LogFile: " + QString(filename.c_str()) + "]  Executing Test ") + QString::number(testID+1) + "/" + QString::number(MaxTestID), 0);
    // exit to allow GUI to run, let the timer trigger the next test
}


void RtlSdr_Receiver::CreateRadio(int RADIO_ID)
{
    if(m_radio_rtlsdr[RADIO_ID] != nullptr)
    {
        std::cout << "Radio appears to already be running.  Not configuring" << std::endl;
        return;
    }

    if(RADIO_ID >= rtlsdr_get_device_count())
    {
        std::cout << "Invalid Radio ID" << std::endl;
        return;
    }
    std::cout << "Creating Radio: " << RADIO_ID << std::endl;
    memset(m_pkt[RADIO_ID], 0, MAX_PKT_SIZE);
    memset(m_temp_pkt[RADIO_ID], 0, MAX_PKT_SIZE);
    m_framer[RADIO_ID]->checkCRC(ui->m_check_crc->isChecked());
    m_ones_zeros[RADIO_ID]->reset();
    m_ring_buffer[RADIO_ID]->reset(); // flush buffers

    m_radio_rtlsdr[RADIO_ID] = new Radio_RTLSDR(static_cast<unsigned int>(RADIO_ID), callbackRoutine);
    m_radio_rtlsdr[RADIO_ID]->set_active(false);

    setMode();
    m_radio_rtlsdr[RADIO_ID]->set_agc_enabled(false); // Leave it always disabled
    m_radio_rtlsdr[RADIO_ID]->set_gain_agc(static_cast<int>(ui->m_gain->value()*10));
    m_radio_rtlsdr[RADIO_ID]->set_center_freq(static_cast<unsigned int>(ui->m_freq->value()*1000000));
    m_radio_rtlsdr[RADIO_ID]->set_sample_rate(2000000); // Samples per second 1/5 of the hackrf
    m_radio_rtlsdr[RADIO_ID]->set_active(true);
    m_num_activate_Radios++;

  //  ui->statusbar->showMessage("Radio Configured " + QString::number(RADIO_ID), 1000);
}

void RtlSdr_Receiver::TearDownRadios()
{
    for(int i = 0; i < MAX_NUM_RADOIOS; i++)
    {
        if(m_radio_rtlsdr[i] != nullptr)
        {
            m_radio_rtlsdr[i]->set_active(false); // disable radio
            delete m_radio_rtlsdr[i];
            m_radio_rtlsdr[i] = nullptr;
        }
    }
}


void RtlSdr_Receiver::ConfigureRadio(int RADIO_ID)
{
    if(ui == nullptr && RADIO_ID < MAX_NUM_RADOIOS)
    {
        return;
    }

    if(m_radio_rtlsdr[RADIO_ID] != nullptr)
    {
        m_radio_rtlsdr[RADIO_ID]->set_active(false);
        ui->m_ReceiverActive->setOn(QtLight::BAD);
        m_bps = static_cast<size_t>(ui->m_bps->value());
        setMode();
        m_radio_rtlsdr[RADIO_ID]->set_agc_enabled(false); // Leave it always disabled
        m_radio_rtlsdr[RADIO_ID]->set_gain_agc(static_cast<int>(ui->m_gain->value()*10));
        m_radio_rtlsdr[RADIO_ID]->set_center_freq(static_cast<unsigned int>(MHZ_TO_HZ(ui->m_freq->value())));
        m_radio_rtlsdr[RADIO_ID]->set_sample_rate(2000000); // Samples per second 1/5 of the hackrf
        ui->m_ReceiverActive->setOn(QtLight::GOOD);
    //    ui->statusbar->showMessage("Radio Configured", 1000);
        m_framer[RADIO_ID]->checkCRC(ui->m_check_crc->isChecked());
        m_check_the_crc = ui->m_check_crc->isChecked();

        // reset stuff
        ui->m_data_msg->clear();
        ui->m_ValidCRC->setOn(QtLight::BAD);
        ui->m_ModemLock->setOn(QtLight::BAD);
        m_ones_zeros[RADIO_ID]->reset();
        m_ring_buffer[RADIO_ID]->reset(); // flush buffers
        m_radio_rtlsdr[RADIO_ID]->set_active(true);
    }

    // Reset the counters
    m_good_bits = 0;
    m_bad_bits = 0;
    m_actual_bits_per_second = 0;
    m_num_validCRCs=0;
    m_num_validHeaders=0;


    if(ui->m_rb_CPU->isChecked())
    {
        // Configure CPU Demodulation
        m_cpu_vs_gpu = false;
    }
    else
    {
        // Configure GPU Demodulation
        m_cpu_vs_gpu = true;
    }
}


void RtlSdr_Receiver::FinishTest()
{
    TearDownRadios();

    // disable all editable fields
    ui->m_radio_list->setEnabled(true);
    ui->m_num_radios_to_activate->setEnabled(true);
    ui->m_sample_block_multiplier->setEnabled(true);
    ui->pushButton_2->setEnabled(true);
    ui->pushButton->setEnabled(true);
    ui->m_rb_GPU->setCheckable(true);
    ui->m_rb_CPU->setCheckable(true);
    ui->m_freq->setEnabled(true);
    ui->m_bps->setEnabled(true);
    ui->m_gain->setEnabled(true);
    ui->m_decoding_scheme->setEnabled(true);
    ui->m_check_crc->setEnabled(true);

    if(currentTestID < MaxTestID)
    {
        currentTestID++;
        emit PerformanceTestSignal(currentTestID);
    }
}

