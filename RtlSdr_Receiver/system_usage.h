#ifndef SYSTEM_USAGE_H
#define SYSTEM_USAGE_H

#include <QFile>
#include <unistd.h>
#include <QQueue>
#include <vector>
#include <qcustomplot.h>
#include <thread>
#include <fstream>
#include <future>

using namespace std;


class System_Usage
{
public:
    System_Usage();

    ~System_Usage();

    QVector<double> m_xvector;
    QVector<double> m_usedRam_vec;
    QVector<double> m_cpu_avg_load;
    QVector<double> m_cpu0_avg_load;
    QVector<double> m_cpu1_avg_load;
    QVector<double> m_cpu2_avg_load;
    QVector<double> m_cpu3_avg_load;
    QVector<double> m_cpu4_avg_load;
    QVector<double> m_cpu5_avg_load;
    QVector<double> m_cpu_filt_load;
    QVector<double> m_cpu0_filt_load;
    QVector<double> m_cpu1_filt_load;
    QVector<double> m_cpu2_filt_load;
    QVector<double> m_cpu3_filt_load;
    QVector<double> m_cpu4_filt_load;
    QVector<double> m_cpu5_filt_load;
    QVector<double> m_gpu_filt_load;
    QVector<double> m_cpu_freq_avg_load;
    QVector<double> m_cpu0_freq_avg_load;
    QVector<double> m_cpu1_freq_avg_load;
    QVector<double> m_cpu2_freq_avg_load;
    QVector<double> m_cpu3_freq_avg_load;
    QVector<double> m_cpu4_freq_avg_load;
    QVector<double> m_cpu5_freq_avg_load;
    QVector<double> m_usedGpu_avg_load;

private:
    promise<void> m_exit_signal;
    future<void> m_future_obj;
    std::thread m_tegrastats_thread;
    std::ifstream m_tegrastats_file;

    size_t m_totalRam;
    size_t m_usedRam;
    size_t m_average_cpu;
    size_t m_cpu0;
    size_t m_cpu1;
    size_t m_cpu2;
    size_t m_cpu3;
    size_t m_cpu4;
    size_t m_cpu5;
    size_t m_average_cpu_freq;
    size_t m_cpu0_freq;
    size_t m_cpu1_freq;
    size_t m_cpu2_freq;
    size_t m_cpu3_freq;
    size_t m_cpu4_freq;
    size_t m_cpu5_freq;
    size_t m_gpu;
    size_t m_gpu_freq;

    float m_temp_PLL;
    float m_temp_CPU;
    float m_temp_PMIC;
    float m_temp_GPU;
    float m_temp_AO;
    float m_temp_thermal;
    float m_temp_TBoard;
    float m_temp_BCPU;
    float m_temp_diode;

    bool isJetsonTx2Board;
};

#endif // SYSTEM_USAGE_H
