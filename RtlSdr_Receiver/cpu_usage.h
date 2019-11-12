#ifndef CPU_USAGE_H
#define CPU_USAGE_H

#include <QFile>
#include <unistd.h>
#include <QQueue>
#include <vector>
#include <qcustomplot.h>
#include <thread>
#include <future>

using namespace std;


class CPU_Usage
{
public:
    CPU_Usage();

    ~CPU_Usage();

    void updateTimes();

    QVector<double> m_xvector;
    QVector<double> m_cpu_avg_load;
    QVector<double> m_cpu0_avg_load;
    QVector<double> m_cpu1_avg_load;
    QVector<double> m_cpu2_avg_load;
    QVector<double> m_cpu3_avg_load;

    bool get_cpu_times(std::vector<size_t> cpu_times, size_t& prev_idle, size_t& prev_total_time, float& util);


private:
    promise<void> m_exit_signal;
    future<void> m_future_obj;
    std::thread m_thread;

    vector<size_t> m_cpu_inst_load;
    vector<size_t> m_cpu0_inst_load;
    vector<size_t> m_cpu1_inst_load;
    vector<size_t> m_cpu2_inst_load;
    vector<size_t> m_cpu3_inst_load;

    size_t m_cpu_prev_idle;
    size_t m_cpu_prev_total_time;
    size_t m_cpu0_prev_idle;
    size_t m_cpu0_prev_total_time;
    size_t m_cpu1_prev_idle;
    size_t m_cpu1_prev_total_time;
    size_t m_cpu2_prev_idle;
    size_t m_cpu2_prev_total_time;
    size_t m_cpu3_prev_idle;
    size_t m_cpu3_prev_total_time;
};

#endif // CPU_USAGE_H
