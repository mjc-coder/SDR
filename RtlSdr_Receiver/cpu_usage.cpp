#include "cpu_usage.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <unistd.h>
#include <vector>
#include <chrono>

#define FREQ 0.25
#define SAMPLE_SET 400

CPU_Usage::CPU_Usage()
: m_xvector(SAMPLE_SET)
, m_cpu_avg_load(SAMPLE_SET)
, m_cpu0_avg_load(SAMPLE_SET)
, m_cpu1_avg_load(SAMPLE_SET)
, m_cpu2_avg_load(SAMPLE_SET)
, m_cpu3_avg_load(SAMPLE_SET)
, m_future_obj(m_exit_signal.get_future())
, m_thread(thread([this]()
{
   while(this->m_future_obj.wait_for(chrono::milliseconds(1)) == future_status::timeout)
   {
       updateTimes();
       std::this_thread::sleep_for(std::chrono::milliseconds(250)); // 4 times a seconds
   }
}))
{
    for(size_t i = 0; i < SAMPLE_SET; i++)
    {
        m_xvector[i] = i*FREQ;
    }
}

CPU_Usage::~CPU_Usage()
{
    m_exit_signal.set_value();
    m_thread.join();
}


void CPU_Usage::updateTimes()
{
    std::ifstream proc_stat("/proc/stat");
    char buffer[255];

    unsigned long long int fields[10] = {0};

    // Read Average
    proc_stat.getline(buffer, 254);
    int retval = sscanf (buffer, "cpu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu",
                                &fields[0],
                                &fields[1],
                                &fields[2],
                                &fields[3],
                                &fields[4],
                                &fields[5],
                                &fields[6],
                                &fields[7],
                                &fields[8],
                                &fields[9]);
    m_cpu_inst_load.clear();
    for(size_t i = 0; i < 10; i++) { m_cpu_inst_load.push_back(fields[i]); }



    // Read Average
    proc_stat.getline(buffer, 254);
    retval = sscanf (buffer, "cpu0 %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu",
                                &fields[0],
                                &fields[1],
                                &fields[2],
                                &fields[3],
                                &fields[4],
                                &fields[5],
                                &fields[6],
                                &fields[7],
                                &fields[8],
                                &fields[9]);
    m_cpu0_inst_load.clear();
    for(size_t i = 0; i < 10; i++) { m_cpu0_inst_load.push_back(fields[i]); }

    // Read Average
    proc_stat.getline(buffer, 254);
    retval = sscanf (buffer, "cpu3 %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu",
                                &fields[0],
                                &fields[1],
                                &fields[2],
                                &fields[3],
                                &fields[4],
                                &fields[5],
                                &fields[6],
                                &fields[7],
                                &fields[8],
                                &fields[9]);
    m_cpu1_inst_load.clear();
    for(size_t i = 0; i < 10; i++) { m_cpu1_inst_load.push_back(fields[i]); }

    // Read Average
    proc_stat.getline(buffer, 254);
    retval = sscanf (buffer, "cpu4 %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu",
                                &fields[0],
                                &fields[1],
                                &fields[2],
                                &fields[3],
                                &fields[4],
                                &fields[5],
                                &fields[6],
                                &fields[7],
                                &fields[8],
                                &fields[9]);
    m_cpu2_inst_load.clear();
    for(size_t i = 0; i < 10; i++) { m_cpu2_inst_load.push_back(fields[i]); }

    // Read Average
    proc_stat.getline(buffer, 254);
    retval = sscanf (buffer, "cpu5 %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu %Lu",
                                &fields[0],
                                &fields[1],
                                &fields[2],
                                &fields[3],
                                &fields[4],
                                &fields[5],
                                &fields[6],
                                &fields[7],
                                &fields[8],
                                &fields[9]);
    m_cpu3_inst_load.clear();
    for(size_t i = 0; i < 10; i++) { m_cpu3_inst_load.push_back(fields[i]); }


    float utilization = 0;
    // Calculate values -- total
    if(get_cpu_times(m_cpu_inst_load, m_cpu_prev_idle, m_cpu_prev_total_time, utilization))
    {
        m_cpu_avg_load.pop_front();
        m_cpu_avg_load.push_back(utilization);
    }
    // Calculate values -- cpu 0
    if(get_cpu_times(m_cpu0_inst_load, m_cpu0_prev_idle, m_cpu0_prev_total_time, utilization))
    {
        m_cpu0_avg_load.pop_front();
        m_cpu0_avg_load.push_back(utilization);
    }
    // Calculate values -- cpu 1
    if(get_cpu_times(m_cpu1_inst_load, m_cpu1_prev_idle, m_cpu1_prev_total_time, utilization))
    {
        m_cpu1_avg_load.pop_front();
        m_cpu1_avg_load.push_back(utilization);
    }
    // Calculate values -- cpu 2
    if(get_cpu_times(m_cpu2_inst_load, m_cpu2_prev_idle, m_cpu2_prev_total_time, utilization))
    {
        m_cpu2_avg_load.pop_front();
        m_cpu2_avg_load.push_back(utilization);
    }
    // Calculate values -- cpu 3
    if(get_cpu_times(m_cpu3_inst_load, m_cpu3_prev_idle, m_cpu3_prev_total_time, utilization))
    {
        m_cpu3_avg_load.pop_front();
        m_cpu3_avg_load.push_back(utilization);
    }
}

bool CPU_Usage::get_cpu_times(std::vector<size_t> cpu_times, size_t& prev_idle, size_t& prev_total_time, float& util)
{
    if (cpu_times.size() < 4)
        return false;
    size_t idle_time = cpu_times[3];
    size_t total_time = std::accumulate(cpu_times.begin(), cpu_times.end(), 0);

    const float idle_time_delta = idle_time - prev_idle;
    const float total_time_delta = total_time - prev_total_time;
    util = 100.0 * (1.0 - idle_time_delta / total_time_delta);

    //set previous
    prev_idle = idle_time;
    prev_total_time = total_time;

    return true;
}



