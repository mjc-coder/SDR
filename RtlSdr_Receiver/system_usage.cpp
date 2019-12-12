#include "system_usage.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <unistd.h>
#include <vector>

#include <thread>
using namespace std;

#define FREQ 0.5
#define NUMBER_OF_SAMPLES static_cast<size_t>(100.0/FREQ)


System_Usage::System_Usage()
: m_xvector(NUMBER_OF_SAMPLES)
, m_usedRam_vec(NUMBER_OF_SAMPLES)
, m_cpu_avg_load(NUMBER_OF_SAMPLES)
, m_cpu0_avg_load(NUMBER_OF_SAMPLES)
, m_cpu1_avg_load(NUMBER_OF_SAMPLES)
, m_cpu2_avg_load(NUMBER_OF_SAMPLES)
, m_cpu3_avg_load(NUMBER_OF_SAMPLES)
, m_cpu4_avg_load(NUMBER_OF_SAMPLES)
, m_cpu5_avg_load(NUMBER_OF_SAMPLES)
, m_cpu_filt_load(NUMBER_OF_SAMPLES)
, m_cpu0_filt_load(NUMBER_OF_SAMPLES)
, m_cpu1_filt_load(NUMBER_OF_SAMPLES)
, m_cpu2_filt_load(NUMBER_OF_SAMPLES)
, m_cpu3_filt_load(NUMBER_OF_SAMPLES)
, m_cpu4_filt_load(NUMBER_OF_SAMPLES)
, m_cpu5_filt_load(NUMBER_OF_SAMPLES)
, m_gpu_filt_load(NUMBER_OF_SAMPLES)
, m_cpu_freq_avg_load(NUMBER_OF_SAMPLES)
, m_cpu0_freq_avg_load(NUMBER_OF_SAMPLES)
, m_cpu1_freq_avg_load(NUMBER_OF_SAMPLES)
, m_cpu2_freq_avg_load(NUMBER_OF_SAMPLES)
, m_cpu3_freq_avg_load(NUMBER_OF_SAMPLES)
, m_cpu4_freq_avg_load(NUMBER_OF_SAMPLES)
, m_cpu5_freq_avg_load(NUMBER_OF_SAMPLES)
, m_usedGpu_avg_load(NUMBER_OF_SAMPLES)
, m_future_obj(m_exit_signal.get_future())
, m_tegrastats_thread(thread([this]()
{
   string buffer;
   size_t dummy = 0;
   int retval = 0;
   while(this->m_future_obj.wait_for(chrono::milliseconds(1)) == future_status::timeout &&
         !(this->m_tegrastats_file.is_open()))
   {
       this->m_tegrastats_file.open("/tmp/tegrastats.fifo");
       qDebug() << "Dude file [/tmp/tegrastats.fifo] not open....  " <<
             (this->m_future_obj.wait_for(chrono::milliseconds(1)) == future_status::timeout)
              << "  " << this->m_tegrastats_file.is_open();
   }

    qDebug() << "Dude the file is open.. time to read\n";

    // Test which board we are on
    // Nano or Jetson -- the tegrastats dumps out different strings for each
    // RAM 1436/7852MB (lfb 1403x4MB) SWAP 0/3926MB (cached 0MB)
    // CPU [11%@345,off,off,5%@345,3%@345,7%@345] EMC_FREQ 4%@665
    // GR3D_FREQ 0%@216 APE 150 PLL@38.5C MCPU@38.5C PMIC@100C Tboard@36C
    // GPU@36C BCPU@38.5C thermal@37.5C Tdiode@36C VDD_SYS_GPU 152/153
    // VDD_SYS_SOC 456/453 VDD_4V0_WIFI 0/12 VDD_IN 2057/2396
    // VDD_SYS_CPU 152/438 VDD_SYS_DDR 443/477

    std::getline(this->m_tegrastats_file, buffer);
    std::cout << buffer << std::endl;

    retval = sscanf (buffer.c_str(),
      "RAM %zu/%zuMB (lfb %zux%zuMB) SWAP %zu/%zuMB (cached %zuMB) " \
      "CPU [%zu%%@%zu,%zu%%@%zu,%zu%%@%zu,%zu%%@%zu,%zu%%@%zu,%zu%%@%zu] EMC_FREQ %zu%%@%zu "\
      "GR3D_FREQ %zu%%@%zu APE %zu PLL@%fC MCPU@%fC PMIC@%fC Tboard@%fc " \
      "GPU@%fC BCPU@%fC thermal@%fC Tdiode@%fC VDD_SYS_GPU %zu/%zu " \
      "VDD_SYS_SOC %zu/%zu VDD_4V0_WIFI %zu/%zu VDD_IN %zu/%zu " \
      "VDD_SYS_CPU %zu/%zu VDD_SYS_DDR %zu/%zu",
       &(this->m_usedRam), &(this->m_totalRam), &dummy, &dummy, &(dummy), &(dummy), &(dummy),
       &(this->m_cpu0), &(this->m_cpu0_freq), &(this->m_cpu1), &(this->m_cpu1_freq),&(this->m_cpu2), &(this->m_cpu2_freq),&(this->m_cpu3), &(this->m_cpu3_freq),&(this->m_cpu4), &(this->m_cpu4_freq),&(this->m_cpu5), &(this->m_cpu5_freq),&(dummy), &(dummy),
       &(dummy), &(dummy),&(dummy), &(this->m_temp_PLL),&(this->m_temp_CPU), &(this->m_temp_PMIC),&(dummy),
       &(this->m_temp_GPU),&(dummy),&(this->m_temp_AO),&(this->m_temp_thermal),&(dummy),&(dummy),
       &(dummy), &(dummy), &(dummy), &(dummy), &(dummy), &(dummy),
       &(dummy),&(dummy),&(dummy),&(dummy));

     qDebug() << " Decisions " << retval;

    if(retval == 4) // Nano Board
    {
        isJetsonTx2Board = false;
    }
    else
    {
       isJetsonTx2Board = true;
    }


    while(this->m_future_obj.wait_for(chrono::milliseconds(1)) == future_status::timeout)
    {
        while(std::getline(this->m_tegrastats_file, buffer) && (this->m_future_obj.wait_for(chrono::milliseconds(1)) == future_status::timeout))
        {
            if(!isJetsonTx2Board) // NANO
            {
               // qDebug() << "  NANO";
                retval = sscanf (buffer.c_str(),
                    "RAM %zu/%zuMB (lfb %zux%zuMB) IRAM %zu/%zukB(lfb %zukB) " \
                    "CPU [%zu%%@%zu,%zu%%@%zu,%zu%%@%zu,%zu%%@%zu] " \
                    "EMC_FREQ %zu%%@%zu GR3D_FREQ %zu%%@%zu APE %zu " \
                    "PLL@%fC CPU@%fC PMIC@%fC GPU@%fC AO@%fC " \
                    "thermal@%fC POM_5V_IN %zu/%zu " \
                    "POM_5V_GPU %zu/%zu POM_5V_CPU %zu/%zu",
                     &(this->m_usedRam), &(this->m_totalRam), &dummy, &dummy,
                     &(dummy), &(dummy), &(dummy),
                     &(this->m_cpu0), &(this->m_cpu0_freq),
                     &(this->m_cpu1), &(this->m_cpu1_freq),
                     &(this->m_cpu2), &(this->m_cpu2_freq),
                     &(this->m_cpu3), &(this->m_cpu3_freq),
                     &(dummy), &(dummy), &(this->m_gpu), &(this->m_gpu_freq),
                     &(dummy), &(this->m_temp_PLL),
                     &(this->m_temp_CPU), &(this->m_temp_PMIC),
                     &(this->m_temp_GPU), &(this->m_temp_AO),
                     &(this->m_temp_thermal),
                     &(dummy), &(dummy), &(dummy), &(dummy), &(dummy), &(dummy) );
                m_cpu4 = 0;
                m_cpu5 = 0;
                m_cpu4_freq = 0;
                m_cpu5_freq = 0;
            }
            else  // Jetson TX2
            {
               // qDebug() << "  Jetson TX2";
                retval = sscanf (buffer.c_str(),
                  "RAM %zu/%zuMB (lfb %zux%zuMB) SWAP %zu/%zuMB (cached %zuMB) " \
                  "CPU [%zu%%@%zu,%zu%%@%zu,%zu%%@%zu,%zu%%@%zu,%zu%%@%zu,%zu%%@%zu] EMC_FREQ %zu%%@%zu "\
                  "GR3D_FREQ %zu%%@%zu APE %zu PLL@%fC MCPU@%fC PMIC@%fC Tboard@%fc " \
                  "GPU@%fC BCPU@%fC thermal@%fC Tdiode@%fC VDD_SYS_GPU %zu/%zu " \
                  "VDD_SYS_SOC %zu/%zu VDD_4V0_WIFI %zu/%zu VDD_IN %zu/%zu " \
                  "VDD_SYS_CPU %zu/%zu VDD_SYS_DDR %zu/%zu",
                   &(this->m_usedRam), &(this->m_totalRam), &dummy, &dummy, &(dummy), &(dummy), &(dummy),
                   &(this->m_cpu0), &(this->m_cpu0_freq), &(this->m_cpu1), &(this->m_cpu1_freq),&(this->m_cpu2), &(this->m_cpu2_freq),&(this->m_cpu3), &(this->m_cpu3_freq),&(this->m_cpu4), &(this->m_cpu4_freq),&(this->m_cpu5), &(this->m_cpu5_freq),&(dummy), &(dummy),
                   &(this->m_gpu), &(this->m_gpu_freq) ,&(dummy), &(this->m_temp_PLL),&(this->m_temp_CPU), &(this->m_temp_PMIC),&(dummy),
                   &(this->m_temp_GPU),&(dummy),&(this->m_temp_AO),&(this->m_temp_thermal),&(dummy),&(dummy),
                   &(dummy), &(dummy), &(dummy), &(dummy), &(dummy), &(dummy),
                   &(dummy),&(dummy),&(dummy),&(dummy));

            }

           if(retval == EOF)
           {
               this->m_tegrastats_file.clear();
               break;
           }



           if(retval == 24 || retval == 32)
           {
                m_usedRam_vec.pop_front();
                m_usedRam_vec.push_back(static_cast<double>(m_usedRam)/static_cast<double>(m_totalRam)*100.0); // convert to percent

                m_cpu_avg_load.pop_front();
                m_cpu_avg_load.push_back((this->m_cpu0 + this->m_cpu1 + this->m_cpu2 + this->m_cpu3 + this->m_cpu4 + this->m_cpu5 + 0.001)/6.0);
                m_cpu0_avg_load.pop_front();
                m_cpu0_avg_load.push_back(this->m_cpu0);
                m_cpu1_avg_load.pop_front();
                m_cpu1_avg_load.push_back(this->m_cpu1);
                m_cpu2_avg_load.pop_front();
                m_cpu2_avg_load.push_back(this->m_cpu2);
                m_cpu3_avg_load.pop_front();
                m_cpu3_avg_load.push_back(this->m_cpu3);
                m_cpu4_avg_load.pop_front();
                m_cpu4_avg_load.push_back(this->m_cpu4);
                m_cpu5_avg_load.pop_front();
                m_cpu5_avg_load.push_back(this->m_cpu5);

                const double alpha = 0.20;

               // y[t]=αy[t−1]+(1−α)x[t]

                double avg = alpha * m_cpu_filt_load.back()  + (1.0-alpha) * (m_cpu_avg_load.back());
                double f0  = alpha * m_cpu0_filt_load.back() + (1.0-alpha) * (m_cpu0_avg_load.back());
                double f1  = alpha * m_cpu1_filt_load.back() + (1.0-alpha) * (m_cpu1_avg_load.back());
                double f2  = alpha * m_cpu2_filt_load.back() + (1.0-alpha) * (m_cpu2_avg_load.back());
                double f3  = alpha * m_cpu3_filt_load.back() + (1.0-alpha) * (m_cpu3_avg_load.back());
                double f4  = alpha * m_cpu4_filt_load.back() + (1.0-alpha) * (m_cpu4_avg_load.back());
                double f5  = alpha * m_cpu5_filt_load.back() + (1.0-alpha) * (m_cpu5_avg_load.back());
                double gpu = alpha * m_gpu_filt_load.back()  + (1.0-alpha) * (m_usedGpu_avg_load.back());

                m_cpu_filt_load.pop_front();
                m_cpu_filt_load.push_back(avg);
                m_cpu0_filt_load.pop_front();
                m_cpu0_filt_load.push_back(f0);
                m_cpu1_filt_load.pop_front();
                m_cpu1_filt_load.push_back(f1);
                m_cpu2_filt_load.pop_front();
                m_cpu2_filt_load.push_back(f2);
                m_cpu3_filt_load.pop_front();
                m_cpu3_filt_load.push_back(f3);
                m_cpu4_filt_load.pop_front();
                m_cpu4_filt_load.push_back(f4);
                m_cpu5_filt_load.pop_front();
                m_cpu5_filt_load.push_back(f5);
                m_gpu_filt_load.pop_front();
                m_gpu_filt_load.push_back(gpu);



                m_cpu_freq_avg_load.pop_front();
                m_cpu_freq_avg_load.push_back((this->m_cpu0 + this->m_cpu1 + this->m_cpu2 + this->m_cpu3 + this->m_cpu4 + this->m_cpu5 + 0.001)/6.0);
                m_cpu0_freq_avg_load.pop_front();
                m_cpu0_freq_avg_load.push_back(this->m_cpu0_freq);
                m_cpu1_freq_avg_load.pop_front();
                m_cpu1_freq_avg_load.push_back(this->m_cpu1_freq);
                m_cpu2_freq_avg_load.pop_front();
                m_cpu2_freq_avg_load.push_back(this->m_cpu2_freq);
                m_cpu3_freq_avg_load.pop_front();
                m_cpu3_freq_avg_load.push_back(this->m_cpu3_freq);
                m_cpu4_freq_avg_load.pop_front();
                m_cpu4_freq_avg_load.push_back(this->m_cpu4_freq);
                m_cpu5_freq_avg_load.pop_front();
                m_cpu5_freq_avg_load.push_back(this->m_cpu5_freq);

                m_usedGpu_avg_load.pop_front();
                m_usedGpu_avg_load.push_back(this->m_gpu);
           }
           else
           {
                qDebug() << "  " << retval;
           }


            this->m_tegrastats_file.clear();
       }
   }
}))
{
    for(size_t i = 0; i < NUMBER_OF_SAMPLES; i++)
    {
        m_xvector[i] = i*FREQ;
    }
}

System_Usage::~System_Usage()
{
    m_exit_signal.set_value();
    m_tegrastats_thread.join();
}



