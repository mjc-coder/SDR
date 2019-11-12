//
// Created by Micheal Cowan on 2019-07-5.
//

#ifndef RADIO_RTLSDR_H
#define RADIO_RTLSDR_H

#include <thread>
#include <mutex>
#include <future>
#include <stdint.h>
#include <RTLSDR/include/rtl-sdr.h>
#include <common/BBP_Block.h>

using namespace std;


class Radio_RTLSDR 
{
public:
    enum GainMode
    {
        Auto,
        Manual
    };

    enum RTLSDR_MODE
    {
        ZEROS = 0,
        ONES = 1,
        AM = 2,
        FM = 3,
        BPSK = 4,
        QPSK = 5
    };

    public:
        Radio_RTLSDR(unsigned int device, std::function<void(unsigned char*, uint32_t, int)> callback);

        ~Radio_RTLSDR();

        bool rtlsdr_found() const;

        // Set / Get Center Frequency
        bool set_center_freq(unsigned long hz);
        unsigned long get_center_freq() const;

        // Set / Get Freq Correction PPM
        bool set_freq_corr_ppm(unsigned long ppm);
        unsigned long get_freq_corr_ppm() const;

        // Set / Get Sample Rate
        bool set_sample_rate(unsigned long ppm);
        unsigned long get_sample_rate() const;

        // Set / Get AGC Enable
        bool set_agc_enabled(bool enable);
        bool get_agc_enabled() const;

        // Set / Get Gain cdb
        bool set_gain_agc(int gain);
        int get_gain_agc() const;

        bool get_is_active() const;
        void set_active(bool active);

    private:
        static void rtlsdr_callback(unsigned char* buf, uint32_t len, void *this_radio);

    private:
        std::function<void(unsigned char*, uint32_t, int)> m_callback;
        rtlsdr_dev_t *m_p_dev;
        unsigned int m_device_index;
        bool m_isActive;
        unsigned long m_samplerate;
        GainMode m_gain_mode;

        // Thread Handles
        promise<void> m_exit_signal;
        future<void> m_future_obj;
        std::thread dongle_read_thread;
        std::mutex m_rw_lock;
};


#endif
