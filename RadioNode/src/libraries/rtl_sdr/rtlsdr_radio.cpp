// Radio Node Primary Code
#include "rtlsdr_radio.h"
#include <iostream>
#include <iomanip>
#include <common/BBP_Block.h>


#define DEFAULT_SAMPLE_RATE		2000000

#define DEFAULT_FREQ            100000000

Radio_RTLSDR::Radio_RTLSDR(unsigned int device, std::function<void(unsigned char*, uint32_t, int)> callback)
: m_callback(callback)
, m_p_dev(nullptr)
, m_device_index(device)
, m_isActive(false)
, m_samplerate(DEFAULT_SAMPLE_RATE)
, m_gain_mode(GainMode::Manual)
, m_future_obj(m_exit_signal.get_future())
, dongle_read_thread(std::thread([this]()
    {
        while(this->m_future_obj.wait_for(chrono::milliseconds(1)) == future_status::timeout)
        {
            if(m_isActive && (m_p_dev!=nullptr))
            {
                rtlsdr_read_async(m_p_dev, rtlsdr_callback, (void*)this, 0, 512000*4); // must be a multiple of 512 should return about 4 times / second
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }))
{
    // Found the device
    rtlsdr_open(&m_p_dev, m_device_index);

    if(rtlsdr_found())
    {
        // Required Reset function ** REQUIRED CALL **
        rtlsdr_reset_buffer(m_p_dev);

        // Configure the Radio with a default configuration
        set_sample_rate(DEFAULT_SAMPLE_RATE);
        set_agc_enabled(true);
        set_center_freq(DEFAULT_FREQ);
        set_freq_corr_ppm(0);
        set_gain_agc(0);
        m_isActive = false;
        std::cout << "Radio Configured" << std::endl;
    }
}

Radio_RTLSDR::~Radio_RTLSDR()
{
    cout << "Destroying " << std::endl;

    m_exit_signal.set_value(); // Trigger Stop Thread
    rtlsdr_cancel_async(m_p_dev); // Stop callback function
    m_isActive = false;
    dongle_read_thread.join();

    // Terminate device if opened
    if(m_p_dev)
    {
        rtlsdr_close(m_p_dev);
    }
}

bool Radio_RTLSDR::rtlsdr_found() const
{
    return (m_p_dev != 0);
}

void Radio_RTLSDR::rtlsdr_callback(unsigned char* buf, uint32_t len, void *ctx)
{
    Radio_RTLSDR* s = (Radio_RTLSDR*)(ctx);
    s->m_rw_lock.lock();

    s->m_callback(buf, len, s->m_device_index);

    s->m_rw_lock.unlock();
}

// Set / Get Center Frequency
bool Radio_RTLSDR::set_center_freq(unsigned long hz)
{
    if(m_p_dev != nullptr && rtlsdr_set_center_freq(m_p_dev, hz) == 0)
    {
        std::cout << "[Radio_RTLSDR] Set Center Frequency: " << hz << std::endl;
        return true;
    }
    std::cout << "[Radio_RTLSDR] ERROR: Unabled to set Set Center Frequency: " << hz << std::endl;
    return false;
}
unsigned long Radio_RTLSDR::get_center_freq() const
{
    if(m_p_dev)
    {
        return rtlsdr_get_center_freq(m_p_dev);
    }
    else
    {
        return 0;
    }
}

// Set / Get Freq Correction PPM
bool Radio_RTLSDR::set_freq_corr_ppm(unsigned long ppm)
{
    if(m_p_dev != nullptr && rtlsdr_set_freq_correction(m_p_dev, ppm) == 0)
    {
        std::cout << "[Radio_RTLSDR] Set Frequency Correction: " << ppm << std::endl;
        return true;
    }
    std::cout << "[Radio_RTLSDR] Error: Unabled to Set Frequency Correction: " << ppm << std::endl;
    return false;
}
unsigned long Radio_RTLSDR::get_freq_corr_ppm() const
{
    if(m_p_dev)
    {
        return rtlsdr_get_freq_correction(m_p_dev);
    }
    else
    {
        return 0;
    }
}

// Set / Get Sample Rate
bool Radio_RTLSDR::set_sample_rate(unsigned long samplerate)
{
    if(rtlsdr_set_sample_rate(m_p_dev, samplerate) == 0)
    {
        std::cout << "[Radio_RTLSDR] Set Sample Rate: " << samplerate << std::endl;
        m_samplerate = samplerate;
        return true;
    }
    std::cout << "[Radio_RTLSDR] ERROR: Unable to Set Sample Rate: " << samplerate << std::endl;
    return false;
}
unsigned long Radio_RTLSDR::get_sample_rate() const
{
    if(m_p_dev)
    {
        return rtlsdr_get_sample_rate(m_p_dev);
    }
    else
    {
        return 0;
    }
}

// Set / Get AGC Enable
bool Radio_RTLSDR::set_agc_enabled(bool enable)
{
    if(m_p_dev != nullptr && rtlsdr_set_agc_mode(m_p_dev, enable) == 0)
    {
        std::cout << "[Radio_RTLSDR] Set AGC Mode to: " << (enable ? "Auto" : "Manual") << std::endl;
        m_gain_mode = (enable) ? GainMode::Auto : GainMode::Manual;
        return true;
    }
    std::cout << "[Radio_RTLSDR] ERROR: Failed to configure AGC Mode" << std::endl;
    return false;
}
bool Radio_RTLSDR::get_agc_enabled() const
{
    return (m_gain_mode == GainMode::Auto) ? true : false;
}

// Set / Get Gain cdb
bool Radio_RTLSDR::set_gain_agc(int gain)
{
    if(rtlsdr_set_tuner_gain(m_p_dev, gain) == 0)
    {
        std::cout << "[Radio_RTLSDR] Gain configured to: " << gain << std::endl;
        return true;
    }
    std::cout << "[Radio_RTLSDR] ERROR: Unable to set gain: " << gain << std::endl;
    return false;
}
int Radio_RTLSDR::get_gain_agc() const
{
    if(m_p_dev)
    {
        return rtlsdr_get_tuner_gain(m_p_dev);
    }
    else
    {
        return 0;
    }
}

bool Radio_RTLSDR::get_is_active() const
{
    return m_isActive;
}

void Radio_RTLSDR::set_active(bool active)
{
    m_isActive = active;
}
