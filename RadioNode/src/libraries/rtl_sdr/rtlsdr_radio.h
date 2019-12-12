/// @file SDR/RadioNode/src/libraries/rtl_sdr/rtlsdr_radio.h

#ifndef RADIO_RTLSDR_H
#define RADIO_RTLSDR_H

#include <thread>
#include <mutex>
#include <future>
#include <stdint.h>
#include <RTLSDR/include/rtl-sdr.h>
#include <common/BBP_Block.h>

using namespace std;

/// \brief RTLSDR Wrapper around rtl-sdr radio to simplify and utilize c++
class Radio_RTLSDR 
{
public:
    /// \brief Gain Modes for the Radio
    enum GainMode
    {
        Auto,       ///< Auto Gain
        Manual      ///< Manual Gain
    };

    /// \brief RtlSdr Receiver mode
    enum RTLSDR_MODE
    {
        ZEROS = 0,  ///< Receive all zeros
        ONES = 1,   ///< Receive all ones
        AM = 2,     ///< AM mode
        FM = 3,     ///< FM mode
        BPSK = 4,   ///< BPSK mode
        QPSK = 5    ///< QPSK mode
    };

    public:
        /// \brief Constructor
        /// \param device RtlSdr usb device index
        /// \param callback RtlSdr callback that processes data as it is received.
        Radio_RTLSDR(unsigned int device, std::function<void(unsigned char*, uint32_t, int)> callback);

        /// \brief Destructor
        ~Radio_RTLSDR();

        /// \brief was a radio found
        /// \return true if a radio is attached, false otherwise.
        bool rtlsdr_found() const;

        /// \brief Set the center receiver frequency
        /// \param hz Frequency in hertz
        /// \return true if configured correctly, false otherwise.
        bool set_center_freq(unsigned long hz);

        /// \brief Get the configured center frequency
        /// \return Get the currently configured value.
        unsigned long get_center_freq() const;

        /// \brief Set the frequency correction value parts per million
        /// \param ppm PPM value to configure
        /// \return true if configured correctly, false otherwise
        bool set_freq_corr_ppm(unsigned long ppm);

        /// \brief Get the configured frequency correction value
        /// \return Frequency correction value
        unsigned long get_freq_corr_ppm() const;

        /// \brief Set the sample rate.
        /// \param samplerate sample rate in samples per second
        /// \return true if configured correctly, false otherwise
        bool set_sample_rate(unsigned long samplerate);

        /// \brief Get the configured sample rate value
        /// \return value that the receiver is configured too.
        unsigned long get_sample_rate() const;

        /// \brief configure the automatic gain control to be enabled or not.
        /// \param enable true to enable, false otherwise
        /// \return true if it was enabled, false if not.
        bool set_agc_enabled(bool enable);

        /// \brief Gets the automatic gain control value of the radio.
        /// \return true if enabled, false otherwise.
        bool get_agc_enabled() const;

        /// \brief Configure the gain of the automatic gain control.
        /// \param gain gain value to configure
        /// \return true if configured, false otherwise.
        bool set_gain_agc(int gain);

        /// \brief Get the configured value of the gain
        /// \return value of the gain.
        int get_gain_agc() const;

        /// \brief Check if the receiver is active
        /// \return true if receiving, false otherwise
        bool get_is_active() const;

        /// \brief Enable the radio receiver
        /// \param active true to enable, false otherwise.
        void set_active(bool active);

    private:
        /// \brief Internal callback routine
        /// \param buf buffer of bytes from the rtlsdr receiver
        /// \param len length of the buffer
        /// \param this_radio A pointer to reference this class
        static void rtlsdr_callback(unsigned char* buf, uint32_t len, void *this_radio);

    private:
        std::function<void(unsigned char*, uint32_t, int)> m_callback;  ///< user defined callback routine
        rtlsdr_dev_t *m_p_dev; ///< Radio device pointer
        unsigned int m_device_index; ///< Radio device index
        bool m_isActive; ///< Flag if the radio is active
        unsigned long m_samplerate; ///< configured sample rate value
        GainMode m_gain_mode; ///< Radio gain mode

        // Thread Handles
        promise<void> m_exit_signal;    ///< Thread exit signal for clean termination
        future<void> m_future_obj;      ///< Future object for clean termination
        std::thread dongle_read_thread; ///< Radio Read thread
        std::mutex m_rw_lock;           ///< Thread buffer read/write mutex lock
};


#endif
