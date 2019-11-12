//
// Created by Micheal Cowan on 2019-08-04.
//

#ifndef RADIONODE_HACKRF_RADIO_H
#define RADIONODE_HACKRF_RADIO_H

#include <../../third-party/hackrf/host/libhackrf/src/hackrf.h>
#include <stdlib.h>
#include <fstream>
#include <common/Common_Deffinitions.h>
#include <string>
#include <network/PacketFramer.h>
#include <streams/AM.h>
#include <streams/FM.h>
#include <streams/BPSK.h>
#include <streams/QPSK.h>
#include <mutex>
#include <functional>

// 10 Mhz bit rate to transmit
#define OUTPUT_SAMP_RATE    10000000

using namespace std;


class HackRF_radio
{
public:
    static uint64_t unique_word[]; // 0xDEAD 0xBEEF 0xG00D 0xF00D

    enum HACKRF_MODE
    {
        ZEROS = 0,
        ONES = 1,
        TONE50000HZ = 2,
        TONE10000HZ = 3,
        TONE20000HZ = 4,
        AM = 5,
        FM = 6,
        BPSK = 7,
        QPSK = 8
    };

    static const char* mode_hackrf(HACKRF_MODE mode)
    {
        if(mode == ZEROS)
        {
            return "Zeros";
        }
        else if(mode == ONES)
        {
            return "Ones";
        }
        else if(mode == TONE50000HZ)
        {
            return "Tone @ 5KHz";
        }
        else if(mode == TONE10000HZ)
        {
            return "Tone @ 10KHz";
        }
        else if(mode == TONE20000HZ)
        {
            return "Tone @ 20 KHz";
        }
        else if(mode == AM)
        {
            return "AM";
        }
        else if(mode == FM)
        {
            return "FM";
        }
        else if(mode == BPSK)
        {
            return "BPSK";
        }
        else if(mode == QPSK)
        {
            return "QPSK";
        }
        else
        {
            return "UNKNOWN";
        }
    }

public:
    HackRF_radio(hackrf_device_list_t* list, int deviceIndex);

    HackRF_radio(hackrf_device_list_t* list, int deviceIndex, std::function<void(unsigned char*, uint32_t)> callback);

    ~HackRF_radio();



/* Common Functions */
    bool hackrf_found() const;



/* Transmitter Functionality */
    void set_encoding_mode(HACKRF_MODE mode);
    HACKRF_MODE get_encoding_mode(void);

    bool radio_active() const;

    bool hackrf_is_streaming() const;

    void set_freq(unsigned long val);
    unsigned long get_tx_freq() const;

    void set_baud_rate(unsigned long val);
    unsigned long get_baud_rate() const;

    void set_txvga_gain(unsigned int val);
    unsigned int get_txvga_gain() const;



    void set_rf_gain(bool val);
    bool get_rf_gain() const;

    void set_data_message(string msg, int data_type);

    bool transmit_enabled(bool enable);

    static int tx_callback(hackrf_transfer* transfer);

    static int rx_callback(hackrf_transfer* transfer);

/* Receiver Functionality */
    bool receiver_enabled(bool enable);
    void set_baseband_gain(unsigned int val);
    void set_lna_gain(unsigned int val);
    void set_bandwidth(double sample_rate);
    void set_sample_rate(size_t sample_rate);

private:
    void configure_data();

private:
    hackrf_device* p_device;
    std::function<void(unsigned char*, uint32_t)> m_callback;
    HACKRF_MODE m_mode;
    bool m_radio_active;
    unsigned long m_freq;
    unsigned long m_bps;
    unsigned int m_lna_gain;
    bool m_rf_gain;
    string m_msg;
    int m_data_type;
    PacketFramer<RADIO_DATA_TYPE> m_framer;
    BBP_Block* m_block;
    RADIO_DATA_TYPE* m_bit_stream_imag;
    RADIO_DATA_TYPE* m_bit_stream_real;
    uint8_t* m_bit_stream_Quad;
    size_t m_total_bits;
    size_t m_broadcast_index;
    double m_VCO_time;
    ::AM<RADIO_DATA_TYPE> m_modulator_am;
    ::FM<RADIO_DATA_TYPE> m_modulator_fm;
    ::BPSK m_modulator_bpsk;
    ::QPSK m_modulator_qpsk;
    mutex m_rw_lock;
    mutex m_tx_data_msg_lock;
};


#endif //RADIONODE_HACKRF_RADIO_H


