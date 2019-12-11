/// @file SDR/RadioNode/src/libraries/hackrf/HackRF_radio.h

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

using namespace std;

/// \brief Hack RF Wrapper to extract functionality into a simpler c++ interface
class HackRF_radio
{
public:
    static uint64_t unique_word[]; ///< 0xDEAD 0xBEEF 0xG00D 0xF00D

    /// \brief HackRF Transmit Mode Enumeration
    enum HACKRF_MODE
    {
        ZEROS = 0,                  ///< Transmit all Zeros
        ONES = 1,                   ///< Trasnmit all 127 (Max Amplitude)
        TONE50000HZ = 2,            ///< Transmit a 50 khz Tone
        TONE10000HZ = 3,            ///< Transmit a 10 khz Tone
        TONE20000HZ = 4,            ///< Transmit a 20 khz Tone
        AM = 5,                     ///< Transmit an ASK amplitude modulated binary signal
        FM = 6,                     ///< Transmit a FSK frequency modulated binary signal
        BPSK = 7,                   ///< Transmit a BPSK modulated binary signal
        QPSK = 8                    ///< Transmit a QPSK modulated binary signal
    };

    /// Convert mode to a string
    ///
    /// \param[in] mode     HackRF Mode to convert
    /// \return Return the appropriate string.
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
    /// \brief Creates a running instance of a HackRF in a Transmit configuration
    /// \param list A pre-generated hackrf device list that the HackRF interface uses to instantiate the radio.
    /// \param deviceIndex  Index of the radio in the list to initialize.  Invalid entries are not supported.
    HackRF_radio(hackrf_device_list_t* list, int deviceIndex);

    /// \brief Creates a running instance of a HackRF in a Receive configuration.
    /// \param list A pre-generated hackrf device list that the HackRF interface uses to instantiate the radio.
    /// \param deviceIndex Index of the radio in the list to initialize.  Invalid entries are not supported.
    /// \param callback Callback routine that the user can specifie that the radio triggers to return read data from
    /// the radio.
    HackRF_radio(hackrf_device_list_t* list, int deviceIndex, std::function<void(unsigned char*, uint32_t)> callback);

    /// \brief Destroys the running instance of the radio.
    ~HackRF_radio();


    /// \brief Check if the hackrf radio is found.
    /// \return true if the radio is found, false otherwise.
    bool hackrf_found() const;


    /// \brief Configure the encoding mode.
    /// \details This is a Transmit only functionality.  Has no effect on a receiver configuration.
    /// \param mode Mode to configure the transmitter encoding.
    void set_encoding_mode(HACKRF_MODE mode);

    /// \brief Get the configured encoding mode.
    /// \details This is a Transmit only functionality.  Value returned will have no meaning in a receiver configuration.
    /// \return Currently configured encoding mode.
    HACKRF_MODE get_encoding_mode(void);

    /// \brief Check if the Radio is connected and communicating with the system.
    /// \return True if the radio is active, false otherwise.
    bool radio_active() const;

    /// \brief Check if the Radio is actively streaming data.
    /// \return True if the radio is active, false otherwise.
    bool hackrf_is_streaming() const;

    /// \brief Configure Frequency for radio.
    /// \details Receiver / Transmitter function.
    /// \param val Frequency in Hz to configure the radio.
    void set_freq(unsigned long val);

    /// \brief Get the Frequency the radio is configured for.
    /// \details Receiver / Transmitter Function
    /// \return Returns the value in Hz the radio is configured.
    unsigned long get_tx_freq() const;

    /// \brief Configure the Baud Radio for encoded signal.
    /// \details Receiver / Transmitter Function
    /// \param val Baud rate in bits per second
    void set_baud_rate(unsigned long val);

    /// \brief Get the current Baud Rate of the encoded signal.
    /// \details Receiver / Transmitter Function
    /// \return Returns the configured bps in bits per second.
    unsigned long get_baud_rate() const;

    /// \brief Configures the Internal Gain (IF)
    /// \details Transmitter Function
    /// \param val Value in centi-db to configure.
    void set_txvga_gain(unsigned int val);

    /// \brief Gets the configured Internal Gain (IF)
    /// \details Transmitter Function
    /// \return Returns the value in centi-db.
    unsigned int get_txvga_gain() const;

    /// \brief Configure the RF Gain
    /// \details Transmitter Function
    /// \param val value to configure in Centi-db
    void set_rf_gain(bool val);

    /// \brief Get the configured the RF Gain
    /// \details Transmitter Function
    /// \return The value configured in centi-db
    bool get_rf_gain() const;

    /// \brief Configure the data message to transmit
    /// \details Transmitter Function
    /// \param msg Ascii message to transmit over the link
    /// \param data_type [not implemented] configures the User,
    /// Incrementing Numbers, Incrementing alphabet data message.
    void set_data_message(string msg, int data_type);

    /// \brief Enable the transmitter.
    /// \details Transmitter function.
    /// \param enable true turns on transmitter, false stops.
    /// \return true if transmitter was enabled, false if it was not.
    bool transmit_enabled(bool enable);

    /// \brief Transmitter callback routine, triggered by the radio to send data.
    /// \param transfer hackrf_transfer object
    /// \return 0 if successful, -1 on an error.
    static int tx_callback(hackrf_transfer* transfer);

    /// \brief Receiver callback routine, triggered by the radio to receive data.
    /// \param transfer hackrf transfer object
    /// \return 0 if successful, -1 on an error.
    static int rx_callback(hackrf_transfer* transfer);

    /// \brief Is the receiver enabled.
    /// \details Receiver only function.
    /// \param enable Start the receiver if true, disable if false.
    /// \return Returns true is successful, false otherwise.
    bool receiver_enabled(bool enable);

    /// \brief Configure the Receiver baseband gain.
    /// \details Receiver Function.
    /// \param val  Value in centi-db to configure the baseband gain.
    void set_baseband_gain(unsigned int val);

    /// \brief Configures the LNA Receiver Gain
    /// \details Receiver Function
    /// \param val  Value in centi-db to configure the baseband gain.
    void set_lna_gain(unsigned int val);

    /// \brief Configures the Hackrf bandwidth based on the sample rate
    /// \details Receiver Function
    /// \param sample_rate The sample rate the radio is configured to.
    void set_bandwidth(double sample_rate);

    /// \brief Configures the Hackrf Sample Rate
    /// \details Receiver / Transmitter Function
    /// \param sample_rate Sample Rate to configure the Hackrf too.
    void set_sample_rate(size_t sample_rate);

private:
    /// \brief Configures the data message off the string previously provided.
    void configure_data();

private:
    hackrf_device* p_device;    ///< hackrf_device interface
    std::function<void(unsigned char*, uint32_t)> m_callback; ///< Transmitter Callback
    HACKRF_MODE m_mode; ///< Transmitter mode
    bool m_radio_active; ///< flag that shows if the radio is active
    unsigned long m_freq; ///< frequency of the radio in hz
    unsigned long m_bps; ///< bits per second
    unsigned int m_lna_gain; ///< configured lna gain in centi-db
    bool m_rf_gain; ///< true if the RF Gain is enabled
    string m_msg; ///< data message
    int m_data_type; ///< user data = 0, incrementing numbers = 1, incrementing alphabet = 2
    PacketFramer<RADIO_DATA_TYPE> m_framer; ///< Packet Framer
    BBP_Block* m_block; ///< BBP Block for processing and passing complex data
    RADIO_DATA_TYPE* m_bit_stream_imag; ///< imaginary bit stream samples
    RADIO_DATA_TYPE* m_bit_stream_real; ///< real bit stream samples
    uint8_t* m_bit_stream_Quad; ///< Bit stream Quadriture data
    size_t m_total_bits; ///< Total numberr of bits processed
    size_t m_broadcast_index; ///< Broadcast index
    double m_VCO_time; ///< Variable Control Oscillator time value
    ::AM<RADIO_DATA_TYPE> m_modulator_am; ///< AM modulator
    ::FM<RADIO_DATA_TYPE> m_modulator_fm; ///< FM modulator
    ::BPSK m_modulator_bpsk; ///< BPSK Modulator
    ::QPSK m_modulator_qpsk; ///< QPSK Modulator
    mutex m_rw_lock; ///< RW Lock on buffers
    mutex m_tx_data_msg_lock; ///< TX Data Message Lock
};


#endif //RADIONODE_HACKRF_RADIO_H


