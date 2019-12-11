/// @file SDR/RadioNode/src/libraries/hackrf/HackRF_radio.cpp

#include <hackrf/HackRF_radio.h>
#include <DigitalSignalProcessing/Resample.h>
#include <../../third-party/hackrf/host/libhackrf/src/hackrf.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <FIR-filter-class/filt.h>
#include <streams/FM.h>

/* 10MHz default sample rate */
#define DEFAULT_SAMPLE_RATE_HZ      10000000                    ///< Default Transmit Sample Rate
#define SIGNAL_FREQ                 11000 * (1/(2 * PI))        ///< Default Signal Frequency
#define DELTA_T                     1.0/DEFAULT_SAMPLE_RATE_HZ  ///< Default Sample Rate
#define AMPLITUDE 127.0 ///< MAX Transmission amplitude.

uint64_t HackRF_radio::unique_word[] =
{1,1,0,1,1,1,1,0,
 1,0,1,0,1,1,0,1,
 1,0,1,1,1,1,1,0,
 1,1,1,0,1,1,1,1,
 0,1,1,0,0,0,0,0,
 0,0,0,0,1,1,0,1,
 1,1,1,1,0,0,0,0,
 0,0,0,0,1,1,0,1};

HackRF_radio::HackRF_radio(hackrf_device_list_t* list, int deviceIndex)
: p_device(nullptr)
, m_callback(nullptr)
, m_mode(FM)
, m_radio_active(false)
, m_freq(800000000)
, m_bps(100000)
, m_lna_gain(39)
, m_rf_gain(true)
, m_msg("Hello World")
, m_block(nullptr)
, m_bit_stream_imag(nullptr)
, m_bit_stream_real(nullptr)
, m_bit_stream_Quad(nullptr)
, m_total_bits(0)
, m_broadcast_index(0)
, m_VCO_time(0)
, m_modulator_fm(10000000.0,2000000.0)
, m_modulator_bpsk(0x600DF00DDEADBEEF, 64)  // BEEF 11011 1110 1110 1111
{
    m_block = new BBP_Block(12088);

    int result = hackrf_init();
    if( result != HACKRF_SUCCESS )
    {
        fprintf(stderr, "hackrf_init() failed: (%d)\n", result);
    }

    result = hackrf_device_list_open(list, deviceIndex, &p_device);
    if( result != HACKRF_SUCCESS )
    {
        fprintf(stderr, "hackrf_open() failed: (%d)\n", result);
    }
    hackrf_set_hw_sync_mode(p_device, 0); // disable
    hackrf_set_sample_rate(p_device, DEFAULT_SAMPLE_RATE_HZ);
}


HackRF_radio::HackRF_radio(hackrf_device_list_t* list, int deviceIndex, std::function<void(unsigned char*, uint32_t)> callback)
: p_device(nullptr)
, m_callback(callback)
, m_mode(ZEROS)                 /* Not Used */
, m_radio_active(false)      /* Not Used */
, m_freq(100000000)
, m_bps(100000)
, m_lna_gain(39)
, m_rf_gain(true)
, m_msg("Hello World")           /* Not Used */
, m_block(nullptr)                 /* Not Used */
, m_bit_stream_imag(nullptr)       /* Not Used */
, m_bit_stream_real(nullptr)       /* Not Used */
, m_bit_stream_Quad(nullptr)        /* Not Used */
, m_total_bits(0)                  /* Not Used */
, m_broadcast_index(0)             /* Not Used */
, m_VCO_time(0)                    /* Not Used */
, m_modulator_fm(2000000, 2000000)
, m_modulator_bpsk(0x600DF00DDEADBEEF, 64)  // BEEF 11011 1110 1110 1111
{
    int result = hackrf_init();
    if( result != HACKRF_SUCCESS )
    {
        fprintf(stderr, "hackrf_init() failed: (%d)\n", result);
    }

    result = hackrf_device_list_open(list, deviceIndex, &p_device);
    if( result != HACKRF_SUCCESS )
    {
        fprintf(stderr, "hackrf_open() failed: (%d)\n", result);
    }
    hackrf_set_hw_sync_mode(p_device, 0); // disable
    hackrf_set_sample_rate(p_device, DEFAULT_SAMPLE_RATE_HZ);
}


HackRF_radio::~HackRF_radio()
{
    int result = 0;

    result = hackrf_stop_tx(p_device);
    if( result != HACKRF_SUCCESS )
    {
        fprintf(stderr, "hackrf_stop_tx() failed: (%d)\n", result);
    }
    else
    {
        fprintf(stderr, "hackrf_stop_tx() done\n");
    }

    result = hackrf_close(p_device);
    if(result != HACKRF_SUCCESS) {
        fprintf(stderr, "hackrf_close() failed: (%d)\n",  result);
    } else {
        fprintf(stderr, "hackrf_close() done\n");
    }

    hackrf_exit();
    fprintf(stderr, "hackrf_exit() done\n");
    delete m_block;

    if(m_bit_stream_real)
    {
        delete[] m_bit_stream_Quad;
        delete[] m_bit_stream_imag;
        delete[] m_bit_stream_real;
    }
}


int HackRF_radio::rx_callback(hackrf_transfer* transfer)
{
    HackRF_radio* r = reinterpret_cast<HackRF_radio*>(transfer->rx_ctx);

    r->m_rw_lock.lock();
    if(transfer->valid_length > 0)
    {
        r->m_callback(transfer->buffer, static_cast<unsigned int>(transfer->valid_length));
    }
    r->m_rw_lock.unlock();
    return 0;
}

int HackRF_radio::tx_callback(hackrf_transfer* transfer)
{
    HackRF_radio* r = reinterpret_cast<HackRF_radio*>(transfer->tx_ctx);
    int bytes_to_read = transfer->valid_length;
    int dst_index = 0;

    if(r->m_tx_data_msg_lock.try_lock())
    {
        if(r->m_mode == ONES)
        {
            memset(transfer->buffer, AMPLITUDE, bytes_to_read);
        }
        else if(r->m_mode == ZEROS)
        {
            memset(transfer->buffer, 0, bytes_to_read);
        }
        else if(r->m_mode == TONE10000HZ || r->m_mode == TONE20000HZ || r->m_mode == TONE50000HZ || r->m_mode == AM || r->m_mode == FM || r->m_mode == BPSK || r->m_mode == QPSK)
        {
            while(dst_index < bytes_to_read)
            {
                int remaining_bytes_in_msg_buffer = r->m_total_bits - r->m_broadcast_index;
                int remaining_bytes_in_dst_buffer = bytes_to_read - dst_index;


                if(remaining_bytes_in_msg_buffer < remaining_bytes_in_dst_buffer)
                {
                    memcpy(&transfer->buffer[dst_index], &r->m_bit_stream_Quad[r->m_broadcast_index], remaining_bytes_in_msg_buffer);
                    r->m_broadcast_index = 0;
                    dst_index+= remaining_bytes_in_msg_buffer;
                }
                else if(remaining_bytes_in_msg_buffer == remaining_bytes_in_dst_buffer)
                {
                    memcpy(&transfer->buffer[dst_index], &r->m_bit_stream_Quad[r->m_broadcast_index], remaining_bytes_in_msg_buffer);
                    r->m_broadcast_index = 0; // reset to the header
                    dst_index+= remaining_bytes_in_msg_buffer;
                }
                else // remaining_bytes_in_msg_buffer < remaining_bytes_in_dst_buffer
                {
                    memcpy(&transfer->buffer[dst_index], &r->m_bit_stream_Quad[r->m_broadcast_index], remaining_bytes_in_dst_buffer);
                    r->m_broadcast_index += remaining_bytes_in_dst_buffer;
                    dst_index+= remaining_bytes_in_msg_buffer;
                }
            }
        }
        r->m_tx_data_msg_lock.unlock();
        return 0;
    }

    return -1;
}

void HackRF_radio::set_sample_rate(size_t sample_rate)
{
    hackrf_set_sample_rate(p_device, sample_rate);
}


bool HackRF_radio::hackrf_found() const
{
    return (p_device != nullptr);
}

bool HackRF_radio::hackrf_is_streaming() const
{
    return (p_device != nullptr && (::hackrf_is_streaming(p_device) == HACKRF_TRUE));
}


void HackRF_radio::set_encoding_mode(HACKRF_MODE mode)
{
    std::cout << "[HackRF_radio] Encoding mode: " << mode_hackrf(mode) << std::endl;
    m_mode = mode;
}

HackRF_radio::HACKRF_MODE HackRF_radio::get_encoding_mode(void)
{
    return m_mode;
}

bool HackRF_radio::radio_active() const
{
    return (p_device != nullptr) && m_radio_active;
}

void HackRF_radio::set_freq(unsigned long val)
{
    if(hackrf_set_freq(p_device, val) == HACKRF_SUCCESS)
    {
        std::cout << "[HackRF_radio] TX Frequency: " << val << "(hz)" << std::endl;
        m_freq = val;
    }
    else
    {
        std::cout << "[HackRF_radio] Failed to configure Tx Frequency" << std::endl;
    }
}

unsigned long HackRF_radio::get_tx_freq() const
{
    return m_freq;
}

void HackRF_radio::set_baud_rate(unsigned long val)
{
    std::cout << "[HackRF_radio] Baud Rate: " << val << "(bits per second)" << std::endl;
    m_bps = val;
}

unsigned long HackRF_radio::get_baud_rate() const
{
    return m_bps;
}



void HackRF_radio::set_baseband_gain(unsigned int val)
{
    /* range 0-62 step 2db */
    if(hackrf_set_vga_gain(p_device, val&0xFFFE) == HACKRF_SUCCESS)
    {
        std::cout << "[HackRF_radio] Baseband Gain: " << val << "(db)" << std::endl;
        m_lna_gain = val;
    }
    else
    {
        std::cout << "[HackRF_radio] Failed to configure Baseband Gain" << std::endl;
    }
}

void HackRF_radio::set_lna_gain(unsigned int val)
{
    /* range 0-40 step 8db */
    if(hackrf_set_lna_gain(p_device, val & 0xFFF8) == HACKRF_SUCCESS)
    {
        std::cout << "[HackRF_radio] LNA Gain: " << val << "(db)" << std::endl;
        m_lna_gain = val;
    }
    else
    {
        std::cout << "[HackRF_radio] Failed to configure LNA Gain" << std::endl;
    }
}



void HackRF_radio::set_txvga_gain(unsigned int val)
{
    /* range 0-47 step 1db */
    if(hackrf_set_txvga_gain(p_device, val) == HACKRF_SUCCESS)
    {
        std::cout << "[HackRF_radio] Tx VGA Gain: " << val << "(db)" << std::endl;
        m_lna_gain = val;
    }
    else
    {
        std::cout << "[HackRF_radio] Failed to configure Tx VGA Gain" << std::endl;
    }
}


unsigned int HackRF_radio::get_txvga_gain() const
{
    return m_lna_gain;
}

void HackRF_radio::set_rf_gain(bool val)
{
    /* F - 0db, T - 14db   */
    if(hackrf_set_amp_enable(p_device, val) == HACKRF_SUCCESS)
    {
        std::cout << "[HackRF_radio] RF Gain: " << (val ? "true" : "false") << "(true - +14db, false - 0db)" << std::endl;
        m_rf_gain = val;
    }
    else
    {
        std::cout << "[HackRF_radio] Failed to configure RF Gain" << std::endl;
    }

}

void HackRF_radio::set_bandwidth(double sample_rate)
{
    double bandwidth = sample_rate * 0.75; /* select narrower filters to prevent aliasing */

    /* compute best default value depending on sample rate (auto filter) */
   hackrf_set_baseband_filter_bandwidth( p_device, hackrf_compute_baseband_filter_bw( uint32_t(bandwidth) ) );
}

bool HackRF_radio::get_rf_gain() const
{
    return m_rf_gain;
}

void HackRF_radio::set_data_message(string msg, int data_type)
{
    std::cout << "[HackRF_radio] Data Message: " + msg << std::endl;
    m_msg = msg;
    m_data_type = data_type;
}

bool HackRF_radio::transmit_enabled(bool enable)
{
    if(hackrf_found())
    {
        if(enable)
        {
            std::cout << "Start Transmitting" << std::endl;
            configure_data(); // build message string
            int returnValue = hackrf_start_tx(p_device, HackRF_radio::tx_callback, this);
            std::cout << "[HackRF_radio] Transmit enable: " << ((returnValue==0) ? "Success" : "Failure") << std::endl;
            m_radio_active = true;
            return true;
        }
        else
        {
            std::cout << "Stop Transmitting" << std::endl;
            hackrf_stop_tx(p_device);
            while(hackrf_is_streaming()) {};
            m_radio_active = false;
            return false;
        }
    }
    else
    {
        m_radio_active = false;
        return false;
    }
}

bool HackRF_radio::receiver_enabled(bool enable)
{
    if(hackrf_found())
    {
        if(enable)
        {
            int returnValue = hackrf_start_rx(p_device, HackRF_radio::rx_callback, this);
            std::cout << "[HackRF_radio] Transmit enable: " << ((returnValue==0) ? "Success" : "Failure") << std::endl;
            m_radio_active = true;
            return true;
        }
        else
        {
            hackrf_stop_rx(p_device);
            m_radio_active = false;
            return false;
        }
    }
    else
    {
        m_radio_active = false;
        return false;
    }
}


void HackRF_radio::configure_data()
{
    uint8_t* input_stream = nullptr;
    size_t length = 0;
    size_t samples_per_bit = static_cast<size_t>(floor(10000000/m_bps));
    string message;

    m_tx_data_msg_lock.lock();

    if(m_mode == ONES)
    {
        input_stream = nullptr;
        length = 10;
        m_total_bits = length*samples_per_bit;
    }
    else if(m_mode == ZEROS)
    {
        input_stream = nullptr;
        length = 10;
        m_total_bits = length*samples_per_bit;
    }
    else if(m_mode == TONE10000HZ || m_mode == TONE20000HZ || m_mode == TONE50000HZ)
    {
        m_total_bits = 10000000;
    }
    else if(m_mode == BPSK || m_mode == QPSK || m_mode == AM || m_mode == FM)
    {
        if(m_data_type == 0) // user data
        {
            // 10,000,000 samples per second transmit
            // bps
            length = m_framer.serialize((uint8_t*)m_msg.c_str(), m_msg.length(), m_block);
        }
        else if(m_data_type == 1) // Incrementing numbers to 100 * 10 pkts per stop
        {
            for(size_t i = 0; i < 100; i++)
            {
                for(size_t j = 0; j < 10; j++)
                {
                    message.append(std::to_string(i));
                }
            }

            length = m_framer.serialize((uint8_t*)message.c_str(), message.length(), m_block);
        }
        else if(m_data_type == 2) // incrementing letters (capitals A-Z) * 10pkts per stop
        {
            for(char c = 65; c < 65+26; c++)
            {
                for(size_t j = 0; j < 10; j++)
                {
                    message.append(std::to_string(c));
                }
            }

            length = m_framer.serialize((uint8_t*)message.c_str(), message.length(), m_block);
        }


//        samples_per_bit = static_cast<size_t>(floor(10000000/m_bps));
//        m_total_bits = length*samples_per_bit;

        if(m_mode == BPSK || m_mode == QPSK)
        {
            m_total_bits*=8; // need to bump up the samples for the modulation
        }

        if(m_mode == FM) // Just a test sample case
        {
            samples_per_bit = 5000000;
            length = 2;
            m_total_bits = length*samples_per_bit; // going to upsample by 5 natural downsample of the radios should help
            input_stream = new uint8_t[length];

            input_stream[0] = 1;
            input_stream[1] = 0;

            // 10000000/500000 = 20
            //  2000000/100000 = 20

        }
        else
        {
            input_stream = new uint8_t[length];

            for(size_t i = 0; i < length; i++)
            {
                input_stream[i] = static_cast<uint8_t>(m_block->points[i].real());
            }
        }

        std::cout << "Samples Per Bit " << samples_per_bit << std::endl;
        std::cout << "Input Size " << length << std::endl;
        std::cout << "Total Num Samples " << m_total_bits << std::endl;

        std::cout << "Transmission Time " << static_cast<double>(length)/static_cast<double>(m_bps) << " (Seconds)" << std::endl;
        std::cout << "Packets per Second " << 1.0/(static_cast<double>(length)/static_cast<double>(m_bps)) << std::endl;

        std::cout << std::endl;
        std::cout << std::endl;
    }


    if(m_bit_stream_real != nullptr)
    {
        delete[] m_bit_stream_imag;
        delete[] m_bit_stream_real;
        delete[] m_bit_stream_Quad;
    }

    m_bit_stream_real = new RADIO_DATA_TYPE[m_total_bits];
    m_bit_stream_imag = new RADIO_DATA_TYPE[m_total_bits];
    m_bit_stream_Quad = new uint8_t[m_total_bits*2];

    if(m_mode == AM)
    {
        m_modulator_am.modulate(input_stream, length, m_bit_stream_real, nullptr, m_total_bits);
        // Create Array
        for(size_t i = 0, j = 0; j < m_total_bits; i+=2, j++)
        {
            m_bit_stream_Quad[i]   = static_cast<uint8_t>(m_bit_stream_real[j]*AMPLITUDE);
            m_bit_stream_Quad[i+1] = static_cast<uint8_t>(m_bit_stream_real[j]*AMPLITUDE);
        }
    }
    else if(m_mode == FM)
    {
        m_modulator_fm.modulate(input_stream, length, m_bit_stream_real, m_bit_stream_imag, m_total_bits, samples_per_bit);

        double max = 0;
        std::cout << "Total Bits " << m_total_bits << std::endl;

        ofstream fout("testFile.csv",ios::out|ios::trunc);
        for(size_t i = 0, j = 0; j < m_total_bits; i+=2, j++)
        {
            if(max < m_bit_stream_real[i]) max = m_bit_stream_real[i];
        }
        std::cout << "MAX " << max << std::endl;

        //Normalize
        for(size_t i = 0; i < m_total_bits; i+= 2)
        {
            m_bit_stream_Quad[i] = static_cast<uint8_t>(m_bit_stream_real[i]/max * AMPLITUDE);
            m_bit_stream_Quad[i+1] = m_bit_stream_Quad[i];
        }
    }
    else if(m_mode == BPSK)
    {
        RADIO_DATA_TYPE* m_bitstream_temp = new RADIO_DATA_TYPE[length*8];
        m_modulator_bpsk.modulate(input_stream, length, m_bitstream_temp, length*8);

        for(int i = 0; i < 1000; i++)
        {
            std::cout << (int)input_stream[i] << std::endl;
        }
        raw_upsample<RADIO_DATA_TYPE>(m_bitstream_temp, length*8, m_bit_stream_real, m_total_bits, m_total_bits/(length*8));
        delete[] m_bitstream_temp;
    }
    else if(m_mode == QPSK)
    {
        m_modulator_qpsk.modulate(input_stream, length, m_bit_stream_real, m_bit_stream_real, m_total_bits);
    }
    else if(m_mode == ZEROS)
    {
        // Do Nothing
    }
    else if(m_mode == ONES)
    {
        // Do Nothing
    }
    else if(m_mode == TONE50000HZ)
    {
        m_VCO_time=0;
        double max = 0;
        for(size_t i = 0; i < m_total_bits*2; i+= 2)
        {
            m_bit_stream_Quad[i] = static_cast<uint8_t>(cos(5000000.0 * (2*PI) * m_VCO_time));
            if(max < m_bit_stream_Quad[i]) max = m_bit_stream_Quad[i];
            m_VCO_time+=1.0/10000000.0;
        }
        std::cout << "MAX " << max << std::endl;
        //Normalize
        for(size_t i = 0; i < m_total_bits*2; i+= 2)
        {
            m_bit_stream_Quad[i] = static_cast<uint8_t>(m_bit_stream_Quad[i]/max * AMPLITUDE);
            m_bit_stream_Quad[i+1] = m_bit_stream_Quad[i];
        }
    }
    else if(m_mode == TONE10000HZ)
    {
        m_VCO_time=0;
        for(size_t i = 0; i < m_total_bits*2; i+= 2)
        {
            m_bit_stream_Quad[i] = static_cast<uint8_t>(cos(1000000.0 * (2*PI) * m_VCO_time)*AMPLITUDE);
            m_bit_stream_Quad[i+1] = m_bit_stream_Quad[i];
            m_VCO_time+=1.0/10000000.0;
        }
    }
    else if(m_mode == TONE20000HZ)
    {
        m_VCO_time=0;
        for(size_t i = 0; i < m_total_bits*2; i+= 2)
        {
            m_bit_stream_Quad[i] = static_cast<uint8_t>(cos(2000000.0 * (2*PI) * m_VCO_time)*AMPLITUDE);
            m_bit_stream_Quad[i+1] = m_bit_stream_Quad[i];
            m_VCO_time+=1.0/10000000.0;
        }
    }
    else
    {
        std::cout << "INVALID MODE !!!!" << std::endl;
    }

    std::cout << "[HackRF_radio] Total number of bits in message: " << m_total_bits << std::endl;
    m_broadcast_index = 0; // reset index
    if(input_stream)
    {
        delete[] input_stream;
    }

    m_tx_data_msg_lock.unlock();
}







