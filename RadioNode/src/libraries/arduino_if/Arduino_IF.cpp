/// @file SDR/RadioNode/src/libraries/arduino_if/Arduino_IF.h

#include <arduino_if/Arduino_IF.h>

Arduino_IF::Arduino_IF(const char* serial_if)
: serial_port(0)
, m_future_obj(m_exit_signal.get_future())
, process_thread(std::thread([this]()
 {
     while(this->m_future_obj.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout)
     {
         if(serial_port > 0)
         {
            m_rw_lock.lock();
            std::cout << "Reading Data " << std::endl;


            // read the rest of the data
             if(::read( serial_port, &m_tmp_pkt, 14 ) == 14)  // 14 bytes in the packet
             {
                if(m_tmp_pkt.HEADER == HEADER)
                {
                    ::memcpy(&m_pkt, &m_tmp_pkt, 16);
                    std::cout << "Got a packet " << m_pkt.sensorValue_A0
                                         << "  " << m_pkt.sensorValue_A1
                                         << "  " << m_pkt.sensorValue_A2
                                         << "  " << m_pkt.sensorValue_A3
                                         << "  " << m_pkt.sensorValue_A4
                                         << "  " << m_pkt.sensorValue_A5
                                         << std::endl;
                }
                else
                {
                    // flush data -- in case we are out of sync
                    char ch;
                    while ( ::read( serial_port, &ch, 1 ) > 0) {}
                }
             }


            m_rw_lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
         }
         else
         {
             std::this_thread::sleep_for(std::chrono::milliseconds(250));
         }
     }
 }))
{
    // open the serial
    serial_port = ::open(serial_if, O_RDWR | O_NOCTTY | O_NDELAY);

    if(serial_port == -1)
    {
        fprintf(stderr, "Error opening device: %s (ERRNO: %s)\n", serial_if, strerror(errno));
        return;
    }
    else
    {
        if(tcgetattr(serial_port, &m_options) < 0)
        {
            fprintf(stderr, "Unable to get attributes Error opening device: %s (ERRNO: %s)\n", serial_if, strerror(errno));
        }
        else
        {
            fprintf(stderr, "Configuring Serial Device %s ... \n", serial_if);

            fcntl(serial_port, F_SETFL, FNDELAY);                        // Open the device in nonblocking mode

            // Set parameters
            tcgetattr(serial_port, &m_options);                          // Get the current options of the port
            m_options.c_cc[VTIME]=0;                              // Timer unused
            m_options.c_cc[VMIN]=0;                               // At least on character before satisfy reading
            m_options.c_iflag |= ( IGNPAR | IGNBRK );             // Input Flags
            m_options.c_oflag=0;
            m_options.c_lflag=0;
            m_options.c_cflag |= B9600;
            m_options.c_cflag |= CS8;
            m_options.c_cflag &= ~CSTOPB;
            m_options.c_cflag &= ~(PARENB | PARODD); // NO PARITY


            if (tcsetattr(serial_port, TCSANOW, &m_options) < 0)
            {
                fprintf(stderr, "ERROR Configuring Device\n");
            }
            else
            {
                fprintf(stdout, "Device Opened\n");
            }
        }
    }

}

Arduino_IF::~Arduino_IF()
{
    // stop read thread
    m_exit_signal.set_value();
    process_thread.join();

    // close port
    if(serial_port > 0)
    {
        close(serial_port);
    }
}

uint16_t Arduino_IF::Analog_0()
{
    std::lock_guard<std::mutex> lock(m_rw_lock);
    return m_pkt.sensorValue_A0;
}

uint16_t Arduino_IF::Analog_1()
{
    std::lock_guard<std::mutex> lock(m_rw_lock);
    return m_pkt.sensorValue_A1;
}

uint16_t Arduino_IF::Analog_2()
{
    std::lock_guard<std::mutex> lock(m_rw_lock);
    return m_pkt.sensorValue_A2;
}

uint16_t Arduino_IF::Analog_3()
{
    std::lock_guard<std::mutex> lock(m_rw_lock);
    return m_pkt.sensorValue_A3;
}

uint16_t Arduino_IF::Analog_4()
{
    std::lock_guard<std::mutex> lock(m_rw_lock);
    return m_pkt.sensorValue_A4;
}

uint16_t Arduino_IF::Analog_5()
{
    std::lock_guard<std::mutex> lock(m_rw_lock);
    return m_pkt.sensorValue_A5;
}



