/// @file SDR/RadioNode/src/libraries/arduino_if/Arduino_IF.h

#ifndef RADIONODE_ARDUINO_IF_H
#define RADIONODE_ARDUINO_IF_H

#include <stdint.h>
#include <functional>
#include <thread>
#include <future>
#include <mutex>
#include <iostream>
#include <unistd.h>     // UNIX standard function definitions
#include <fcntl.h>      // File control definitions
#include <errno.h>      // Error number definitions
#include <termios.h>    // POSIX terminal control definitions
#include <thread>

/// Simple Arduino interface
class Arduino_IF
{
public:
    /// Constructor
    ///
    /// \param[in] serial_if    name of the serial interface.
    Arduino_IF(const char* serial_if);

    /// Destructor
    ~Arduino_IF();

    /// Read Analog 0 value
    ///
    /// \return unsigned analog value 0-1023
    uint16_t Analog_0();

    /// Read Analog 1 value
    ///
    /// \return unsigned analog value 0-1023
    uint16_t Analog_1();

    /// Read Analog 2 value
    ///
    /// \return unsigned analog value 0-1023
    uint16_t Analog_2();

    /// Read Analog 3 value
    ///
    /// \return unsigned analog value 0-1023
    uint16_t Analog_3();

    /// Read Analog 4 value
    ///
    /// \return unsigned analog value 0-1023
    uint16_t Analog_4();

    /// Read Analog 5 value
    ///
    /// \return unsigned analog value 0-1023
    uint16_t Analog_5();

private:
    /// UDP Packet structure
    struct Packet
    {
        uint16_t HEADER = 0xDEAD;       ///< Packet Header
        uint16_t sensorValue_A0 = 0;    ///< Analog 0
        uint16_t sensorValue_A1 = 0;    ///< Analog 1
        uint16_t sensorValue_A2 = 0;    ///< Analog 2
        uint16_t sensorValue_A3 = 0;    ///< Analog 3
        uint16_t sensorValue_A4 = 0;    ///< Analog 4
        uint16_t sensorValue_A5 = 0;    ///< Analog 5
    };


private:
    int serial_port;                                ///< Serial Port File descriptor
    static constexpr uint16_t HEADER = 0xDEAD;      ///< Packet Header for validation
    Packet m_pkt;                                   ///< Last valid Packet
    Packet m_tmp_pkt;                               ///< Temp Packet when checking for Validation

    // Process Thread
    std::promise<void> m_exit_signal;               ///< Exit Signal to terminate thread
    std::future<void> m_future_obj;                 ///< Future object to terminate threads
    std::thread process_thread;                     ///< Process thread for receiving arduino packets
    std::mutex m_rw_lock;                           ///< Read/Write lock on valid packet
    struct termios m_options;                       ///< Serial terminal options
};


#endif //RADIONODE_ARDUINO_IF_H
