//
// Created by Micheal Cowan on 10/25/19.
//

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

class Arduino_IF
{
public:
    Arduino_IF(const char* serial_if);

    ~Arduino_IF();

    uint16_t Analog_0();
    uint16_t Analog_1();
    uint16_t Analog_2();
    uint16_t Analog_3();
    uint16_t Analog_4();
    uint16_t Analog_5();

private:
    struct Packet
    {
        uint16_t HEADER = 0xDEAD;
        uint16_t sensorValue_A0 = 0;
        uint16_t sensorValue_A1 = 0;
        uint16_t sensorValue_A2 = 0;
        uint16_t sensorValue_A3 = 0;
        uint16_t sensorValue_A4 = 0;
        uint16_t sensorValue_A5 = 0;
    };


private:
    int serial_port;
    static constexpr uint16_t HEADER = 0xDEAD;
    Packet m_pkt;
    Packet m_tmp_pkt;

    // Process Thread
    std::promise<void> m_exit_signal;
    std::future<void> m_future_obj;
    std::thread process_thread;
    std::mutex m_rw_lock;
    struct termios m_options;
};


#endif //RADIONODE_ARDUINO_IF_H
