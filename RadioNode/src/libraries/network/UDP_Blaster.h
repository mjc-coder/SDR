//
// Created by Micheal Cowan on 2019-07-14.
//

#ifndef RADIONODE_UDP_BLASTER_H
#define RADIONODE_UDP_BLASTER_H

#include <string>
#include <boost/asio.hpp>
#include <boost/asio/deadline_timer.hpp>

class UDP_Blaster
{
public:
    UDP_Blaster(std::string name, std::string address, std::string port_tx, std::string port_rx = "0");

    virtual ~UDP_Blaster();

    int send(const char* buf, size_t len);

    int receive(uint8_t* buffer, size_t length, boost::posix_time::time_duration timeout);

private:
    static void handle_receive(
            const boost::system::error_code& ec, std::size_t length,
            boost::system::error_code* out_ec, std::size_t* out_length);
private:
    std::string m_name;
    boost::asio::io_service m_io_service;
    boost::asio::ip::udp::socket m_socket;
    boost::asio::ip::udp::resolver m_resolver;  // UDP constructor
    boost::asio::ip::udp::endpoint m_endpoint;
    boost::asio::ip::udp::endpoint m_listen_endpoint;
    boost::asio::deadline_timer m_deadline;
};


#endif //RADIONODE_UDP_BLASTER_H
