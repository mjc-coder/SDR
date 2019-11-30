/// @file SDR/RadioNode/src/libraries/network/UDP_Blaster.h


#ifndef RADIONODE_UDP_BLASTER_H
#define RADIONODE_UDP_BLASTER_H

#include <string>
#include <boost/asio.hpp>
#include <boost/asio/deadline_timer.hpp>

/// \brief Implementation of a UDP multicaster sender / receiver class.
class UDP_Blaster
{
public:
    /// \brief Constructor
    /// \param name     Name of this instance used for debugging.
    /// \param address  IP Address
    /// \param port_tx  Transmit Port
    /// \param port_rx  Receive Port
    UDP_Blaster(std::string name, std::string address, std::string port_tx, std::string port_rx = "0");

    /// \brief Destructor
    virtual ~UDP_Blaster();

    /// \brief broadcast a message
    /// \param buf  Buffer of byte data to send
    /// \param len  length of data in the buffer
    /// \return return number of bytes sent, <0 is an error.
    int send(const char* buf, size_t len);

    /// \brief Receive a UDP message on the given port.
    /// \param buffer Buffer of byte data to send
    /// \param length length of the data in the buffer
    /// \param timeout Timeout to receive the message
    /// \return Number of bytes received.
    int receive(uint8_t* buffer, size_t length, boost::posix_time::time_duration timeout);

private:
    /// \brief Handler to manage receiving a UDP message
    /// \param ec Error code
    /// \param length Length of the bufer that it can receive
    /// \param out_ec Output Error code
    /// \param out_length Output buffer length
    static void handle_receive(
            const boost::system::error_code& ec, std::size_t length,
            boost::system::error_code* out_ec, std::size_t* out_length);

private:
    std::string m_name; ///< Name of this instance for debug
    boost::asio::io_service m_io_service;   ///< Boost ASIO service
    boost::asio::ip::udp::socket m_socket;  ///< Boost Socket
    boost::asio::ip::udp::resolver m_resolver;  ///< Boost Resolver
    boost::asio::ip::udp::endpoint m_endpoint;  ///< Boost Endopoint
    boost::asio::ip::udp::endpoint m_listen_endpoint;   ///< Boost Endpoint
    boost::asio::deadline_timer m_deadline; /// Boost timer
};


#endif //RADIONODE_UDP_BLASTER_H
