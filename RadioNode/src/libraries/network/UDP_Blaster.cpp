/// @file SDR/RadioNode/src/libraries/network/UDP_Blaster.cpp


#include <network/UDP_Blaster.h>
#include <iostream>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

using boost::asio::deadline_timer;
using boost::asio::ip::udp;
using boost::asio::ip::udp;

UDP_Blaster::UDP_Blaster(std::string name, std::string address, std::string port_tx, std::string port_rx)
: m_name(name)
, m_socket(m_io_service, udp::endpoint(udp::v4(), 0))
, m_resolver(m_io_service)
, m_endpoint(*m_resolver.resolve({udp::v4(), address, port_tx}))
, m_listen_endpoint(*m_resolver.resolve({udp::v4(), address, port_rx}))
, m_deadline(m_io_service)
{
    std::cout << "[" << m_name << "] Address " << address << "  Tx Port " << port_tx << "  Rx Port " << port_rx << std::endl;
}

UDP_Blaster::~UDP_Blaster()
{
    // nothing to destroy
}

int UDP_Blaster::send(const char *buf, size_t len)
{
    try
    {
        return m_socket.send_to(boost::asio::buffer(buf, len), m_endpoint);
    }
    catch(...)
    {
        std::cout << "[" << m_name << "] " << "Didnt send packet... whoops. [" << len << "]" << std::endl;
        return 0;
    }
}

int UDP_Blaster::receive(uint8_t* buffer, size_t length, boost::posix_time::time_duration timeout)
{
    boost::system::error_code ec;
    // Set a deadline for the asynchronous operation.
    m_deadline.expires_from_now(timeout);

    // Set up the variables that receive the result of the asynchronous
    // operation. The error code is set to would_block to signal that the
    // operation is incomplete. Asio guarantees that its asynchronous
    // operations will never fail with would_block, so any other value in
    // ec indicates completion.
    ec = boost::asio::error::would_block;
    std::size_t bytes_read = 0;

    // Start the asynchronous operation itself. The handle_receive function
    // used as a callback will update the ec and length variables.
    m_socket.async_receive(boost::asio::buffer(buffer, length),
                          boost::bind(&UDP_Blaster::handle_receive, _1, _2, &ec, &length));

    // Block until the asynchronous operation has completed.
    do
    {
        m_io_service.run_one();
    } while (ec == boost::asio::error::would_block);


    return length;
}

void UDP_Blaster::handle_receive(
        const boost::system::error_code& ec, std::size_t length,
        boost::system::error_code* out_ec, std::size_t* out_length)
{
    *out_ec = ec;
    *out_length = length;
}