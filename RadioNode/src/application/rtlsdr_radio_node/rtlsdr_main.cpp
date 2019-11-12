
// Status blaster is a versatile application that when configured with command line options will
// parse and broadcast multiple data points through udp to a target IP / Port combination.


#include <iostream>
#include <boost/program_options.hpp>
#include <string>
#include <common/Messages.h>
#include <boost/asio.hpp>
#include <RTLSDR/include/rtl-sdr.h>
#include <rtl_sdr/rtlsdr_radio.h>
using namespace std;
using namespace boost::program_options;
using boost::asio::ip::udp;

// Globals
std::string address("127.0.0.1");
std::string base_port  = "5000";
unsigned int config = 0;
std::string name("RadioNode");
unsigned int device = 0;
bool cancel = false;


// Signals
void signalIntHandler( int signum )
{
    cout << "Interrupt signal (" << signum << ") received.\n";

    // cleanup and close up stuff here
    // terminate program
    cancel = true;
}

void signalTerminateHandler( int signum )
{
    cout << "Terminate signal (" << signum << ") received.\n";

    // cleanup and close up stuff here
    // terminate program
    cancel = true;
}


int main(int argc, char* argv[])
{
  // Connect signals
  signal(SIGINT, signalIntHandler);
  signal(SIGTERM, signalTerminateHandler);

  try
  {
    // Boost Options
    // https://theboostcpplibraries.com/boost.program_options
    options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
      ("list,l", "List all devices")
      ("base_port",      value<std::string>(&base_port),     "Base Port")
      ("remote_ip",      value<std::string>(&address),             "Remote IP Address")
      ("device,d",       value<unsigned int>(&device),             "Device index of the RTLSDR radio")
      ("config,c",       value<unsigned int>(&config),             "HW Decoding configuration")
      ("name, n",        value<std::string>(&name),                "Descriptive Name of this RadioNode for Status");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << '\n';
        return 0;
    }
    if(vm.count("list"))
    {
        for(unsigned int i = 0; i < rtlsdr_get_device_count(); i++)
        {
            char manufact[255] = {0};
            char product[255] = {0};
            char serial[255] = {0};
            rtlsdr_get_device_usb_strings(i, manufact, product, serial);
            std::cout << "Device [" << i << "] " << rtlsdr_get_device_name(i)
                      << "\n\t\t Manufac ... " << manufact
                      << "\n\t\t Product ... " << product
                      << "\n\t\t Serial .... " << serial
                      << std::endl;
        }
        return 0;
    }
    std::cout << "\nStarting Application with following configuration ..." << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "Base Command Port ............. " << base_port << std::endl;
    std::cout << "Remote IP ..................... " << address << std::endl;
    std::cout << "Device Index .................. " << device << std::endl;
    std::cout << "HW Configuration .............. " << config << std::endl;
    std::cout << "Name .......................... " << name << std::endl;
    std::cout << std::endl;

    std::cout << "Starting Radio Node ... " << std::endl;
    Radio_RTLSDR radio(device, (config == 0) ? HardwareType::CPU : HardwareType::GPU, address, base_port);

    if(radio.rtlsdr_found())
    {
        while (!cancel)
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    else
    {
        std::cout << "Radio Did not Open correctly" << std::endl;
    }
  }
  catch (const error &ex)
  {
    std::cerr << ex.what() << '\n';
  }

  std::cout << "Terminating Radio Node [" << name << "]" << std::endl;
  return 0;
}