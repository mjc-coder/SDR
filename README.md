# SDR
Software Defined Radio in C++


Build Instructions: RadioNode
To build and compile the RadioNode libraries and unit tests, execute the command "make runRadioNode" in the SDR directory.



Build Instructions: HackRF Transmitter
To compile the HackRF_Transmitter application, QT 5.13 or better is require.  
	1) load the project file "SDR/HackRF_Transmitter/HackRF_Transmitter.pro"
	2) modications, may be required in this .pro file to link to the correct hackrf library.  This code has been tested on Ubuntu 18 and MacOSX Mavericks, but on each system the HackRF libraries were installed to a unique location.  As such, the include directories for the headers, and the .dylib or .so libraries will need to be adjusted for the given system.
	3) Ensure that you have a HackRF connected to the sytem, and run the build command in QT.  
	
If there is no radio connected when the program loads,  you may have to restart the application as radios are only scanned for when the program starts.



Build Instructions: RtlSdr Receiver
To compile the RtlSdr Receiver application, QT 5.13 or better is require.  
1) load the project file "SDR/RtlSdr_Receiver/RtlSdr_Receiver.pro"
2) modications, may be required in this .pro file to link to the correct hackrf library.  This code has been tested on Ubuntu 18 and MacOSX Mavericks, but on each system the HackRF libraries were installed to a unique location.  As such, the include directories for the headers, and the .dylib or .so libraries will need to be adjusted for the given system.
3) Ensure that you have a rtlsdr receiver connected to the sytem, and run the build command in QT.  

If there is no radio connected when the program loads,  you may have to restart the application as radios are only scanned for when the program starts.  

The TegraStatsD service should be installed on the system if it is the platform is one of the Nvidia Jetsons or Nanos.  That application takes care of autostarting the tegrastats application and directing it to a fifo that the Receiver can then read and process.

To get the power and temperature statistics to show on the GUI, and arduino must be connected with the appropriate hardware and flashed with the Arduino binary code.  The system controller has to be setup an appropriate IP address to receive the UDP messages that the arduino will be broadcasting.



