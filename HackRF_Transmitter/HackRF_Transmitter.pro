QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
CONFIG+=sdk_no_version_check
CONFIG += c++11
ICON = wireless.icns

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH +=  ../RadioNode/src/libraries \
                ../RadioNode/third-party/ \
                /opt/local/include/libhackrf/ \
                /opt/local/include/libusb-1.0/ \
                /usr/include/libhackrf/ \
                /usr/include/libusb-1.0/ \

SOURCES += \
    ../RadioNode/src/libraries/DigitalSignalProcessing/DC_Filter.cpp \
    ../RadioNode/src/libraries/DigitalSignalProcessing/FFT.cpp \
    ../RadioNode/src/libraries/DigitalSignalProcessing/FirstOrderDigitalIntegration.cpp \
    ../RadioNode/src/libraries/DigitalSignalProcessing/LowPassFilter.cpp \
    ../RadioNode/src/libraries/DigitalSignalProcessing/Normalize.cpp \
    ../RadioNode/src/libraries/DigitalSignalProcessing/NotchFilter.cpp \
    ../RadioNode/src/libraries/DigitalSignalProcessing/Resample.cpp \
    ../RadioNode/src/libraries/DigitalSignalProcessing/fifth_order_filter.cpp \
    ../RadioNode/src/libraries/common/BBP_Block.cpp \
    ../RadioNode/src/libraries/common/Baseband_Stream.cpp \
    ../RadioNode/src/libraries/common/SafeBufferPoolQueue.cpp \
    ../RadioNode/src/libraries/common/confidence_counter.cpp \
    ../RadioNode/src/libraries/crc/crc8.cpp \
    ../RadioNode/src/libraries/hackrf/HackRF_radio.cpp \
    ../RadioNode/src/libraries/streams/BPSK.cpp \
    ../RadioNode/src/libraries/streams/QPSK.cpp \
    ../RadioNode/third-party/FIR-filter-class/filt.cpp \
    ../RadioNode/third-party/hackrf/host/hackrf-tools/getopt/getopt.c \
    ../RadioNode/third-party/hackrf/host/libhackrf/src/hackrf.c \
    main.cpp \
    hackrf_transmitter.cpp

HEADERS += \
    ../RadioNode/src/libraries/DigitalSignalProcessing/DC_Filter.h \
    ../RadioNode/src/libraries/DigitalSignalProcessing/FFT.h \
    ../RadioNode/src/libraries/DigitalSignalProcessing/FirstOrderDigitalIntegration.h \
    ../RadioNode/src/libraries/DigitalSignalProcessing/LowPassFilter.h \
    ../RadioNode/src/libraries/DigitalSignalProcessing/Normalize.h \
    ../RadioNode/src/libraries/DigitalSignalProcessing/NotchFilter.h \
    ../RadioNode/src/libraries/DigitalSignalProcessing/Resample.h \
    ../RadioNode/src/libraries/DigitalSignalProcessing/fifth_order_filter.h \
    ../RadioNode/src/libraries/common/BBP_Block.h \
    ../RadioNode/src/libraries/common/Baseband_Stream.h \
    ../RadioNode/src/libraries/common/Common_Deffinitions.h \
    ../RadioNode/src/libraries/common/Messages.h \
    ../RadioNode/src/libraries/common/SafeBufferPoolQueue.h \
    ../RadioNode/src/libraries/common/confidence_counter.h \
    ../RadioNode/src/libraries/crc/crc8.h \
    ../RadioNode/src/libraries/hackrf/HackRF_radio.h \
    ../RadioNode/src/libraries/network/PacketFramer.h \
    ../RadioNode/src/libraries/streams/AM.h \
    ../RadioNode/src/libraries/streams/BPSK.h \
    ../RadioNode/src/libraries/streams/FM.h \
    ../RadioNode/src/libraries/streams/QPSK.h \
    ../RadioNode/third-party/FIR-filter-class/filt.h \
    ../RadioNode/third-party/hackrf/host/hackrf-tools/getopt/getopt.h \
    ../RadioNode/third-party/hackrf/host/libhackrf/src/hackrf.h \
    hackrf_transmitter.h \
    qtlight.h

FORMS += \
    hackrf_transmitter.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


unix: LIBS += -L/lib/aarch64-linux-gnu/ -lusb-1.0
#unix: LIBS += -L/usr/local/lib -lusb-1.0


RESOURCES += \
    HackRf.qrc

DISTFILES +=
