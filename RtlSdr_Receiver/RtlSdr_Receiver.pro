#-------------------------------------------------
#
# Project created by QtCreator
#
#-------------------------------------------------
QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets network printsupport

TARGET = RtlSdrDebugger
TEMPLATE = app
CONFIG += c++14 sdk_no_version_check
ICON = Boombox.icns


CONFIG += MACOSX   # Remove this line to enable UBUNTU compiling

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS


MACOSX {
    INCLUDEPATH +=  ../RadioNode/src/libraries \
                    ../RadioNode/third-party/ \
                    ../RadioNode/third-party/hackrf/host/libhackrf/src/ \
                    /usr/include/libusb-1.0/ \
                    /usr/include/libhackrf/ \
                    /opt/local/include/libhackrf/ \
                    /opt/local/include/libusb-1.0/ \
                    /usr/include/libhackrf/ \
                    /usr/include/libusb-1.0/
} #else {
#    INCLUDEPATH +=  /home/nano/Desktop/CLionProjects/RadioNode/src/libraries \
#                    /home/nano/Desktop/CLionProjects/RadioNode/third-party/ \
#                    /home/nano/Desktop/CLionProjects/RadioNode/third-party/hackrf/host/libhackrf/src/ \
#                    /usr/include/libusb-1.0/ \
#                    /usr/include/libhackrf/ \
#                    /opt/local/include/libhackrf/ \
#                    /opt/local/include/libusb-1.0/ \
#                    /usr/include/libhackrf/ \
#                    /usr/include/libusb-1.0/
#}


SOURCES += \
    ../RadioNode/src/libraries/streams/BPSK.cpp \
    ../RadioNode/src/libraries/streams/QPSK.cpp \
    ../RadioNode/third-party/FIR-filter-class/filt.cpp \
    cpu_usage.cpp \
    qcustomplot.cpp \
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
    ../RadioNode/src/libraries/rtl_sdr/rtlsdr_radio.cpp \
    ../RadioNode/third-party/RTLSDR/convenience/convenience.c \
    ../RadioNode/third-party/RTLSDR/getopt/getopt.c \
    ../RadioNode/third-party/RTLSDR/librtlsdr.c \
    ../RadioNode/third-party/RTLSDR/tuner_e4k.c \
    ../RadioNode/third-party/RTLSDR/tuner_fc0012.c \
    ../RadioNode/third-party/RTLSDR/tuner_fc0013.c \
    ../RadioNode/third-party/RTLSDR/tuner_fc2580.c \
    ../RadioNode/third-party/RTLSDR/tuner_r82xx.c \
    main.cpp \
    rtlsdr_receiver.cpp \
    ../RadioNode/src/cuda/am.cu \
    ../RadioNode/src/cuda/ones_zeros.cu \
    system_usage.cpp

SOURCES -= ../RadioNode/src/cuda/am.cu \
           ../RadioNode/src/cuda/ones_zeros.cu

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
    ../RadioNode/src/libraries/common/RingBuffer.h \
    ../RadioNode/src/libraries/common/SafeBufferPoolQueue.h \
    ../RadioNode/src/libraries/common/confidence_counter.h \
    ../RadioNode/src/libraries/crc/crc8.h \
    ../RadioNode/src/libraries/network/PacketFramer.h \
    ../RadioNode/src/libraries/rtl_sdr/rtlsdr_radio.h \
    ../RadioNode/src/libraries/streams/AM.h \
    ../RadioNode/src/libraries/streams/BPSK.h \
    ../RadioNode/src/libraries/streams/FM.h \
    ../RadioNode/src/libraries/streams/QPSK.h \
    ../RadioNode/third-party/FIR-filter-class/filt.h \
    ../RadioNode/third-party/RTLSDR/convenience/convenience.h \
    ../RadioNode/third-party/RTLSDR/getopt/getopt.h \
    ../RadioNode/third-party/RTLSDR/include/reg_field.h \
    ../RadioNode/third-party/RTLSDR/include/rtl-sdr.h \
    ../RadioNode/third-party/RTLSDR/include/rtl-sdr_export.h \
    ../RadioNode/third-party/RTLSDR/include/rtlsdr_i2c.h \
    ../RadioNode/third-party/RTLSDR/include/tuner_e4k.h \
    ../RadioNode/third-party/RTLSDR/include/tuner_fc0012.h \
    ../RadioNode/third-party/RTLSDR/include/tuner_fc0013.h \
    ../RadioNode/third-party/RTLSDR/include/tuner_fc2580.h \
    ../RadioNode/third-party/RTLSDR/include/tuner_r82xx.h \
    cpu_usage.h \
    qcustomplot.h \
    qtlight.h \
    rtlsdr_receiver.h \
    system_usage.h

FORMS += \
    rtlsdr_receiver.ui




MACOSX {
    unix: LIBS += -L/usr/local/lib -lusb-1.0

} else {
    DEFINES += "GPU_ENABLED=\"true\""

    # Ubuntu
    unix: LIBS += -L/usr/lib/x86_64-linux-gnu/ -lusb-1.0

    # Default rules for deployment.
    qnx: target.path = /tmp/$${TARGET}/bin
    else: unix:!android: target.path = /opt/$${TARGET}/bin
    !isEmpty(target.path): INSTALLS += target

    RESOURCES += \
        RtlSdr_Receiver.qrc
    DESTDIR     = $$system(pwd)
    OBJECTS_DIR = $$DESTDIR/Obj
    QMAKE_CXXFLAGS_RELEASE =-O3
    CUDA_SOURCES += ../RadioNode/src/cuda/am.cu ../RadioNode/src/cuda/ones_zeros.cu
    CUDA_DIR      = /usr/local/cuda
    INCLUDEPATH  += $$CUDA_DIR/include /usr/include/aarch64-linux-gnu/qt5/ /usr/include/aarch64-linux-gnu/qt5/QtCore/
    QMAKE_LIBDIR += $$CUDA_DIR/lib64
    LIBS += -lcudart -lcuda
    CUDA_ARCH     = sm_32
    NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
    CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
    cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS       \
                    $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}          \
                    2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
    cuda.dependency_type = TYPE_C
    cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
    cuda.input = CUDA_SOURCES
    cuda.output = ${QMAKE_FILE_BASE}.o
    QMAKE_EXTRA_COMPILERS += cuda

    DISTFILES += \
        ../RadioNode/src/cuda/ones_zeros.cu \
        ../RadioNode/src/cuda/am.cu
}

DISTFILES += \
    ../RadioNode/src/cuda/ones_zeros.cu
    ../RadioNode/src/cuda/am_copy.cu




