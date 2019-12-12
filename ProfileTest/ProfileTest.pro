QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


INCLUDEPATH +=  /home/jetson/dev/SDR/RadioNode/src/libraries  /home/jetson/dev/SDR/RadioNode/third-party

SOURCES += \
    main.cpp \
    ../RadioNode/src/cuda/am.cu \
    ../RadioNode/src/cuda/fm.cu \
    ../RadioNode/src/cuda/ones_zeros.cu

SOURCES -= ../RadioNode/src/cuda/am.cu \
           ../RadioNode/src/cuda/fm.cu \
           ../RadioNode/src/cuda/ones_zeros.cu

HEADERS += \
    ../RadioNode/src/libraries/streams/AM.h \
    ../RadioNode/src/libraries/streams/FM.h \


#CONFIG += MACOSX   # Remove this line to enable UBUNTU compiling

MACOSX {
    unix: LIBS += -L/usr/local/lib -lusb-1.0

} else {
    DEFINES += "GPU_ENABLED=\"true\""

    # Ubuntu
    unix: LIBS += -L/usr/lib/x86_64-linux-gnu/

    # Default rules for deployment.
    qnx: target.path = /tmp/$${TARGET}/bin
    else: unix:!android: target.path = /opt/$${TARGET}/bin
    !isEmpty(target.path): INSTALLS += target

    DESTDIR     = $$system(pwd)
    OBJECTS_DIR = $$DESTDIR/Obj
    QMAKE_CXXFLAGS_RELEASE =-O3
    CUDA_SOURCES += ../RadioNode/src/cuda/am.cu ../RadioNode/src/cuda/ones_zeros.cu ../RadioNode/src/cuda/fm.cu
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
        ../RadioNode/src/cuda/fm.cu \
        ../RadioNode/src/cuda/am.cu
}

DISTFILES += \
    ../RadioNode/src/cuda/ones_zeros.cu \
    ../RadioNode/src/cuda/fm.cu \
    ../RadioNode/src/cuda/am.cu




