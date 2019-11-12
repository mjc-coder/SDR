//
// Created by Micheal Cowan on 2019-08-16.
//

#ifndef RADIONODE_BPSK_H
#define RADIONODE_BPSK_H

#include <string>
#include <streams/DataOutput.h>
#include <common/Baseband_Stream.h>
#include <fstream>
#include <common/confidence_counter.h>

using namespace std;

#ifndef SIGN_FUNC
#define SIGN_FUNC(val) (( val == 0.0 ) ? 0.0 : ( (val < 0.0) ? -1.0 : 1.0))
#endif


class BPSK : public Baseband_Stream
{
public:
    static const RADIO_DATA_TYPE LUT_REAL[];
    static const RADIO_DATA_TYPE LUT_180[];
    static const RADIO_DATA_TYPE RRC_COEFF[];
public:
    enum PhaseAmbiguity
    {
        Ambiguity_0   = 0,
        Ambiguity_180 = 1
    };

public:
    BPSK(SafeBufferPoolQueue* BBP_Buffer_Pool,
         std::string name,
         std::string address,
         std::string m_td_port,
         std::string m_fd_port,
         size_t array_size,
         uint64_t UniqueWord = 0x4E,
         size_t UniqueWordLength = 8);

    ~BPSK();

    void modulate(uint8_t* data_in, size_t length, RADIO_DATA_TYPE* modulated_real, size_t length_modulated);

    void demodulate(BBP_Block* input, BBP_Block* output);


    /// Encode Functions
    void encode_LUT(uint8_t* data_in, size_t length);
    void encode_pulseShape(RADIO_DATA_TYPE* output_data, size_t output_length);


    /// Decode Functions
    size_t decode_pulseShape(BBP_Block* input, RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag);
    void symbol_timing_synchronization(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, size_t num_of_points);
    void phase_shift_correction(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, size_t num_of_points);
    void phase_ambiguity_correction(RADIO_DATA_TYPE* real, RADIO_DATA_TYPE* imag, size_t num_of_points);

    // others
    void complex_multiply(RADIO_DATA_TYPE ra, RADIO_DATA_TYPE ia, RADIO_DATA_TYPE rb, RADIO_DATA_TYPE ib, RADIO_DATA_TYPE& rc, RADIO_DATA_TYPE& ic);


    /// Symbol Timing Synchronization
    struct SymbolTimingSynchronizationVars
    {
        RADIO_DATA_TYPE mu;
        RADIO_DATA_TYPE I;
        RADIO_DATA_TYPE Q;
        RADIO_DATA_TYPE muDelay;

        SymbolTimingSynchronizationVars()
        : mu(0)
        , I(0)
        , Q(0)
        , muDelay(0)
        {
        }
    } SymbolTimingSynchronizationVariables;


    struct InterpData
    {
        RADIO_DATA_TYPE delay1;
        RADIO_DATA_TYPE delay2;
        RADIO_DATA_TYPE delay3;
        RADIO_DATA_TYPE delay4;
        RADIO_DATA_TYPE delay5;

        InterpData()
        : delay1(0)
        , delay2(0)
        , delay3(0)
        , delay4(0)
        , delay5(0)
        {
        }
    };
    InterpData Interp_I_Channel;
    InterpData Interp_Q_Channel;
    RADIO_DATA_TYPE InterpolationFilter(InterpData& data, RADIO_DATA_TYPE in, RADIO_DATA_TYPE mu);

    struct GardnerDetectorData
    {
        RADIO_DATA_TYPE delay1;
        RADIO_DATA_TYPE delay2;
        RADIO_DATA_TYPE delay3;
        RADIO_DATA_TYPE delay4;

        GardnerDetectorData()
        : delay1(1)
        , delay2(1)
        , delay3(1)
        , delay4(1)
        {
        }
    } GardnerDetectorVariables;

    RADIO_DATA_TYPE GardnerDetector(RADIO_DATA_TYPE i, RADIO_DATA_TYPE Q, RADIO_DATA_TYPE strobe);

    struct LoopFilterVars
    {
        RADIO_DATA_TYPE delay_x;
        RADIO_DATA_TYPE delay_y;

        LoopFilterVars()
        : delay_x(0)
        , delay_y(0)
        {
        }
    } LoopFilterVariables;

    RADIO_DATA_TYPE LoopFilter(RADIO_DATA_TYPE val);

    struct NcoControlVars
    {
        RADIO_DATA_TYPE delay;

        NcoControlVars()
        : delay(0)
        {
        }
    } NcoControlVariables;

    RADIO_DATA_TYPE NcoControl(RADIO_DATA_TYPE input);

    RADIO_DATA_TYPE StrobeEnabledRegister(RADIO_DATA_TYPE input, RADIO_DATA_TYPE strobe, RADIO_DATA_TYPE& reg);
    RADIO_DATA_TYPE Register_mu;
    RADIO_DATA_TYPE Register_I;
    RADIO_DATA_TYPE Register_Q;


/// Phase Correction
    struct PhaseCorrectionVars
    {
        RADIO_DATA_TYPE dds_real;
        RADIO_DATA_TYPE dds_imag;
        RADIO_DATA_TYPE dds_delay;

        PhaseCorrectionVars()
        : dds_real(1)
        , dds_imag(0)
        , dds_delay(0)
        {
        }
    } PhaseCorrectionVariables;


    struct PhaseLoopFilterVars
    {
        RADIO_DATA_TYPE delay_x;
        RADIO_DATA_TYPE delay_y;

        PhaseLoopFilterVars()
        : delay_x(0)
        , delay_y(0)
        {
        }
    } PhaseLoopFilterVariables;
    RADIO_DATA_TYPE PhaseLoopFilter(RADIO_DATA_TYPE val);

    RADIO_DATA_TYPE PhaseShiftCorrection_OutputLookup(RADIO_DATA_TYPE real, RADIO_DATA_TYPE imag);

    inline RADIO_DATA_TYPE constrainAngle(RADIO_DATA_TYPE x)
    {
        x = fmod(x,2.0*PI);
        if (x < 0.0)
            x += 2.0*PI;
        return x;
    }

    /// Other variables
    DataOutput m_dataoutput;
    RADIO_DATA_TYPE* m_RealData;
    size_t m_length;
    RADIO_DATA_TYPE normalization_max;
    ofstream fout;
    PhaseAmbiguity m_phase_ambiguity_shift;
    confidence_counter m_phase_ambiguity_cc;
    RADIO_DATA_TYPE m_differential_decode_delay;
    size_t UW_SIZE;
    RADIO_DATA_TYPE uw_0[64];
    RADIO_DATA_TYPE uw_180[64];
};


#endif //RADIONODE_BPSK_H
