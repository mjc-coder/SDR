//
// Created by Micheal Cowan on 2019-08-16.
//

#include <common/Common_Deffinitions.h>
#include <streams/QPSK.h>
#include <DigitalSignalProcessing/Resample.h>
#include <FIR-filter-class/filt.h>
#include <DigitalSignalProcessing/Normalize.h>
#include <fstream>
#include <iomanip>
#include <iostream>

const RADIO_DATA_TYPE QPSK::LUT_REAL[4] = { 1, -1,  1, -1};
const RADIO_DATA_TYPE QPSK::LUT_IMAG[4] = { 1,  1, -1, -1};
const RADIO_DATA_TYPE QPSK::LUT_REAL___0[4] = {0,1,2,3};
const RADIO_DATA_TYPE QPSK::LUT_REAL__90[4] = {2,0,3,1};
const RADIO_DATA_TYPE QPSK::LUT_REAL_180[4] = {3,2,1,0};
const RADIO_DATA_TYPE QPSK::LUT_REAL_270[4] = {1,3,0,2};
const RADIO_DATA_TYPE QPSK::DEFAULT_UNIQUE_WORD[64] =
        { 0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,
          3,0,2,1,0,3,2,1,
          2,0,1,2,3,2,2,3};


// MATLAB rcosine(1/8,1,'sqrt',0.5,8)
const RADIO_DATA_TYPE QPSK::RRC_COEFF[] =    {  -0.000882663055055987,  -0.000746201590523265, -0.000336789968798916,  0.000225038893553655,
                                                 0.000757880681389978,   0.001077943504359050,  0.001062303725777150,  0.000695521249813576,
                                                 8.24465490986367e-05,  -0.000579554041955506, -0.001062303725777150, -0.001183702054816000,
                                                 -0.000874477709296126,  -0.000211083163638599,  0.000599193283619645,  0.001279536876191750,
                                                 0.001573979573701230,   0.001338143762729140,  0.000599193283619642, -0.000438236415412023,
                                                 -0.001446863119017230,  -0.002078893324143190, -0.002084671221261930, -0.001408617284422990,
                                                 -0.000227352605090172,   0.001088699270085450,  0.002084671221261940,  0.002371289907799650,
                                                 0.001768388256576610,   0.000391988582678128, -0.001351122110122730, -0.002869993360503870,
                                                 -0.003572683794274230,  -0.003072685906611650, -0.001351122110122720,  0.001186107048403510,
                                                 0.003789403406949890,   0.005575980755162950,  0.005816670050933640,  0.004213278581633850,
                                                 0.001071805138282270,  -0.002702696087965530, -0.005816670050933650, -0.006968745846090550,
                                                 -0.005305164769729840,  -0.000815536975401244,  0.005468827588591990,  0.011573823048484000,
                                                 0.015005271935951800,   0.013430574641186900,  0.005468827588591980, -0.008615615450369140,
                                                 -0.026525823848649200,  -0.044008760276746200, -0.055452254485567400, -0.054931750238658300,
                                                 -0.037513179839879400,  -0.000532386899804849,  0.055452254485567400,  0.126288206185736000,
                                                 0.204577471545948000,   0.280728910881642000,  0.344536138081295000,  0.386975375596443000,
                                                 0.401855774335913000,   0.386975375596443000,  0.344536138081295000,  0.280728910881642000,
                                                 0.204577471545948000,   0.126288206185736000,  0.055452254485567400, -0.000532386899804849,
                                                 -0.037513179839879400,  -0.054931750238658300, -0.055452254485567400, -0.044008760276746200,
                                                 -0.026525823848649200,  -0.008615615450369140,  0.005468827588591980,  0.013430574641186900,
                                                 0.015005271935951800,   0.011573823048484000,  0.005468827588591990, -0.000815536975401244,
                                                 -0.005305164769729840,  -0.006968745846090550, -0.005816670050933650, -0.002702696087965530,
                                                 0.001071805138282270,   0.004213278581633850,  0.005816670050933640,  0.005575980755162950,
                                                 0.003789403406949890,   0.001186107048403510, -0.001351122110122720, -0.003072685906611650,
                                                 -0.003572683794274230,  -0.002869993360503870, -0.001351122110122730,  0.000391988582678128,
                                                 0.001768388256576610,   0.002371289907799650,  0.002084671221261940,  0.001088699270085450,
                                                 -0.000227352605090172,  -0.001408617284422990, -0.002084671221261930, -0.002078893324143190,
                                                 -0.001446863119017230,  -0.000438236415412023,  0.000599193283619642,  0.001338143762729140,
                                                 0.001573979573701230,   0.001279536876191750,  0.000599193283619645, -0.000211083163638599,
                                                 -0.000874477709296126,  -0.001183702054816000, -0.001062303725777150, -0.000579554041955506,
                                                 8.24465490986367e-05,   0.000695521249813576,  0.001062303725777150,  0.001077943504359050,
                                                 0.000757880681389978,   0.000225038893553655, -0.000336789968798916, -0.000746201590523265,
                                                 -0.000882663055055987 };

QPSK::QPSK(SafeBufferPoolQueue* BBP_Buffer_Pool,
           std::string name,
           std::string address,
           std::string m_td_port,
           std::string m_fd_port,
           size_t array_size,
           const RADIO_DATA_TYPE* UniqueWord,
           size_t UniqueWordLength)
        : Baseband_Stream([this](BBP_Block* block)
                          {
                              decimate(block, 50);
                             // demod(block);
                          }, BBP_Buffer_Pool)
        , m_dataoutput(BBP_Buffer_Pool, name, address, m_td_port, m_fd_port, array_size)
        , Register_mu(0)
        , Register_I(0)
        , Register_Q(0)
        , m_RealData(0)
        , m_length(0)
        , normalization_max(0)
        , fout("QPSK.raw", std::ios::binary | std::ios::out | std::ios::trunc)
        , m_phase_ambiguity_cc_000(10, 5, 5)
        , m_phase_ambiguity_cc_090(10, 5, 5)
        , m_phase_ambiguity_cc_180(10, 5, 5)
        , m_phase_ambiguity_cc_270(10, 5, 5)
        , m_differential_decode_delay(1)
        , UW_SIZE(UniqueWordLength)
{
    // Initialize Structures
    memset(&SymbolTimingSynchronizationVariables, 0, sizeof(SymbolTimingSynchronizationVars));
    memset(&Interp_I_Channel, 0, sizeof(InterpData));
    memset(&Interp_Q_Channel, 0, sizeof(InterpData));
    GardnerDetectorVariables.delay1 = 1;
    GardnerDetectorVariables.delay2 = 1;
    GardnerDetectorVariables.delay3 = 1;
    GardnerDetectorVariables.delay4 = 1;
    memset(&LoopFilterVariables, 0, sizeof(LoopFilterVars));
    memset(&NcoControlVariables, 0, sizeof(NcoControlVars));
    memset(&PhaseCorrectionVariables, 0, sizeof(PhaseCorrectionVars));

    PhaseCorrectionVariables.dds_real = 1;

    // Configure Unique Words
    if(UW_SIZE > 64) return; // Size check
    memset(uw_000, 0, 64*sizeof(RADIO_DATA_TYPE));
    memset(uw_090, 0, 64*sizeof(RADIO_DATA_TYPE));
    memset(uw_180, 0, 64*sizeof(RADIO_DATA_TYPE));
    memset(uw_270, 0, 64*sizeof(RADIO_DATA_TYPE));

    for(int i = 64-1; i >= 64-UniqueWordLength; i--)
    {
        if(UniqueWord[i] >= 0 && UniqueWord[i] <= 3)
        {
            uw_000[i] = LUT_REAL___0[(int)UniqueWord[i]];
            uw_090[i] = LUT_REAL_270[(int)UniqueWord[i]];
            uw_180[i] = LUT_REAL_180[(int)UniqueWord[i]];
            uw_270[i] = LUT_REAL__90[(int)UniqueWord[i]];
        }
        else
        {
            uw_000[i] = 0;
            uw_090[i] = 0;
            uw_180[i] = 0;
            uw_270[i] = 0;
        }
    }

    this->add_next_buffer(&m_dataoutput);
}

QPSK::~QPSK()
{
    fout.close();
}

void QPSK::modulate(uint8_t* data_in, size_t length, RADIO_DATA_TYPE* modulated_real, RADIO_DATA_TYPE* modulated_imag, size_t length_modulated)
{
    m_RealData = new RADIO_DATA_TYPE[length*8];
    m_ImagData = new RADIO_DATA_TYPE[length*8];
    memset(m_RealData, 0, sizeof(RADIO_DATA_TYPE)*length*8);
    memset(m_ImagData, 0, sizeof(RADIO_DATA_TYPE)*length*8);

    // LUT convert 0,1 to 1,-1
    encode_LUT(data_in, length);

    // Pulse Shape
    encode_pulseShape(modulated_real, modulated_imag, length_modulated);

    delete[] m_RealData;
    delete[] m_ImagData;
}

void QPSK::encode_LUT(uint8_t* data_in, size_t length)
{
    m_length = length;

    for(size_t index = 0; index < m_length; index++)
    {
        m_RealData[index] = QPSK::LUT_REAL[data_in[index]];
        m_ImagData[index] = QPSK::LUT_IMAG[data_in[index]];
    }
}


void QPSK::encode_pulseShape(RADIO_DATA_TYPE* real_data, RADIO_DATA_TYPE* imag_data, size_t output_length)
{
    if(output_length != 8*m_length)
    {
        std::cout << "ERROR:: QPSK modulation output arrays are not large enough!!!" << std::endl;
        return;
    }
    // clear array
    memset(real_data, 0, output_length);
    memset(imag_data, 0, output_length);
    RADIO_DATA_TYPE* r_data = new RADIO_DATA_TYPE[(m_length*8)+(129*2)];  // Upsample 8, and include forward buffer for FIR transport
    RADIO_DATA_TYPE* i_data = new RADIO_DATA_TYPE[(m_length*8)+(129*2)];  // Upsample 8, and include forward buffer for FIR transport

    // clear memory
    memset(r_data, 0, ((m_length*8)+(129*2))*sizeof(RADIO_DATA_TYPE));
    memset(i_data, 0, ((m_length*8)+(129*2))*sizeof(RADIO_DATA_TYPE));

    // Upsample the arrays
    (void)upsample_fill_w_zeros(m_RealData, m_length, &r_data[129], m_length*8, 8);
    (void)upsample_fill_w_zeros(m_ImagData, m_length, &i_data[129], m_length*8, 8);

    for(int i = 0; i < output_length; i++)
    {
        for(int j = 0; j < 129; j++)
        {
            real_data[i] += r_data[i + 129 - j] * RRC_COEFF[129 - j - 1];
            imag_data[i] += i_data[i + 129 - j] * RRC_COEFF[129 - j - 1];
        }
    }

    delete[] r_data;
    delete[] i_data;
}

void QPSK::demodulate(BBP_Block* input, BBP_Block* output)
{
    std::cout << "demod length " << input->number_of_points() << std::endl;
    m_length = input->number_of_points()/4;
    m_RealData = new RADIO_DATA_TYPE[m_length];
    m_ImagData = new RADIO_DATA_TYPE[m_length];

    memset(m_RealData, 0, m_length*sizeof(RADIO_DATA_TYPE));
    memset(m_ImagData, 0, m_length*sizeof(RADIO_DATA_TYPE));

    // RX Correlator
    decode_pulseShape(input);

    symbol_timing_synchronization();

    phase_shift_correction();

    phase_ambiguity_correction();

    output->reset();
    // load the output block
    for(int i = 0; i < m_length; i++)
    {
        output->points[i].real(m_RealData[i]);
        output->points[i].imag(m_ImagData[i]);
        output->n_points++;
    }

    delete[] m_RealData;
    delete[] m_ImagData;
}


size_t QPSK::decode_pulseShape(BBP_Block* input)
{
    RADIO_DATA_TYPE* real_data = new RADIO_DATA_TYPE[input->number_of_points()+(129*2)];  // Upsample 8, and include forward buffer for FIR transport
    RADIO_DATA_TYPE* imag_data = new RADIO_DATA_TYPE[input->number_of_points()+(129*2)];  // Upsample 8, and include forward buffer for FIR transport
    RADIO_DATA_TYPE* real_data_interp = new RADIO_DATA_TYPE[input->number_of_points()];   // Upsample 8, and include forward buffer for FIR transport
    RADIO_DATA_TYPE* imag_data_interp = new RADIO_DATA_TYPE[input->number_of_points()];   // Upsample 8, and include forward buffer for FIR transport

    memset(real_data, 0, (input->number_of_points()+(129*2))*sizeof(RADIO_DATA_TYPE));
    memset(imag_data, 0, (input->number_of_points()+(129*2))*sizeof(RADIO_DATA_TYPE));
    memset(real_data_interp, 0, input->number_of_points()*sizeof(RADIO_DATA_TYPE));
    memset(imag_data_interp, 0, input->number_of_points()*sizeof(RADIO_DATA_TYPE));

    // Initialize data
    for(int i = 0; i < input->number_of_points(); i++)
    {
        real_data[i+129] = input->points[i].real();
        imag_data[i+129] = input->points[i].imag();
    }

    for(int i = 0; i < input->number_of_points(); i++)
    {
        for(int j = 0; j < 129; j++)
        {
            real_data_interp[i] += real_data[i+129-j] * RRC_COEFF[129-j-1];
            imag_data_interp[i] += imag_data[i+129-j] * RRC_COEFF[129-j-1];
        }
    }

    // Downsample by 4
    (void)decimate(real_data_interp, input->number_of_points(), 4);
    (void)decimate(imag_data_interp, input->number_of_points(), 4);

    memcpy(m_RealData, real_data_interp, sizeof(RADIO_DATA_TYPE)*m_length);
    memcpy(m_ImagData, imag_data_interp, sizeof(RADIO_DATA_TYPE)*m_length);

    delete[] real_data_interp;
    delete[] imag_data_interp;
    delete[] real_data;
    delete[] imag_data;
    return m_length;
}

void QPSK::symbol_timing_synchronization()
{
    RADIO_DATA_TYPE strobeTemp = 0;

    // Symbol Timing Synchronization
    for(int i = 0; i < m_length; i++)
    {
        strobeTemp = ((NcoControlVariables.delay < 0) ? 1 : 0);
        SymbolTimingSynchronizationVariables.mu = StrobeEnabledRegister(SymbolTimingSynchronizationVariables.muDelay*2, strobeTemp, Register_mu);
        SymbolTimingSynchronizationVariables.Q = InterpolationFilter(Interp_Q_Channel, m_RealData[i], SymbolTimingSynchronizationVariables.mu);
        SymbolTimingSynchronizationVariables.I = InterpolationFilter(Interp_I_Channel, m_ImagData[i], SymbolTimingSynchronizationVariables.mu);
        SymbolTimingSynchronizationVariables.muDelay = NcoControl(LoopFilter(GardnerDetector(SymbolTimingSynchronizationVariables.I, SymbolTimingSynchronizationVariables.Q, strobeTemp)));
        m_RealData[i] = StrobeEnabledRegister(SymbolTimingSynchronizationVariables.Q, strobeTemp, Register_Q);
        m_ImagData[i] = StrobeEnabledRegister(SymbolTimingSynchronizationVariables.I, strobeTemp, Register_I);
    }
}

RADIO_DATA_TYPE QPSK::InterpolationFilter(InterpData& data, RADIO_DATA_TYPE in, RADIO_DATA_TYPE mu)
{
    const RADIO_DATA_TYPE K = -0.5;

    RADIO_DATA_TYPE V1 = data.delay1 - K*in;
    RADIO_DATA_TYPE V2 = V1 + data.delay2;
    RADIO_DATA_TYPE V3 = V2 - data.delay3;
    RADIO_DATA_TYPE V4 = K*in - data.delay1 + data.delay4;
    RADIO_DATA_TYPE V5 = V4 + data.delay2;
    RADIO_DATA_TYPE V6 = data.delay3 + V5;
    RADIO_DATA_TYPE V7 = V3*mu + V6;
    RADIO_DATA_TYPE V8 = data.delay5 + mu * V7;

    // Generate output before updating the Delays
    RADIO_DATA_TYPE toReturn = V8;

    // Update Delays
    data.delay3 = data.delay2;
    data.delay2 = data.delay1;
    data.delay1 = K*in;

    data.delay5 = data.delay4;
    data.delay4 = in;

    return toReturn;
}

RADIO_DATA_TYPE QPSK::GardnerDetector(RADIO_DATA_TYPE i, RADIO_DATA_TYPE q, RADIO_DATA_TYPE strobe)
{
    RADIO_DATA_TYPE toReturn = 0;

    if(strobe >= 0.5)
    {
        toReturn = (GardnerDetectorVariables.delay3 * (-SIGN_FUNC(q)+SIGN_FUNC(GardnerDetectorVariables.delay4)) +
                    (GardnerDetectorVariables.delay1 * (-SIGN_FUNC(i) + SIGN_FUNC(GardnerDetectorVariables.delay2))));
    }

    // Configure Delayed values
    GardnerDetectorVariables.delay2 = GardnerDetectorVariables.delay1;
    GardnerDetectorVariables.delay1 = i;
    GardnerDetectorVariables.delay4 = GardnerDetectorVariables.delay3;
    GardnerDetectorVariables.delay3 = q;

    return toReturn;
}

RADIO_DATA_TYPE QPSK::LoopFilter(RADIO_DATA_TYPE val)
{
    const RADIO_DATA_TYPE K1 = -0.00587880145347487;
    const RADIO_DATA_TYPE K2 = -2.35152058138995e-05;
    RADIO_DATA_TYPE toReturn = LoopFilterVariables.delay_y + K2*val + K1*(val-LoopFilterVariables.delay_x);
    LoopFilterVariables.delay_x = val;
    LoopFilterVariables.delay_y = toReturn;
    return toReturn;
}

RADIO_DATA_TYPE QPSK::NcoControl(RADIO_DATA_TYPE input)
{
    RADIO_DATA_TYPE toReturn = NcoControlVariables.delay - floor(NcoControlVariables.delay);
    NcoControlVariables.delay = toReturn-0.5-input;
    return toReturn;
}

RADIO_DATA_TYPE QPSK::StrobeEnabledRegister(RADIO_DATA_TYPE input, RADIO_DATA_TYPE strobe, RADIO_DATA_TYPE& reg)
{
    if(strobe >= 1)
    {
        reg = input;    // Only change value if the strobe goes high
    }
    return reg;
}



void QPSK::phase_shift_correction()
{
    RADIO_DATA_TYPE temp_real = 0;
    RADIO_DATA_TYPE temp_imag = 0;
    const RADIO_DATA_TYPE constant = 8.0;

    for(size_t i = 0; i < m_length; i++)
    {
        // Multiply DDS results with Input to Correct Phase
        RADIO_DATA_TYPE temp = PhaseLoopFilter(constant * (SIGN_FUNC(temp_imag)*temp_real - SIGN_FUNC(temp_real)*temp_imag));
        PhaseCorrectionVariables.dds_real = cos(PhaseCorrectionVariables.dds_delay + temp);
        PhaseCorrectionVariables.dds_imag = sin(PhaseCorrectionVariables.dds_delay + temp);
        PhaseCorrectionVariables.dds_delay = PhaseCorrectionVariables.dds_delay    + temp;
        complex_multiply(m_RealData[i], m_ImagData[i], PhaseCorrectionVariables.dds_real, PhaseCorrectionVariables.dds_imag, temp_real, temp_imag);

        // Generate Output
        m_RealData[i] = PhaseShiftCorrection_OutputLookup(SIGN_FUNC(temp_real), SIGN_FUNC(temp_imag));
        m_ImagData[i] = 0;
    }

    m_length = decimate(m_RealData, m_length, 2); // Rx correlated data is decimated by 2
}


void QPSK::complex_multiply(RADIO_DATA_TYPE ra, RADIO_DATA_TYPE ia, RADIO_DATA_TYPE rb, RADIO_DATA_TYPE ib, RADIO_DATA_TYPE& rc, RADIO_DATA_TYPE& ic)
{
    rc = ra*rb - ia*ib;
    ic = ia*rb + ra*ib;
}


RADIO_DATA_TYPE QPSK::PhaseLoopFilter(RADIO_DATA_TYPE val)
{
    const RADIO_DATA_TYPE K1 = 0.0620001240002480;
    const RADIO_DATA_TYPE K2 = 0.000992001984003968;
    RADIO_DATA_TYPE toReturn = PhaseLoopFilterVariables.delay_y - K1*PhaseLoopFilterVariables.delay_x + (K1+K2)*val;
    PhaseLoopFilterVariables.delay_x = val;
    PhaseLoopFilterVariables.delay_y = toReturn;
    return toReturn;
}

RADIO_DATA_TYPE QPSK::PhaseShiftCorrection_OutputLookup(RADIO_DATA_TYPE real, RADIO_DATA_TYPE imag)
{
    if(real < 0 && imag < 0)
    {
        return 3;
    }
    else if(real > 0 && imag < 0)
    {
        return 2;
    }
    else if(real < 0 && imag > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void QPSK::phase_ambiguity_correction()
{
    // Check ambiguity
    for(size_t i = 0; i < m_length-UW_SIZE; i++)
    {
        size_t match_count_000 = 0;
        size_t match_count_090 = 0;
        size_t match_count_180 = 0;
        size_t match_count_270 = 0;
        // check
        for(size_t j = 0; j < UW_SIZE; j++)
        {
            if(m_RealData[i+j] == uw_000[63 - j])
            {
                match_count_000++;
            }
            else if(m_RealData[i+j] == uw_090[63 - j])
            {
                match_count_090++;
            }
            else if(m_RealData[i+j] == uw_180[63 - j])
            {
                match_count_180++;
            }
            else if(m_RealData[i+j] == uw_270[63 - j])
            {
                match_count_270++;
            }
        }

        if(match_count_000 == UW_SIZE)
        {
            std::cout << "No Phase " << i << std::endl;
            ++m_phase_ambiguity_cc_000;
            --m_phase_ambiguity_cc_090;
            --m_phase_ambiguity_cc_180;
            --m_phase_ambiguity_cc_270;
        }
        else if(match_count_090 == UW_SIZE)
        {
            std::cout << "90 Phase " << i << std::endl;
            --m_phase_ambiguity_cc_000;
            ++m_phase_ambiguity_cc_090;
            --m_phase_ambiguity_cc_180;
            --m_phase_ambiguity_cc_270;
        }
        else if(match_count_180 == UW_SIZE)
        {
            std::cout << "180 Phase " << i << std::endl;
            --m_phase_ambiguity_cc_000;
            --m_phase_ambiguity_cc_090;
            ++m_phase_ambiguity_cc_180;
            --m_phase_ambiguity_cc_270;
        }
        else if(match_count_270 == UW_SIZE)
        {
            std::cout << "270 Phase " << i << std::endl;
            --m_phase_ambiguity_cc_000;
            --m_phase_ambiguity_cc_090;
            --m_phase_ambiguity_cc_180;
            ++m_phase_ambiguity_cc_270;
        }
    }


    if(m_phase_ambiguity_cc_090.high_confidence())
    {
        std::cout << "90 Degree Phase Detected" << std::endl;
        // we have 180 Degree inversion
        for(size_t i = 0; i < m_length; i++)
        {
            m_RealData[i] = LUT_REAL__90[(int)m_RealData[i]];
        }
    }
    else if(m_phase_ambiguity_cc_180.high_confidence())
    {
        std::cout << "180 Degree Phase Detected" << std::endl;
        // we have 180 Degree inversion
        for(size_t i = 0; i < m_length; i++)
        {
            m_RealData[i] = LUT_REAL_180[(int)m_RealData[i]];
        }
    }
    else if(m_phase_ambiguity_cc_270.high_confidence())
    {
        std::cout << "270 Degree Phase Detected" << std::endl;
        // we have 270 Degree inversion
        for(size_t i = 0; i < m_length; i++)
        {
            m_RealData[i] = LUT_REAL_270[(int)m_RealData[i]];
        }
    }
    // if phase ambiguity 000 was high then nothing needs to be done
}

