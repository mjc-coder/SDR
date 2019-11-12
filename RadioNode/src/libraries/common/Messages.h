//
// Created by Micheal Cowan on 2019-06-30.
//

#ifndef RADIO_NODE_MESSAGES_H
#define RADIO_NODE_MESSAGES_H

enum HardwareType
{
    CPU,
    GPU
};

enum DecodingType
{
    FM,
    AM,
    LSB,
    USB,
    WBFM
};

enum GainMode
{
    Auto,
    Manual
};

struct Radio_msg
{
  float bps_inst;     // % BPS
  float bps_avg;      // % BPS
  float pow_inst;     // watt
  float pow_avg;      // watt
  bool  active;
};

struct Radio_Config
{
    unsigned long center_freq;
    unsigned long freq_corr_ppm;
    unsigned long sample_rate;
    bool agc_enabled;
    int gain_cdb;
    HardwareType decoder_hw;
    unsigned long samples_per_second;
}; 



#endif //RADIO_NODE_MESSAGES_H
