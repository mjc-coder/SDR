//
// Created by Micheal Cowan on 9/7/19.
//

#ifndef RADIONODE_NORMALIZE_H
#define RADIONODE_NORMALIZE_H

#include <common/Common_Deffinitions.h>
#include <common/BBP_Block.h>

RADIO_DATA_TYPE normalize(RADIO_DATA_TYPE* block, size_t len, RADIO_DATA_TYPE prev_max = 0);

RADIO_DATA_TYPE normalize(BBP_Block* block, RADIO_DATA_TYPE prev_max = 0);


#endif //RADIONODE_NORMALIZE_H
