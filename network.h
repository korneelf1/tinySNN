#pragma once

#include "lif.h"
#include "lif_non_spiking.h"
#include "linear.h"

typedef struct SNN{
    int in_size, out_size, hidden_size;

    linear *inhid;
    lif *hidhid;
    lif *hidout;
    LIF_non_spiking *output;
 } SNN;

SNN build_snn(int const in_size, int const out_size, int const hidden_size);

void reset_net(SNN* snn);

float* forward(SNN* snn, float* input);

void destroy_net(SNN* snn);