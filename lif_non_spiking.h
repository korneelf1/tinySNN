#pragma once

typedef struct LIF_non_spiking {
    // int size = 7;
    // float beta = 0.8798;
    int size;
    float beta;
    float *states;
} LIF_non_spiking;

LIF_non_spiking build_lif_non_spiking(int const size);

void destroy(LIF_non_spiking* lif);
void reset_states(LIF_non_spiking* lif);

float* update(LIF_non_spiking* lif, float* input);

typedef struct LIF_non_spiking_conf{
    int const size;
    float const beta;
} LIF_non_spiking_conf;

void load_lif_non_spiking_from_conf(LIF_non_spiking *lif, LIF_non_spiking_conf const *conf);
// LIFNONSPIKING_H
