#pragma once

typedef struct lif {
    int in_size;
    int out_size;
    float *beta;
    float *states;
    float *thresholds;

    // it is a fused operator with next layer for speedup
    float *bias;
    float *output;
    float *weights;
} lif;

typedef struct lif_conf{
    int const in_size;
    int const out_size;
    float const *beta;
    float const *thresholds;
    float const *bias;
    float const *weights;
} lif_conf;

lif build_lif(int const in_size, int const out_size);

void destroy_lif(lif* neuron);

void reset_lif(lif* neuron);

float* update_lif(lif* neuron, float* input);

void load_lif_from_conf(lif *neuron, lif_conf const *conf);
