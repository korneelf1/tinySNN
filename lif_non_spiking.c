#include "LIF_non_spiking.h"
#include <stdlib.h>

LIF_non_spiking build_lif_non_spiking(int const size) {
    LIF_non_spiking lif;
    lif.size = size;
    lif.beta = 0.8798;
    lif.states = calloc(size, sizeof(*lif.states));

    return lif;
};


void reset_state(LIF_non_spiking* lif) {
    // lif->size=7;
    lif->beta=0.8798;
    for (int i=0;i<lif->size;i++){
        lif->states[i] = 0.0f;
    }
};

float* update(LIF_non_spiking* lif, float* input) {
    for (int i=0;i<lif->size;i++){
        lif->states[i] = lif->states[i] * lif->beta + input[i];
    }
    return lif->states;
};

void destroy(LIF_non_spiking* lif) {
    free(lif->states);
    free(lif->size);
    // free(lif->beta); gives an error?
    free(lif);    
};

void load_lif_non_spiking_from_conf(LIF_non_spiking *lif, LIF_non_spiking_conf const *conf){
    lif->size = conf->size;
    lif->beta = conf->beta;
};