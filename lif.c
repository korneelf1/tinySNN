#include "lif.h"
#include <stdlib.h>

lif build_lif(int const in_size, int const out_size){
    lif lif;
    lif.in_size = in_size;
    lif.out_size = out_size;
    // lif.beta = beta;
    // lif.thresholds = thresholds;
    // lif.bias = bias;
    // lif.weights = weights;
    lif.beta = malloc(in_size * sizeof(*lif.beta));
    lif.thresholds = malloc(in_size * sizeof(*lif.thresholds));
    lif.bias = malloc(out_size * sizeof(*lif.bias));
    lif.weights = malloc(in_size*out_size * sizeof(*lif.weights));

    lif.states = calloc(in_size, sizeof(*lif.states));
    lif.output = calloc(out_size, sizeof(*lif.output));
    return lif;
};

void reset_lif(lif* neuron) {
    for (int i=0;i<neuron->in_size;i++){
        neuron->states[i] = 0.0f;
    }
};

float* update_lif(lif* neuron, float* input) {
    for (int i=0;i<neuron->in_size;i++){
        neuron->states[i] = neuron->states[i] * neuron->beta[i] + input[i];

        if (neuron->states[i] > neuron->thresholds[i]){
            neuron->states[i] = neuron->states[i] - neuron->thresholds[i];
            for (int j=0;j<neuron->out_size;j++){
                neuron->output[j] += neuron->weights[i*neuron->out_size + j];
            }
        } else {
            neuron->output[i] = 0.0f;
        }
    }
    return neuron->output;
};

void destroy_lif(lif* neuron) {
    free(neuron->states);
    free(neuron->in_size);
    free(neuron->out_size);
    free(neuron->beta);
    free(neuron->thresholds);
    free(neuron->bias);
    free(neuron->weights);

    free(neuron);
};

void load_lif_from_conf(lif *neuron, lif_conf const *conf){
    neuron->in_size = conf->in_size;
    neuron->out_size = conf->out_size;
    neuron->beta = conf->beta;
    neuron->thresholds = conf->thresholds;
    neuron->bias = conf->bias;
    neuron->weights = conf->weights;
};