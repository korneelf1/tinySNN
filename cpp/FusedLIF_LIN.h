#ifndef LIF_NEURON_LAYER_H
#define LIF_NEURON_LAYER_H

#include <stdbool.h>
#include <stdlib.h>

struct first_layer {
    int size;
    int nr_outs;
    bool reset; // saves if reset needs to be applied or not
    float bias[32];
    float output[32]; // You can use this as an array
    float states[32];
    float betas[32];
    float thresholds[32];
    float** weights[32][32]; // 2D array

    // Function to initialize the LIFNeuronLayer_var_beta_fused
   void (*initialize)(struct LIFNeuronLayer_var_beta_fused *layer, int size, int nr_outs);

    // Other functions can be added as needed
};

// Function to free the memory allocated for the LIFNeuronLayer_var_beta_fused instance
void destroy_LIFNeuronLayer_var_beta_fused(struct LIFNeuronLayer_var_beta_fused* layer);

#endif // LIF_NEURON_LAYER_H
