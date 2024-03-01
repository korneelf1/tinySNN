#include "linear.h"

linear build_linear(int const in_size, int const out_size){
    linear linear;
    linear.in_size = in_size;
    linear.out_size = out_size;
    linear.weights = malloc(in_size*out_size * sizeof(*linear.weights));
    linear.bias = malloc(out_size * sizeof(*linear.bias));
    linear.output = calloc(out_size, sizeof(*linear.output));
    return linear;
};

void load_linear_from_conf(linear *layer, linear_conf const *conf){
    layer->in_size = conf->in_size;
    layer->out_size = conf->out_size;
    layer->weights = conf->weights;
    layer->bias = conf->bias;
};

void destroy_linear(linear* layer) {
    free(layer->in_size);
    free(layer->out_size);
    free(layer->weights);
    free(layer->bias);
    free(layer);
};

float* forward_linear(linear* layer, float* input) {
    for (int i=0;i<layer->out_size;i++){
        for (int j=0;j<layer->in_size;j++){
            layer->output[i] += layer->weights[i*layer->in_size + j] * input[j];
        }
        layer->output[i] += layer->bias[i];
    }
    return layer->output;
};