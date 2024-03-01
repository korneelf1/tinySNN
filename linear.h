#pragma once

typedef struct linear
{
    int in_size;
    int out_size;
    float *weights;
    float *bias;
    float *output;
} linear;

typedef struct linear_conf
{
    int const in_size;
    int const out_size;
    float const *weights;
    float const *bias;
} linear_conf;

linear build_linear(int const in_size, int const out_size);

void load_weights(linear *l, linear_conf const *conf);
void destroy_linear(linear *l);

float* linear_forward(linear *l, float *input);