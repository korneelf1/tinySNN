// NeuralNetwork.cpp

#include "HELPERS.h"
#include <algorithm> // for std::fill
#include <vector>

void LIF::reset() {
    this->state = 0;
}

bool LIF::update(float& input) {
    if (state <= 0) {
        state = input; // avoid multiply
    } else {
        state = state * beta + input;
    }
    if (state > threshold) {
        state = 0;
        return true;
    }
    return false;
}

LIFNeuronLayer::LIFNeuronLayer(int size) : size(size), states(size), out(size) {}

void LIFNeuronLayer::reset_states() {
    for (int i = 0; i < size; ++i) {
        states[i] = 0;
    }
}

void LIFNeuronLayer::state_update(float& state, float& inp) {
    if (state <= 0) {
        state = inp; // avoid multiply
    } else {
        state = state * beta + inp;
    }
}

std::vector<bool> LIFNeuronLayer::update(std::vector<float>& input) {
    // now zero reset
    for (int i = 0; i < input.size(); ++i) {
        state_update(states[i], input[i]);

        if (states[i] > threshold) {
            states[i] = 0;
            out[i] = true;
        } else {
            out[i] = false;
        }
    }
    return out;
}

void LIF_non_spiking::reset_states(){
    std::vector<float> states(size, 0);
};

std::vector<float> LIF_non_spiking::update(std::vector<float>& input) {
    for (int i = 0; i < size; ++i) {
        if (states[i] <= 0) {
            states[i] = input[i]; // avoid multiply
        }
        else {
            states[i] = states[i] * beta + input[i];
        }
    }
return states;
}

void LIFNeuronLayer_var_beta_fused::initialize(const int size,const int nr_outs, const std::vector<float>& betas, const std::vector<float>& thresholds, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias){
    this->size = size;
    this->nr_outs = nr_outs;
    this->bias = bias;
    this->states = std::vector<float>(size, 0);
    this->betas = betas;
    this->thresholds = thresholds;
    this->weights = weights;
};
void LIFNeuronLayer_var_beta_fused::reset_states() {
    
    // Initialize a vector with zeros
    std::vector<float> states(size, 0);
    };

std::vector<float> LIFNeuronLayer_var_beta_fused::update(std::vector<float>& input) {
    // now zero reset
    // state_update(input);
    // std::vector<float> output = bias;
    std::vector<float> output(bias);

    for (int i = 0; i < size; ++i) {
        if (states[i] <= 0) {
            states[i] = input[i]; // avoid multiply
        }
        else {
            states[i] = states[i] * betas[i] + input[i];
        }
        // in snntorch the threshold check is an elif statement so that it only does it the next step
        if (states[i] > thresholds[i]) {
            states[i] = states[i] - thresholds[i]; // subtract reset mechanism
            for (int j = 0; j < nr_outs; ++j) {
                output[j] += weights[i][j] + bias[j];
            }
            
        }
    }
    return output;
}

AccumLinear::AccumLinear(int nr_ins, int nr_outs) : nr_ins(nr_ins), nr_outs(nr_outs), weights(nr_ins, std::vector<float>(nr_outs, 0.0)), out(nr_outs, 0.0) {}

AccumLinear::AccumLinear(int nr_ins, int nr_outs, std::vector<std::vector<float>>& weights) : nr_ins(nr_ins), nr_outs(nr_outs), weights(weights), out(nr_outs, 0.0) {}


std::vector<float> AccumLinear::forward(std::vector<bool>& input) {
    // Initialize output array to zero
    std::fill(out.begin(), out.end(), 0.0);

    for (int i = 0; i < nr_ins; i++) {
        if (input[i]) {
            for (int j = 0; j < nr_outs; ++j) {
                out[j] += weights[i][j];
            }
        }
    }
    return out;
}
