// NeuralNetwork.cpp

#include "HELPERS.h"
#include <algorithm> // for std::fill

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
