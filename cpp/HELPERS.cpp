// NeuralNetwork.cpp

#include "HELPERS.h"
#include <cmath> // for exp
#include <random> // sampling
#include <algorithm> // max_element
#include <vector>

void LIF_non_spiking::reset_states(){
    std::vector<float> states(size, 0);
};

void LIF_non_spiking::initialize(const int size){
    this->size = size;
    this->states = std::vector<float>(size, 0);
};

std::vector<float> LIF_non_spiking::update(std::vector<float>& input) {
    for (int i = 0; i < size; ++i) {

            states[i] = states[i] * beta + input[i];
        
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
    std::vector<float> output_lif(bias);
    // std::vector<bool> spikes(bias.size(),false);

    for (int i = 0; i < size; ++i) {
        states[i] = states[i] * betas[i] + input[i];
        // in snntorch the threshold check affects output but not yet resetted, this only happens in next step
        // doesnt matter for hidden layer but matters for last? idk
        if (states[i] > thresholds[i]) {
            
            // spikes[i] = 1;
            states[i] = states[i] - thresholds[i]; // subtract reset mechanism
            for (int j = 0; j < nr_outs; ++j) {

                output_lif[j] += weights[j][i];
            }

            
        }
        
    }
    return output_lif;
}

void Softmax_Multinomial::initialize(const int size){
    this->size = size;
};

float Softmax_Multinomial::compute(std::vector<float>& input) {
    std::vector<float> probabilities(size);
    float max_val = *std::max_element(input.begin(), input.end());
    float exp_sum = 0.0;

    for (int i=0;i<size;i++) {
        float exp_val = std::exp(input[i] - max_val);
        probabilities[i] = exp_val;
        exp_sum += exp_val;
    };

    for (float& val : probabilities) {
        val /= exp_sum;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

    std::vector<float> actions = {-1., -0.66666667, -0.33333333, 0., 0.33333333, 0.66666667, 1.};
    return actions[dist(gen)];

}
