// NeuralNetwork.h

#ifndef HELPERS
#define HELPERS

#include <vector>

class LIF_non_spiking {
public:
    int size;
    float beta = 0.8798;

    std::vector<float> states;

    // constructors
    LIF_non_spiking() : size(0) {};
    LIF_non_spiking(int size) : size(size), states(size, 0) {};

    void reset_states();
    void initialize(const int size);
    std::vector<float> update(std::vector<float>& input);
};

class LIFNeuronLayer_var_beta_fused {
    // this is a fusion of the forward operator and neuron model, avoids multiplies
public:
    int size;
    int nr_outs;
    bool reset; // saves if reset needs to be applied or not
    std::vector<float> bias;
    // std::vector<float> output(nr_outs);
    // std::vector<bool> out;
    std::vector<float> states;
    std::vector<float> betas;
    std::vector<float> thresholds;
    std::vector<std::vector<float>> weights;

    // constructors
    LIFNeuronLayer_var_beta_fused(){
        this->size = 0;
        this->nr_outs = 0;
    };

    void initialize(const int size,const int nr_outs, const std::vector<float>& betas, const std::vector<float>& thresholds, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias);
    LIFNeuronLayer_var_beta_fused(const int size,const int nr_outs, const std::vector<float>& betas, const std::vector<float>& thresholds, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias){
        this->size = size;
        this->nr_outs = nr_outs;
        this->bias = bias;
        this->states = std::vector<float>(size, 0);
        this->betas = betas;
        this->thresholds = thresholds;
        this->weights = weights;
    };

    void reset_states();

    std::vector<float> update(std::vector<float>& input);
};

class Softmax_Multinomial {
public:
    int size;
    float output;

    // constructors
    Softmax_Multinomial() : size(0) {};
    Softmax_Multinomial(int size) : size(size){};

    void initialize(const int size);
    
    float compute(std::vector<float>& input);
};


#endif // NEURAL_NETWORK_H
