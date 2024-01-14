// NeuralNetwork.h

#ifndef HELPERS
#define HELPERS

#include <vector>

class LIF {
public:
    float beta;
    constexpr static int threshold = 1;
    float state;

    // constructors
    LIF() : beta(0.5), state(0) {}
    LIF(float beta) : beta(beta), state(0) {}
    
    void reset();
    bool update(float& input);
};

class LIFNeuronLayer {
public:
    int size;
    float beta = 0.8798;
    std::vector<bool> out;
    std::vector<float> states;
    const int threshold = 1;

    // constructors
    LIFNeuronLayer(int size);

    void reset_states();
    void state_update(float& state, float& inp);
    std::vector<bool> update(std::vector<float>& input);
};

class LIF_non_spiking {
public:
    int size;
    float beta = 0.8798;

    std::vector<float> states;

    // constructors
    LIF_non_spiking() : size(0) {};
    LIF_non_spiking(int size) : size(size), states(size, 0) {};

    void reset_states();
    std::vector<float> update(std::vector<float>& input);
};

class LIFNeuronLayer_var_beta_fused {
    // this is a fusion of the forward operator and neuron model, avoids multiplies
public:
    int size;
    int nr_outs;
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

class AccumLinear {
public:
    int nr_ins;
    int nr_outs;
    std::vector<std::vector<float>> weights;
    std::vector<float> out;

    // constructor default

    // Constructor with initialization
    AccumLinear(int nr_ins, int nr_outs);

    // Constructor with provided weights
    AccumLinear(int nr_ins, int nr_outs, std::vector<std::vector<float>>& weights);

    // Function to perform the forward pass
    std::vector<float> forward(std::vector<bool>& input);
};

#endif // NEURAL_NETWORK_H
