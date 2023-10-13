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
    float beta = 0.9;
    std::vector<bool> out;
    std::vector<float> states;
    const int threshold = 1;

    // constructors
    LIFNeuronLayer(int size);
    LIFNeuronLayer();
    void reset_states();
    void state_update(float& state, const float& inp);
    std::vector<bool> update(const std::vector<float>& input);
};

class AccumLinear {
public:
    int nr_ins;
    int nr_outs;
    std::vector<std::vector<float>> weights;
    std::vector<float> out;

    // constructor default
    AccumLinear();

    // Constructor with initialization
    AccumLinear(int nr_ins, int nr_outs);

    // Constructor with provided weights
    AccumLinear(int nr_ins, int nr_outs, const std::vector<std::vector<float>>& weights);

    // Function to perform the forward pass
    std::vector<float> forward(const std::vector<bool>& input);
};

#endif // NEURAL_NETWORK_H
