#include <iostream>
#include <random>
#include <vector>

class LIF{
    public:
        float beta;
        constexpr static int threshold = 1;
        float state;

        // constructors
        LIF(): beta(0.5), state(0){};

        LIF(float beta): beta(beta), state(0){};


    void reset(){
        this->state = 0;
    }

    bool update(float& input){
        if (state <= 0){
            state = input; // avoid multiply
            }
        else{
            state = state*beta + input;
        }
        if(state > threshold){
            state = 0;
            return true;
        }
        return false;
    }
};


class LIFNeuronLayer{
    public:
        int size;
        float beta =0.9;
        std::vector<bool> out; // Replace bool* with std::vector<bool>
        std::vector<float> states;

        const int threshold = 1;

        // constructors
        LIFNeuronLayer(int size) : size(size), states(size), out(size) {}


    void reset_states(){
        for (int i = 0; i < size; ++i) {
            states[i] = 0;
        }
    };

    void state_update(float& state, const float& inp){
    if (state <= 0){
            state = inp; // avoid multiply
            }
        else{
            state = state*beta + inp;
        }
}



std::vector<bool> update(const std::vector<float>& input) {
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

};


// class AccumLinear{
//     // todo: add bias
//     //       improve memory management
//     public:
//         int nr_ins;
//         int nr_outs;
//         float** weights;
//         float* out;
//         // Constructor with initialization
//         AccumLinear(int nr_ins, int nr_outs) : nr_ins(nr_ins), nr_outs(nr_outs) {
//             float weights[nr_ins][nr_outs];
//             float out[nr_outs];
//             // Initialize weights and output to zero
//             for (int i = 0; i < nr_ins; ++i) {
//                 for (int j = 0; j < nr_outs; ++j) {
//                     weights[i][j] = 0.1;
//                 }
//             }
//             for (int i = 0; i < nr_outs; ++i) {
//             out[i] = 0.0;
//             }
//         }
//         // Constructor with provided weights
//         AccumLinear(int nr_ins, int nr_outs, const float** input_weights) : nr_ins(nr_ins), nr_outs(nr_outs) {
//             float weights[nr_ins][nr_outs];
//             float out[nr_outs];
//             // Copy provided weights
//             for (int i = 0; i < nr_ins; ++i) {
//                 for (int j = 0; j < nr_outs; ++j) {
//                     weights[i][j] = input_weights[i][j];
//                 }
//             }

//             // Initialize output to zero
//             for (int i = 0; i < nr_outs; ++i) {
//                 out[i] = 0.0;
//             }
//         }
        
//     float* forward(float* input){
//         // Initialize output array to zero
//         for (int i = 0; i < nr_outs; ++i) {
//             out[i] = 0.0;
//         }

class AccumLinear {
public:
    int nr_ins;
    int nr_outs;
    std::vector<std::vector<float>> weights;
    std::vector<float> out;

    // Constructor with initialization
    AccumLinear(int nr_ins, int nr_outs) : nr_ins(nr_ins), nr_outs(nr_outs), weights(nr_ins, std::vector<float>(nr_outs, 0.0)), out(nr_outs, 0.0) {}

    // Constructor with provided weights
    AccumLinear(int nr_ins, int nr_outs, const std::vector<std::vector<float>>& weights) : nr_ins(nr_ins), nr_outs(nr_outs), weights(weights), out(nr_outs, 0.0) {}

    // Function to perform the forward pass
    std::vector<float> forward(const std::vector<bool>& input) {
        // Initialize output array to zero
        std::fill(out.begin(), out.end(), 0.0);

        for (int i = 0; i < nr_ins; i++)
        {
            if (input[i] == true)
            {
                for (int j = 0; j < nr_outs; ++j) {
                    out[j] += weights[i][j];
                }
            }
        }
        return out;
        
    }
    
    
};

int main(){
    LIF lif(0.5);

    AccumLinear layer1(3,5);

    // LIFNeuronLayer lif1(3);
    // LIFNeuronLayer lif2(5);

    // lif1.reset_states();
    // lif2.reset_states();


    // float in[2] = {.3, .4};
    std::vector<float> in = {.3, 1};
    
    // Example usage
    const int nr_ins = 3;
    const int nr_outs = 2;

     // Initialize weights
    const std::vector<std::vector<float>> input_weights = {{.30, .40}, {.70, -.30}, {-.20, .0}};

    // Create AccumLinear instance
    AccumLinear accumLinear(nr_ins, nr_outs, input_weights);

    // Example input
    const std::vector<float> input = {1.0, 5.0, 2.0};

    // Perform forward pass
    LIFNeuronLayer lif1(3);
    LIFNeuronLayer lif2(2);

    lif1.reset_states();
    lif2.reset_states();
    for (int i = 0; i < 10; ++i) {
        std::vector<bool> spk1 = lif1.update(input);
        std::vector<float> result = accumLinear.forward(spk1);
        std::vector<bool> spk2 = lif2.update(result);
        // Display result
        std::cout << "first spikes: ";
        for (int i = 0; i < nr_ins; ++i) {
            std::cout << spk1[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "first layer outputs: ";
        for (int i = 0; i < nr_outs; ++i) {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "second spikes: ";
        for (int i = 0; i < nr_outs; ++i) {
            std::cout << spk2[i] << " ";
        }
        std::cout << std::endl;
    }


    return 0;
}