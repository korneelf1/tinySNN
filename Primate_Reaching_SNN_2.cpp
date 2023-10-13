#include "HELPERS.h" 
#include <vector>

// architecture
class Model{
    int inp_size = 96;
    int out_size = 2;
    int hidden_size = 50;
    AccumLinear layer1;
    LIFNeuronLayer layer2;
    AccumLinear layer3;
    LIFNeuronLayer layer4;

    Model(){
        AccumLinear layer1(inp_size, hidden_size);
        LIFNeuronLayer layer2(hidden_size);
        AccumLinear layer3(hidden_size, out_size);
        LIFNeuronLayer layer4(out_size);
    };

    std::vector<bool> forward(std::vector<bool> input){
        std::vector<float> layer1_out = layer1.forward(input);
        std::vector<bool> layer2_out = layer2.update(layer1_out);
        std::vector<float> layer3_out = layer3.forward(layer2_out);
        std::vector<bool> layer4_out = layer4.update(layer3_out);
        return layer4_out;
    };
};