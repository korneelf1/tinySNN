#include "network.h"
#include "lif.h"
#include "lif_non_spiking.h"


int main() {
    SNN snn = build_snn(1, 7, 32);
    float input[1] = {0.0f};§
    destroy_net(&snn);
    return 0;
}