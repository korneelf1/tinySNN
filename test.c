#include "network.h"
#include "lif.h"
#include "lif_non_spiking.h"
#include "linear.h"

int main() {
    network snn = build_snn(1, 7, 32);
    float input[1] = {0.0f};
    destroy_net(&snn);
    return 0;
}