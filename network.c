#include "network.h"
#include "lif.h"
#include "lif_non_spiking.h"

// configurations
#include "network_configuration/inhid_conf.h"
#include "network_configuration/lif1_conf.h"
#include "network_configuration/lif2_conf.h"
#include "network_configuration/non_spiking_conf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

SNN build_snn(int const in_size, int const out_size, int const hidden_size){
    SNN snn;
    snn.in_size = in_size;
    snn.out_size = out_size;
    snn.hidden_size = hidden_size;

    snn.inhid  = malloc(sizeof(*snn.inhid));
    snn.hidhid = malloc(sizeof(*snn.hidhid));
    snn.hidout = malloc(sizeof(*snn.hidout));
    snn.output = malloc(sizeof(*snn.output));

    *snn.inhid = build_linear(in_size, out_size);
    *snn.hidhid = build_lif(in_size, hidden_size);
    *snn.hidout = build_lif(hidden_size, out_size);
    *snn.output = build_lif_non_spiking(out_size);

    load_weights(snn.inhid, &conf_inhid);
    load_lif_from_conf(snn.hidhid, &conf_hid1);
    load_lif_from_conf(snn.hidout, &conf_hid2);
    load_lif_non_spiking_from_conf(snn.output, &conf_out);
    return snn; 
};

void reset_net(SNN* snn){
    reset_lif(snn->hidhid);
    reset_lif(snn->hidout);
    reset_state(snn->output);
};

float* forward(SNN* snn, float* input){
    float* inhid_out = forward_linear(snn->inhid, input);
    float* hidhid_out = update_lif(snn->hidhid, inhid_out);
    float* hidout_out = update_lif(snn->hidout, hidhid_out);
    float* output = update(snn->output, hidout_out);
    return output;
};

void destroy_net(SNN* snn){
    destroy_linear(snn->inhid);
    destroy_lif(snn->hidhid);
    destroy_lif(snn->hidout);
    destroy(snn->output);
    free(snn);
};