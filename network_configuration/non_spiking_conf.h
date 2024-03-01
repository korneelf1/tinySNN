#pragma once

#include "../lif_non_spiking.h"

static int const size_out = 7;
static const float beta_out = 0.8798;

LIF_non_spiking_conf const conf_out = {size_out, beta_out};