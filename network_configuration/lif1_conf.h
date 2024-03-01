#pragma once

#include "../lif.h"

static int const in_size_hidhid = 32;
static int const out_size_hidhid = 32;
static float const beta_hidhid[32] = {-5.0110e-04,  3.9099e-01,  9.8174e-02,  4.9266e-01,  1.4357e-01, 4.5383e-01,  1.0059e+00,  6.3891e-01,  4.7427e-01,  7.1373e-01,6.4809e-01, -6.8452e-03,  1.0131e+00,  2.9987e-01,  1.4174e-01,7.3930e-01, -8.6838e-04,  1.0027e+00,  1.0289e+00, -6.0622e-04,5.0713e-01,  1.0009e+00,  1.0015e+00,  1.0024e+00,  1.0080e+00,3.3709e-01,  3.8320e-01,  4.7710e-01, -2.2778e-03,  1.0194e+00,-2.9756e-03,  1.0006e+00};
static float const thresholds_hidhid[32] = {0.2671, 0.0605, 0.9955, 0.1458, 0.5549, 0.7777, 0.8410, 0.6095, 0.1820,
        0.2454, 0.7686, 0.3475, 0.9736, 0.3929, 0.0378, 0.1485, 0.6558, 0.7679,
        0.0661, 0.9301, 0.0711, 0.8897, 0.6263, 0.9470, 0.0771, 0.5237, 0.1483,
        0.5623, 0.1011, 0.0547, 0.2705, 0.1786};
static float const bias_hidhid[32] = {0.1867, -0.1613, -0.2832, -0.0997, -0.2014,  0.0764, -0.0909,  0.4307,
         0.2154,  0.1304,  0.0686, -0.1192, -2.1218,  0.1334, -0.0417, -4.0888,
         0.2054, -0.0089,  0.0566, -0.0114, -0.0630, -0.0217,  0.4305,  0.1388,
         0.2219,  0.1539, -0.0967,  0.0219,  0.3188, -0.0227, -1.8810, -0.4041};
static float const weights_hidhid[32*32] =  {1.7548e-01,  2.5603e-01,  1.3871e-01,  5.5432e-02,  1.5399e-01,
          1.8984e-01, -1.3097e-01, -5.2181e-01, -5.0593e-02,  6.1301e-01,
          3.0032e-01, -2.0746e+00, -2.4072e-01,  1.3734e-01, -4.6122e-02,
          7.9567e-02,  8.8891e-02,  2.4995e-01, -1.2600e+00,  5.0254e-01,
          1.5271e-01,  7.2549e-02,  1.6276e-01, -1.1167e-01,  6.1049e-02,
          7.6257e-03, -7.5143e-02,  4.6342e-01, -2.1815e-01, -1.6962e-01,
          3.4326e-01,  3.5554e-02,
         1.6505e-01,  6.9194e-02, -3.0454e-02,  1.2428e-01,  4.6881e-02,
         -1.1149e-01, -1.2637e-01,  4.2892e-01, -4.1283e-02, -2.8511e-01,
         -7.1347e-01, -1.2305e+00, -2.9389e-01, -1.0256e-02, -1.6617e-01,
          1.6737e-01,  1.3335e-02,  1.8855e-01,  7.5744e-01, -3.1497e-01,
         -6.0857e-02, -1.5554e-01, -1.0192e-01, -3.8505e-01,  3.6665e-01,
         -1.3476e-01, -9.6063e-01,  2.7917e-01,  1.5710e-01, -1.9908e-01,
         -7.8433e-01,  2.1951e-01,
         2.1339e-02, -2.9443e-01,  1.6983e-02, -7.7195e-02, -1.6580e-01,
          2.4689e-01,  8.8706e-02, -1.0239e+00,  1.5984e-01,  3.7136e-01,
          2.8058e-02, -3.0500e-01,  1.7079e-01,  1.0961e-01, -2.1082e-01,
          1.3764e-01,  4.2693e-03,  2.5503e-01, -2.4982e-02,  2.6303e-01,
          4.7087e-02,  2.5801e-01, -1.1478e-01,  1.6597e-01, -1.2560e-02,
         -4.6323e-02,  8.2597e-01, -1.1525e+00, -6.4155e-01,  5.1574e-01,
         -4.2325e-03,  7.4655e-02,
         1.0419e-01,  1.9665e-02, -1.5944e-01, -3.4968e-02, -1.5388e-01,
          4.1484e-02,  2.9520e-01, -1.0454e+00, -9.2466e-03, -5.4571e-01,
          1.8505e-01, -9.7896e-01, -6.2923e-01, -4.3754e-04,  9.3149e-02,
          8.6913e-02,  3.3465e-02, -2.6954e-01,  2.0336e-01, -1.0053e+00,
          7.6796e-02, -3.0381e-01, -9.2424e-02, -7.0503e-01,  4.5148e-01,
         -2.9036e-03, -7.1926e-01, -2.4045e-01,  4.9608e-01,  4.3308e-01,
         -6.8785e-01, -1.6735e-01,
         9.2384e-02, -1.7900e-01, -6.5816e-02, -7.5689e-02, -1.6896e-01,
         -7.5499e-03,  2.9418e-01, -8.7298e-01,  6.1606e-04,  3.7525e-01,
          3.6757e-01, -1.5200e-02,  1.6045e-02, -4.6890e-02, -1.6245e-01,
          1.1728e-01,  7.3314e-02,  1.1817e-01, -3.4397e-02,  5.1424e-01,
          4.1735e-02,  2.4647e-01, -1.1712e-01,  3.5139e-01, -2.1803e-01,
          8.9399e-02,  1.1339e+00, -1.5700e+00, -8.0552e-01,  3.1578e-01,
          3.6501e-01,  1.1062e-01,
         1.3666e-01, -1.3951e-01, -1.2554e-02,  1.2628e-01,  1.0344e-01,
          1.9065e-02, -1.4943e-01, -1.5142e+00, -1.2531e-01,  6.5355e-01,
         -4.7366e-02,  6.8648e-02, -9.2078e-02, -6.9081e-02, -1.1716e-01,
         -7.1528e-02, -1.1185e-01, -4.7112e-02, -2.6257e-01,  2.7312e-01,
          1.7170e-01,  1.5991e-01, -6.7594e-02, -6.4080e-02,  4.0695e-01,
          7.7544e-02,  1.0185e+00, -7.1075e-02, -9.0093e-01,  2.0514e-01,
          9.0254e-01, -1.4500e-02,
        -1.6526e-01, -1.5866e-01, -1.4402e-01, -1.2738e-01, -1.5877e-01,
         -1.4661e-01,  6.0210e-02, -1.1142e+00, -1.7486e-01,  2.7851e-01,
          6.2690e-01,  5.4785e-01, -3.9622e-02,  1.3951e-01,  4.1766e-02,
         -1.0528e-01, -3.2034e-02,  5.3349e-01, -6.6611e-01, -1.1043e-01,
          8.6276e-02,  1.3780e-01, -2.6950e-02,  2.6313e-01, -1.2654e-01,
         -6.3647e-02,  7.4476e-02, -7.6861e-01, -9.4887e-01,  2.1433e-01,
         -4.4775e-01,  3.7729e-02,
         1.2756e-01,  3.6942e-01, -5.8917e-02,  2.4160e-02,  1.1823e-01,
         -3.0418e-01,  8.8674e-02,  3.7916e+00,  1.0478e-01, -5.5230e-01,
         -1.6814e-01, -1.3491e-01,  1.7032e-01, -1.1383e-01,  3.7753e-01,
          4.0815e-02, -6.8728e-02,  2.9958e-01,  3.8383e-01, -2.5418e-01,
          8.9955e-02, -3.6954e-02,  1.4810e-01, -2.9292e-01, -2.1793e-01,
          9.1293e-02, -1.8242e+00,  1.2147e+00,  6.1892e-01, -2.9331e-01,
         -8.0362e-01, -9.8622e-02,
         1.1707e-01,  2.4769e-01, -1.6939e-01,  1.6543e-01,  1.7419e-01,
         -6.7040e-01, -1.9103e-02,  4.1167e-01,  7.5748e-02, -1.0344e+00,
         -4.2610e-01, -1.0573e+00, -2.5278e-01, -1.0541e-01, -1.5560e-02,
          2.1625e-02,  1.2361e-01, -2.5731e-02, -6.2711e-01, -1.2973e+00,
         -1.7172e-01,  7.0276e-01,  1.3511e-01,  6.8968e-01,  2.9740e-01,
         -1.3108e-01, -2.1434e+00,  3.3164e-01,  4.6518e-01,  1.1091e-01,
         -2.2410e+00,  5.8541e-02,
         1.1158e-01,  4.3626e-02, -5.4887e-03,  9.0898e-03,  9.4677e-02,
          1.3147e-01, -9.6049e-02,  2.7833e+00,  1.4299e-02, -8.3193e-02,
         -6.2095e-01,  9.7191e-01,  5.5673e-01,  1.0337e-01,  1.0784e-01,
         -1.4651e-01,  7.9241e-02,  6.6357e-01,  1.8237e-01,  3.3042e-01,
          1.1655e-01,  1.5186e-01,  1.6213e-01,  7.3286e-02, -7.0549e-01,
          1.3960e-01,  4.1604e-01,  8.1837e-02, -3.1107e-01, -2.7811e-01,
          5.3912e-01,  1.6255e-01,
         9.9424e-02, -3.4591e-01, -2.0183e-02,  2.4880e-02,  7.9982e-02,
         -8.1412e-02, -1.1901e-01, -5.5619e-01,  1.0231e-01,  9.6055e-02,
          6.3845e-02,  3.7932e-01, -7.3438e-03,  1.2407e-01, -3.8139e-01,
          9.7046e-02,  2.8381e-02,  4.5328e-02, -7.3563e-01, -2.4832e-01,
          1.1876e-02,  5.6643e-01, -1.0316e-01,  4.5894e-01,  3.2028e-02,
         -1.0061e-01,  1.0080e-01, -5.1207e-01, -1.2153e+00,  3.4838e-01,
         -1.1082e-01, -1.2152e-01,
         2.8776e-02,  1.4490e+00,  2.2972e-02,  1.2327e-02, -1.3305e-02,
         -2.1157e-01,  2.8154e-02,  3.6336e-01, -1.7360e-01, -5.4355e-01,
         -9.2571e-02, -9.4031e-01,  2.1661e-01, -1.0837e-01,  4.8511e-01,
          1.2901e-01,  1.6624e-01,  7.2110e-01, -5.9837e-01, -9.8077e-02,
         -1.1969e-01,  6.4458e-01, -1.0987e-01,  1.1707e+00,  1.0364e+00,
         -1.5508e-01, -8.6504e-01,  1.7742e+00,  8.2497e-01,  2.8599e-01,
         -1.6533e-01, -1.0032e-01,
        -9.8340e-02, -3.2186e+00, -3.7893e-02, -1.2505e-01, -1.4031e-01,
          3.3384e-01, -4.9825e-02, -8.0870e-01, -3.3238e-02,  7.5303e-01,
         -7.9362e-02, -1.5647e+00, -1.4036e+00,  1.2327e-01, -2.7752e+00,
         -1.3218e-01, -4.2185e-02, -2.7463e+00,  3.5708e-03, -1.3530e+00,
         -1.2914e-01, -2.6328e-01,  1.2175e-01, -6.4352e-01, -4.8536e-01,
         -1.3183e-01,  3.6418e-01, -1.5464e+00, -3.2301e+00, -3.7693e+00,
          6.0644e-01,  1.5776e-01,
         3.8134e-02,  1.7932e-01,  7.7787e-02,  1.6245e-01,  3.8873e-02,
         -1.0108e+00, -1.5456e-01, -5.3002e-01,  1.6265e-01, -6.1804e-01,
          2.5757e-01,  9.6545e-01,  6.0524e-02,  6.1388e-02, -1.1184e-01,
          1.0965e-03, -2.8960e-03,  7.1981e-01, -1.3099e+00, -8.2445e-01,
         -6.9384e-02,  3.7723e-01, -8.2515e-02,  3.2718e-01,  9.4013e-01,
         -3.6145e-02, -2.2855e+00,  8.7666e-02,  7.7357e-02,  6.2154e-01,
         -7.3357e-01, -1.3710e-02,
         5.6355e-02,  2.3203e-01, -5.2528e-02, -1.0955e-01,  1.4516e-01,
         -4.4526e-01, -1.2888e-02,  1.4690e+00, -1.3918e-01, -1.4283e+00,
          4.0461e-01, -6.8948e-01, -2.5477e-01,  6.0435e-02,  1.4897e-01,
          7.0944e-02,  1.0906e-01,  3.2131e-01, -3.5000e-01, -8.8650e-01,
          1.1848e-01,  2.6389e-01,  5.8454e-02,  2.5135e-01,  2.2182e-02,
         -1.2250e-02, -1.6840e+00,  3.5128e-01,  5.0874e-01, -1.7467e-01,
         -1.0841e+00, -9.5123e-03,
         1.1052e-02, -4.3960e+00,  1.1576e-01, -2.1202e-02,  2.1674e-02,
         -6.9671e-01, -3.2314e-01, -1.8855e-01,  1.3070e-01, -1.7762e-01,
         -1.3790e+00, -1.8136e+00, -3.2832e+00,  7.1342e-03, -4.1937e+00,
         -3.6410e-02, -9.1504e-02, -2.5081e+00, -7.7438e-01, -1.8480e+00,
         -5.1530e-02, -4.7424e-01, -1.6024e-01, -6.8116e-01, -2.3088e-01,
          7.2052e-03,  1.1624e-01, -4.6089e+00, -4.2543e+00, -1.5480e+00,
         -1.3239e+00, -6.2687e-03,
        -6.2329e-02, -3.1749e-02, -7.7677e-02,  1.6155e-01,  1.3125e-01,
          1.4040e+00,  1.8083e-01,  5.4695e-01,  3.0933e-02,  7.1311e-01,
          9.3686e-02, -2.6550e-01,  1.5792e-01,  4.1778e-02,  1.0228e-01,
         -1.4305e-01, -1.6639e-01, -5.7170e-01,  1.2193e+00,  3.7862e+00,
          1.3143e-01,  8.2210e-02,  9.7714e-02,  1.6110e-01, -2.0356e-01,
          1.0770e-01,  1.7621e+00,  1.3001e-01, -7.1563e-02, -1.1843e-01,
          1.5700e+00,  7.7421e-02,
         1.5413e-01, -2.9556e-01, -8.3829e-02,  4.3119e-02, -4.1789e-02,
          2.4500e-01,  6.7584e-02, -1.4779e+00,  6.3921e-02,  6.0905e-01,
          2.3840e-01,  2.0221e-01,  2.0940e-01, -9.7609e-02, -3.4610e-01,
          5.8509e-02, -4.1858e-03, -6.9325e-02,  1.0297e-02,  5.9129e-01,
          1.6597e-01,  3.5724e-01, -9.3909e-02,  2.4402e-01,  6.6945e-02,
          1.4502e-01,  1.1482e+00, -1.0061e+00, -8.4647e-01,  1.1410e-01,
          5.0091e-01,  4.7513e-02,
        -7.2878e-02, -2.1479e-01, -4.1860e-02,  1.7632e-01,  1.1382e-01,
          3.8650e-02, -1.1420e-01, -1.0168e+00,  9.0240e-02,  2.0711e-01,
          2.3630e-01,  2.4950e-01, -1.5673e-01,  9.4185e-02, -6.7543e-02,
         -1.5489e-01, -1.1955e-01,  6.0587e-01, -8.0417e-01, -7.1082e-01,
         -6.1847e-02,  1.7577e-01, -1.3219e-01, -1.6420e-01,  2.7586e-01,
          6.1573e-02,  6.0107e-01, -9.5065e-01, -1.1250e+00,  1.4920e-01,
          1.8179e-01, -1.7928e-01,
         1.3077e-01,  3.1700e-01, -3.6942e-02,  1.3563e-01, -9.3561e-02,
          4.7585e-01,  1.6484e-01,  3.3772e-01, -1.3545e-01,  2.4099e-01,
          2.9359e-01, -5.0070e-01,  6.5236e-02, -1.4139e-01, -6.4813e-01,
          2.4394e-02,  3.7736e-02,  1.4527e+00, -1.1477e+00,  3.5966e-01,
          6.1668e-02, -8.4286e-01,  5.1242e-02, -1.2032e+00,  9.2991e-01,
          8.2357e-02, -2.1153e-01, -1.0120e+00, -3.0246e-01, -1.5234e-01,
          2.0777e-01,  8.1320e-02,
         2.9713e-02, -3.2300e-01, -1.3486e-01, -1.4287e-01, -4.5196e-02,
          2.5071e-01,  4.7538e-02, -2.5547e-01, -1.3279e-01,  5.8536e-01,
          7.4633e-02,  9.5708e-01,  3.6255e-01,  1.0723e-01,  8.2264e-02,
          9.6873e-02,  4.4642e-02,  4.7071e-01, -1.7871e+00,  1.0581e+00,
          1.0447e-01,  2.8181e-01,  9.1171e-02,  3.7104e-02,  6.6525e-01,
          7.7212e-02, -7.0464e-02, -1.0775e-01,  2.8250e-01, -2.1396e-01,
          5.4778e-01, -1.5027e-01,
        -1.0188e-01, -1.2471e-01, -1.5925e-01,  1.1510e-01, -5.3118e-02,
          3.4868e-02,  6.2816e-02, -5.3450e-02, -5.7241e-02, -2.0994e-01,
          2.9568e-02, -1.2217e+00, -3.5549e-01,  1.6153e-01,  1.2159e-03,
         -1.3489e-01,  1.0996e-01, -7.1824e-02,  4.5313e-01, -5.3987e-01,
         -5.0106e-02,  1.7844e-01,  9.0063e-02,  2.7671e-01,  3.6566e-01,
         -8.2384e-03, -6.1070e-01, -2.8256e-01,  6.4019e-01,  4.5350e-01,
         -8.9888e-01,  6.9905e-02,
         3.5058e-02,  2.6368e-01,  5.7472e-02, -5.5304e-02, -8.0721e-02,
          1.7698e-01, -3.4535e-01,  2.7111e+00,  4.2903e-03, -6.0751e-01,
         -1.1099e+00, -2.0122e-01,  2.8228e-01,  4.6830e-03,  2.8730e-01,
          1.6861e-01,  1.2939e-01,  5.6885e-01,  4.4772e-01, -5.0856e-01,
         -6.5620e-02, -3.0248e-01,  3.9738e-02, -4.7886e-01,  2.9518e-01,
         -7.7713e-02, -1.5325e+00,  1.2134e+00,  6.3600e-01,  1.5445e-02,
         -9.7451e-01, -2.3870e-03,
        -1.6319e-01, -5.6847e-02, -4.3167e-02,  4.1247e-02,  2.5166e-02,
          2.8410e-01,  6.5254e-02,  1.5950e-01,  9.1206e-02,  6.3753e-01,
          6.1571e-01, -2.6057e-03,  4.5978e-01, -1.6568e-01,  1.5495e-01,
         -1.4448e-02, -5.1036e-02,  1.0587e+00, -1.6582e-01,  7.3071e-01,
          6.2485e-02,  1.1522e+00,  8.6929e-02,  1.8864e+00,  8.9835e-01,
          9.1931e-02, -6.8018e-03,  7.1509e-01, -2.2005e-01, -3.7357e-01,
          8.9581e-01,  7.2476e-03,
        -1.7525e-01, -1.1613e-01,  1.7343e-01,  1.1658e-01,  1.6825e-01,
          1.8407e-01,  2.7717e-01, -1.5887e-01, -1.5965e-01, -3.3403e-01,
          7.2352e-01, -8.5637e-01, -7.4613e-01, -1.7074e-01,  2.9456e-02,
         -1.1981e-01,  1.4134e-01, -3.2879e-01, -5.6149e-01, -1.4426e+00,
          1.2586e-01,  4.0368e-01,  1.4672e-03,  5.0586e-01,  2.8404e-01,
         -5.3887e-03, -1.5366e+00, -2.4416e-01,  6.8769e-01,  3.1338e-01,
         -2.2222e+00, -4.8044e-02,
        -8.4294e-02, -4.9897e-02, -1.2434e-02,  2.7268e-02, -8.3688e-02,
         -7.7309e-01, -1.9426e-01,  1.7214e+00, -4.0216e-02, -5.6046e-01,
         -9.3916e-01, -7.4344e-03,  6.2391e-03,  1.2387e-01,  9.8529e-02,
         -6.0764e-02, -1.1993e-01,  4.0427e-01,  1.6415e-01, -2.7868e-01,
          1.1371e-01,  5.4118e-01, -1.5937e-01,  5.9792e-01, -3.3949e-01,
         -1.0301e-01, -1.5764e+00,  4.7320e-01,  1.0847e-01, -2.4363e-01,
         -8.5283e-01, -1.4536e-01,
        -8.5890e-02, -2.0027e-01,  3.4056e-02, -1.6388e-01, -1.4918e-02,
         -1.7651e-01, -1.4028e-01,  1.5576e-01, -7.0524e-02, -1.0080e-01,
          1.3409e-01,  2.2839e-01,  1.0744e-01, -6.4371e-02,  2.2315e-02,
         -1.4916e-01,  1.0036e-01, -3.6542e-01, -4.4470e-01,  3.3380e-01,
         -1.5314e-01,  3.5375e-01, -1.7584e-01,  3.3317e-01, -1.6834e-01,
         -2.0925e-02, -1.2189e-01, -5.4156e-01, -2.9384e-01,  2.7309e-01,
         -7.9478e-03,  2.2694e-01,
         1.0478e-01, -9.1868e-02,  1.4111e-02,  1.2244e-01,  1.6586e-01,
         -4.6351e-01, -2.2930e-02,  1.1544e+00,  1.0575e-01, -1.4021e+00,
         -4.5551e-01, -1.2069e+00,  1.6026e-02,  1.5416e-01,  1.3979e-01,
         -1.1213e-01,  1.0212e-01,  5.2627e-01, -6.3652e-01, -1.1063e+00,
         -6.3875e-02,  4.3710e-01,  8.9601e-02,  6.1608e-01, -3.0767e-01,
          1.4039e-01, -1.9866e+00,  5.6352e-01,  5.6850e-01,  2.3532e-01,
         -1.7949e+00, -1.0458e-01,
        -1.2646e-01,  3.6868e-01,  1.5648e-01, -6.0103e-02, -4.3062e-02,
         -2.6142e-01, -2.6219e-02,  4.1728e+00,  1.0142e-01, -3.0229e-01,
         -2.3517e-01, -1.6838e-01, -1.7543e-02,  1.2538e-01,  2.8990e-01,
         -3.8520e-02,  9.1202e-02,  5.4514e-02,  4.2804e-01, -1.1698e-01,
          1.0131e-01,  3.6537e-02, -1.2886e-01, -4.7617e-02, -3.7408e-01,
          5.5465e-03, -1.0219e+00,  1.0153e+00,  6.5207e-01, -3.0607e-01,
         -5.1206e-01,  1.7194e-01,
        -1.5746e-01, -3.3237e-01,  1.4681e-02,  4.9753e-02, -1.4142e-01,
          4.7647e-01,  5.0729e-02, -3.8600e+00, -1.2136e-01,  1.4174e+00,
         -4.4668e-01,  9.3460e-01,  2.5559e-03, -1.2557e-01, -6.0575e-02,
          7.6487e-02, -1.4555e-01, -4.6791e-03, -2.7607e-01,  9.7550e-01,
          1.3217e-01, -3.6528e-04,  1.3959e-02, -2.0900e-02,  1.6123e-01,
          8.0793e-02,  2.0268e+00, -8.2359e-01, -5.4122e-01,  1.6727e-01,
          1.5252e+00, -2.1142e-02,
         1.2644e-01, -3.3021e+00,  1.7572e-01,  1.4050e-01, -5.6199e-02,
          7.4617e-01, -8.2134e-02, -4.2878e-01, -1.7155e-01,  7.2132e-01,
          3.3774e-01, -7.0433e-01, -1.2664e+00,  1.0709e-01, -3.0890e+00,
          1.1075e-01,  9.1006e-02, -1.0545e+00,  4.8299e-01, -5.6568e-01,
         -4.4831e-02, -1.1421e-01, -3.7452e-02, -1.9521e-01, -2.5059e-01,
          5.0803e-02,  9.1811e-01, -1.9107e+00, -3.3633e+00, -3.0310e+00,
          5.3608e-01,  5.5582e-02,
        -1.5031e-01, -7.1966e-01,  1.4492e-01,  1.6084e-01,  1.7701e-02,
          4.5027e-01,  2.4283e-01, -4.3155e-01,  1.7130e-01,  1.9000e-01,
          6.3447e-01, -1.5357e-01, -1.6154e-01, -1.0352e-01, -4.4483e-01,
         -7.7662e-02, -1.4062e-01, -9.2750e-01, -7.9699e-03, -1.6834e-01,
          9.3703e-02, -2.5520e-01,  1.2291e-01, -4.4294e-01, -1.4932e+00,
         -1.5450e-01,  1.0180e+00, -1.2136e+00, -6.8896e-01, -4.7930e-01,
          1.9971e+00,  6.8058e-02};

lif_conf const conf_hid1 = {in_size_hidhid, out_size_hidhid, beta_hidhid, thresholds_hidhid, bias_hidhid, weights_hidhid};
