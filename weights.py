import torch

# Load the saved model parameters
model_params = torch.load('SNN2.pt',map_location='cpu')['model_state_dict']

# Extract the weight matrices
w1 = model_params['fc1.weight'].numpy()
w2 = model_params['fc_out.weight'].numpy()

# Convert the weight matrices to C++ arrays
w1_cpp = 'float w1[' + str(w1.shape[0]) + '][' + str(w1.shape[1]) + '] = {\n'
for i in range(w1.shape[0]):
    w1_cpp += '{' + ', '.join([str(x) for x in w1[i]]) + '},\n'
w1_cpp += '};\n'

w2_cpp = 'float w2[' + str(w2.shape[0]) + '][' + str(w2.shape[1]) + '] = {\n'
for i in range(w2.shape[0]):
    w2_cpp += '{' + ', '.join([str(x) for x in w2[i]]) + '},\n'
w2_cpp += '};\n'

# Print the C++ arrays
print(w1_cpp)
print(w2_cpp)
import json
data = {'w1': w1_cpp, 'w2':w2_cpp}
with open('arrays.json', 'w') as json_file:
    json.dump(data, json_file)