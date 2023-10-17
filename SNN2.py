import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import matplotlib.pyplot as plt
import matplotlib.animation as animation


## Define model ##
class SNN2(nn.Module):

    def __init__(self, window=50, input_size=96, hidden_size=50, tau=0.96, p=0.3, device='cpu'):
        super().__init__()

        # self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2
        self.surrogate = surrogate.fast_sigmoid(slope=20)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=False, device=device)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size, bias=False, device=device)

        self.lif1 = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
                              learn_threshold=False, reset_mechanism='zero')
        self.lif_out = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
                              learn_threshold=False, reset_mechanism='none')
        
        self.inputs   = []
        self.spikes_1 = []
        self.spikes_2 = []

        self.dropout = nn.Dropout(p)
        self.mem1, self.mem2 = None, None

        # self.register_buffer('inp', torch.zeros(window, self.input_size))

    def reset(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif_out.init_leaky()

    def single_forward(self, x):
        x = x.squeeze() # convert shape (1, input_dim) to (input_dim)
        cur1 = self.dropout(self.fc1(x))
        spk1, self.mem1 = self.lif1(cur1, self.mem1)

        cur2 = self.fc_out(spk1)
        _, self.mem2 = self.lif_out(cur2, self.mem2)
        
        self.inputs.append(x)
        self.spikes_1.append(spk1)
        self.spikes_2.append(self.mem2)

        return self.mem2.clone()

    def forward(self, x):
        # here x is expected to be shape (len_series, 1, input_dim)
        predictions = []

        for sample in range(x.shape[0]):
            predictions.append(self.single_forward(x[sample, ...]))

        predictions = torch.stack(predictions)
        return predictions

    def plot(self):
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_xlim(-1, self.input_size)
            ax.set_ylim(-1, 3)

            # plot input neurons
            for j in range(self.input_size):
                ax.add_artist(plt.Circle((j, 0), 0.2, color=plt.cm.Reds(self.inputs[i][j])))

            # plot spikes_1
            for j in range(self.hidden_size):
                ax.add_artist(plt.Circle((j, 1), 0.2, color=plt.cm.Blues(self.spikes_1[i][j])))

            # plot spikes_2
            for j in range(self.output_size):
                ax.add_artist(plt.Circle((j, 2), 0.2, color=plt.cm.Greens(self.spikes_2[i][j])))

        ani = animation.FuncAnimation(fig, animate, frames=len(self.inputs), interval=50)
        plt.show()



        