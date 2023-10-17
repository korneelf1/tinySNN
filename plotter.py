from SNN2 import SNN2
import torch
from datasets.primate_reaching import PrimateReaching
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
model = SNN2()
model.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/SNN2_{}.pt".format(filename), map_location=torch.device('cpu'))
                        ['model_state_dict'], strict=False)
file = ["indy_20160622_01.mat"]

dataset = PrimateReaching(file_path=file, filename=file,
                            num_steps=1, train_ratio=0.5, bin_width=0.004,
                            biological_delay=0, split_num=1, remove_segments_inactive=False)

test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=len(dataset.ind_test), shuffle=False)
for data, target in test_set_loader:
    predictions = model(data)
    break   
model.plot()