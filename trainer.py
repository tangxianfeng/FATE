import torch
import numpy as np
import os

from models import FATE
from torch.utils.data import Dataset, DataLoader
import pickle

from tqdm import tqdm

class Trainer():
    def __init__(self, args):
        self.args = args
        self.snapnet = FATE(args.in_dim, args.hidden_dim, args.x_num_day)
        self.optim = torch.optim.Adam(self.snapnet.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss(reduction=True, size_average = True)
        self.snapnet.cuda()
        self.model_str = f'{args.x_num_day}_{args.y_num_day}'
        self.save_path = self.args.save_path.format(self.args.task, self.args.name, self.model_str)
        if not os.path.isdir(self.args.log_path):
            os.mkdir(self.args.log_path)
        if args.load_previous and os.path.isfile(self.save_path):
            self.snapnet.load_state_dict(torch.load(self.save_path))
            print(f'Loaded {args.name} from {self.save_path}')

    def train(self, dataset):
        print(f'Training {self.args.name}')
        if self.args.baseline:
            print('This is a baseline')
        else:
            print('This is not a baseline')
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        for ep in range(1, self.args.epochs + 1):
            print(f'Epoch {ep}')
            pbar = tqdm(data_loader)
            for batch_data in pbar:
                _, _, _, As, Is, Xs, Ys = batch_data
                self.optim.zero_grad()
                Y_preds, _, _, _ = self.snapnet.forward(As, Xs, Is)
                Ys = torch.stack(Ys)
                loss = self.loss_fn(Y_preds, Ys)
                loss.backward()
                self.optim.step()
                pbar.set_description_str(f'Batch average loss (min) {round(np.sqrt(loss.item()), 4)}')
            
            torch.save(self.snapnet.state_dict(), self.save_path)
