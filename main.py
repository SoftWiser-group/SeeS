import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from scipy.io import loadmat
import os

from encoder import *
from decoder import *

from constant import *



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class Sees_model(nn.Module):
    
    def __init__(self, device='cpu', model_path='defaultpath.model'):
        super(Sees_model, self).__init__()
        print('Using device: {}'.format(device))
        self.model_path = model_path
        self.device = device
        self.n_node = -1
        self.support = None
        self.train_ = [None, None]
        self.val_ = [None, None]
        self.test_ = [None, None]
    
    def INIT(self, data_path):
        self.read_data(data_path)
        self.encoder = seesEncoderWithGraphAttention(INPUT_DIM,
                                                       HID_DIM,
                                                       N_LAYERS,
                                                       N_HEADS,
                                                       PF_DIM,
                                                       SEQ_LEN,
                                                       WALK_LEN,
                                                       self.n_node,
                                                       self.get_device())
        self.decoder = seesDecoderWithAttInputs(HID_DIM,
                                                   OUT_DIM,
                                                   SEQ_LEN,
                                                   self.get_device())
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
    
    def get_device(self):
        if self.device == 'gpu':
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def read_data(self, path):
        
        train_path = path+'/train.npz'
        val_path = path+'/val.npz'
        test_path = path+'/test.npz'
        net_path = path+'/net.mat'
        net_path2 = path+'/N.npy'
        
        if os.path.exists(net_path):
            net = loadmat(net_path)
            support = net['S']
            print(support.shape)
        else:
            support = np.load(net_path2)
        
        N_NODE = support.shape[0]
        self.n_node = support.shape[0]
        #support[range(N_NODE),range(N_NODE)] = np.ones(N_NODE)
        self.support = support
        
        train_npz = np.load(train_path)
        train_x_ = train_npz['x']
        train_y_ = train_npz['y']
        train_y_ = np.split(train_y_, 2, axis=-1)[0]
        # train_x_, (data_len, seq_len, n_node, input_dim)
        # train_y_, (data_len, seq_len, n_node, out_dim)
        train_x_ = train_x_.transpose(2,0,1,3)
        train_y_ = train_y_.transpose(2,0,1,3)
        #print(train_x_[0][0].shape)
        # train_x_, (n_node, data_len, seq_len, input_dim)
        # train_y_, (n_node, data_len, seq_len, out_dim)
        
        val_npz = np.load(val_path)
        val_x_ = val_npz['x']
        val_y_ = val_npz['y']
        val_y_ = np.split(val_y_, 2, axis=-1)[0]
        
        val_x_ = val_x_.transpose(2,0,1,3)
        val_y_ = val_y_.transpose(2,0,1,3)
        
        test_npz = np.load(test_path)
        test_x_ = test_npz['x']
        test_y_ = test_npz['y']
        test_y_ = np.split(test_y_, 2, axis=-1)[0]
        
        test_x_ = test_x_.transpose(2,0,1,3)
        test_y_ = test_y_.transpose(2,0,1,3)
        
        self.train_ = [train_x_, train_y_]
        self.val_ = [val_x_, val_y_]
        self.test_ = [test_x_, test_y_]
        
        print('Training size: {}'.format(train_x_.shape[1]))
        print('Validating size: {}'.format(val_x_.shape[1]))
        print('Testing size: {}'.format(test_x_.shape[1]))
    
    def gen_walk(self, start_idx, walk_len):
        walk = [start_idx]
        for i in range(walk_len):
            neigh_weight = self.support[walk[-1]]
            neigh_nonzero = neigh_weight.nonzero()[0]
            sum_neigh = sum([float(neigh_weight[_]) for _ in neigh_nonzero])
            neigh_prob = [neigh_weight[_]/sum_neigh for _ in neigh_nonzero]
            r = random.random()
            for j in range(len(neigh_nonzero)):
                if r < sum(neigh_prob[:j+1]):
                    walk.append(neigh_nonzero[j])
                    break
        return np.array(walk[1:])
    
    def get_neigh(self, src_node, walk_len):
        neighs = []
        neigh_nonzero = self.support[src_node].nonzero()[0]
        for node in neigh_nonzero:
            neighs.append(node)
        while len(neighs) < walk_len:
            neighs.append(self.n_node)
        return np.array(neighs[:walk_len])
    
    def get_batch(self, pdata):
        curr_idx = 0
        c_i, c_j = pdata[0].shape[0], pdata[0].shape[1]
        
        while curr_idx < c_i*c_j:
            start_idx = curr_idx
            end_idx = min(curr_idx + BATCH_SIZE, c_i*c_j)
            
            src = []
            src_mask = []
            trg = []
            walks = []
            walk_src = []
            self_node = []
            for i in range(end_idx-start_idx):
                s_i = int((start_idx+i)/c_j) # idx of node
                s_j = (start_idx+i)%c_j # idx of data seq
                
                src.append(pdata[0][s_i][s_j])
                trg_full = pdata[1][s_i][s_j]
                trg_first = np.split(trg_full, SEQ_LEN, axis=0)[0]
                trg.append(trg_first)
                src_mask.append([1 for _ in range(SEQ_LEN)])
                
                #walk = self.gen_walk(s_i, WALK_LEN)
                walk = self.get_neigh(s_i, WALK_LEN)
                walks.append(walk)
                walk_src.append([])
                walk_src[i] = []
                for w_idx in walk:
                    if w_idx == self.n_node:
                        walk_src[i].append([[0.0, 0.0] for _ in range(SEQ_LEN)])
                    else:
                        walk_src[i].append(pdata[0][w_idx][s_j])
                #walk_src[i] = [self.pdata[w_idx][s_j] for w_idx in walk]
                self_node.append(s_i)
        
            curr_idx = end_idx
            yield (torch.tensor(src, dtype=torch.float, device=self.get_device()), 
                    torch.tensor(src_mask, dtype=torch.int64, device=self.get_device()), 
                    torch.tensor(trg, dtype=torch.float, device=self.get_device()), 
                    torch.tensor(walks, dtype=torch.int64, device=self.get_device()), 
                    torch.tensor(walk_src, dtype=torch.float, device=self.get_device()),
                    torch.tensor(self_node, dtype=torch.int64, device=self.get_device()))
            
    def get_train_batch(self):
        curr_idx = 0
        c_i, c_j = self.train_[0].shape[0], self.train_[0].shape[1]
        
        while curr_idx < c_i*c_j:
            start_idx = curr_idx
            end_idx = min(curr_idx + BATCH_SIZE, c_i*c_j)
            
            src = []
            src_mask = []
            trg = []
            walks = []
            walk_src = []
            self_node = []
            for i in range(end_idx-start_idx):
                s_i = int((start_idx+i)/c_j) # idx of node
                s_j = (start_idx+i)%c_j # idx of data seq
                
                src.append(self.train_[0][s_i][s_j])
                trg_full = self.train_[1][s_i][s_j]
                trg_first = np.split(trg_full, SEQ_LEN, axis=0)[0]
                trg.append(trg_first)
                src_mask.append([1 for _ in range(SEQ_LEN)])
                
                #walk = self.gen_walk(s_i, WALK_LEN)
                walk = self.get_neigh(s_i, WALK_LEN)
                walks.append(walk)
                walk_src.append([])
                walk_src[i] = [self.train_[0][w_idx][s_j] for w_idx in walk]
                self_node.append(s_i)
        
            curr_idx = end_idx
            yield (torch.tensor(src, dtype=torch.float, device=self.get_device()), 
                    torch.tensor(src_mask, dtype=torch.int64, device=self.get_device()), 
                    torch.tensor(trg, dtype=torch.float, device=self.get_device()), 
                    torch.tensor(walks, dtype=torch.int64, device=self.get_device()), 
                    torch.tensor(walk_src, dtype=torch.float, device=self.get_device()),
                    torch.tensor(self_node, dtype=torch.int64, device=self.get_device()))
            
    def get_val_batch(self):
        curr_idx = 0
        c_i, c_j = self.val_[0].shape[0], self.val_[0].shape[1]
        
        while curr_idx < c_i*c_j:
            start_idx = curr_idx
            end_idx = min(curr_idx + BATCH_SIZE, c_i*c_j)
            
            src = []
            src_mask = []
            trg = []
            walks = []
            walk_src = []
            self_node = []
            for i in range(end_idx-start_idx):
                s_i = int((start_idx+i)/c_j) # idx of node
                s_j = (start_idx+i)%c_j # idx of data seq
                
                src.append(self.val_[0][s_i][s_j])
                trg_full = self.val_[1][s_i][s_j]
                trg_first = np.split(trg_full, SEQ_LEN, axis=0)[0]
                trg.append(trg_first)
                src_mask.append([1 for _ in range(SEQ_LEN)])
                
                #walk = self.gen_walk(s_i, WALK_LEN)
                walk = self.get_neigh(s_i, WALK_LEN)
                walks.append(walk)
                walk_src.append([])
                walk_src[i] = [self.val_[0][w_idx][s_j] for w_idx in walk]
                self_node.append(s_i)
        
            curr_idx = end_idx
            yield (torch.tensor(src, dtype=torch.float, device=self.get_device()), 
                    torch.tensor(src_mask, dtype=torch.int64, device=self.get_device()), 
                    torch.tensor(trg, dtype=torch.float, device=self.get_device()), 
                    torch.tensor(walks, dtype=torch.int64, device=self.get_device()), 
                    torch.tensor(walk_src, dtype=torch.float, device=self.get_device()),
                    torch.tensor(self_node, dtype=torch.int64, device=self.get_device()))
            
    def get_test_batch(self):
        curr_idx = 0
        c_i, c_j = self.test_[0].shape[0], self.test_[0].shape[1]
        
        while curr_idx < c_i*c_j:
            start_idx = curr_idx
            end_idx = min(curr_idx + BATCH_SIZE, c_i*c_j)
            
            src = []
            src_mask = []
            trg = []
            walks = []
            walk_src = []
            self_node = []
            for i in range(end_idx-start_idx):
                s_i = int((start_idx+i)/c_j) # idx of node
                s_j = (start_idx+i)%c_j # idx of data seq
                
                src.append(self.test_[0][s_i][s_j])
                trg_full = self.test_[1][s_i][s_j]
                trg_first = np.split(trg_full, SEQ_LEN, axis=0)[0]
                trg.append(trg_first)
                src_mask.append([1 for _ in range(SEQ_LEN)])
                
                #walk = self.gen_walk(s_i, WALK_LEN)
                walk = self.get_neigh(s_i, WALK_LEN)
                walks.append(walk)
                walk_src.append([])
                walk_src[i] = [self.test_[0][w_idx][s_j] for w_idx in walk]
                self_node.append(s_i)
        
            curr_idx = end_idx
            yield (torch.tensor(src, dtype=torch.float, device=self.get_device()), 
                    torch.tensor(src_mask, dtype=torch.int64, device=self.get_device()), 
                    torch.tensor(trg, dtype=torch.float, device=self.get_device()), 
                    torch.tensor(walks, dtype=torch.int64, device=self.get_device()), 
                    torch.tensor(walk_src, dtype=torch.float, device=self.get_device()),
                    torch.tensor(self_node, dtype=torch.int64, device=self.get_device()))
            
    def forward(self, src, src_mask, walk, walk_src, self_node):
        
        src, src_mask = self.encoder(src, src_mask, walk, walk_src, self_node)
        src = self.decoder(src)
        
        return src
    
    def calculate_loss(self, this_batch):
        src, src_mask, trg, walk, walk_src, self_node = this_batch
        src = self.forward(src, src_mask, walk, walk_src, self_node)
        loss_func = nn.MSELoss()
        src = src.view(src.shape[0], OUT_DIM)
        trg = trg.view(trg.shape[0], OUT_DIM)
        mse = loss_func(src, trg)
        return torch.sqrt(mse)
    
    def run_step(self, this_batch):
        self.optimizer.zero_grad()
        loss = self.calculate_loss(this_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), CLIP)
        self.optimizer.step()
        return float(loss.cpu())
    
    def run_training(self):
        
        best_loss = float('inf')
        start_time = time.time()
        ready_to_stop = 0
        for epoch in range(MAX_EPOCHES):
            if ready_to_stop > PATIENCE:
                print('Terminating... No more improvement...')
                break
            self.train()
            batch_data = self.get_batch(self.train_)
            train_loss = 0
            batch_num = 0
            for this_batch in batch_data:
                batch_num += 1
                train_loss += self.run_step(this_batch)
            train_loss /= batch_num
            
            
            self.eval()
            batch_data = self.get_batch(self.val_)
            val_loss = 0
            batch_num = 0
            for this_batch in batch_data:
                batch_num += 1
                val_loss += self.calculate_loss(this_batch)
            val_loss /= batch_num
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if val_loss <= best_loss:
                best_loss = val_loss
                torch.save(self, self.model_path)
                ready_to_stop = 0
            else:
                ready_to_stop += 1
            
            print('Training...')
            print('Epoch: {} ---- Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
            print('Training loss: {}'.format(train_loss))
            print('Validating loss: {}'.format(val_loss))
            print('------------------------------------------')
            
            

    # del
    def get_test_loss(self, this_batch):
        src, src_mask, trg, walk, walk_src = this_batch
        src = self.forward(src, src_mask, walk, walk_src)
        loss_func = nn.MSELoss()
        src = torch.split(src, 1, dim=1)
        trg = torch.split(trg, 1, dim=1)
        mse = []
        for i in range(SEQ_LEN):
            mse.append(torch.sqrt(loss_func(src[i].squeeze(1), trg[i].squeeze(1))))
        return [_.cpu() for _ in mse]
        
    def run_testing(self):
        self = torch.load(self.model_path)
        start_time = time.time()
        self.eval()
        batch_data = self.get_batch(self.test_)
        test_loss = 0.0
        batch_num = 0
        for this_batch in batch_data:
            batch_num += 1
            test_loss += self.calculate_loss(this_batch)
            #mse_loss = self.get_test_loss(this_batch)
            #test_loss = [test_loss[i] + mse_loss[i] for i in range(SEQ_LEN)]
        #test_loss = [test_loss[i]/batch_num for i in range(SEQ_LEN)]
        test_loss /= batch_num
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print('Testing...')
        print('Time: {}m {}s'.format(epoch_mins, epoch_secs))
        print('Testing loss: {}'.format(test_loss))
        #for ts in range(SEQ_LEN):
        #    print('Testing loss on SEQ {}: {}'.format(ts,test_loss[ts]))
        print('-------------------------------------------------')
        
if __name__  == '__main__':
    
    data_path = 'motion/jump'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available():
        model = Sees_model('gpu')
        model.INIT(data_path)
        model.cuda()
        for c in model.children():
            c.cuda()
    else:
        model = Sees_model('cpu')
        model.INIT(data_path)
        model.cpu()
        for c in model.children():
            c.cpu()
       
    model.run_training()
    model.run_testing()
    print('Data: {}, Parameters: [{}, {}, {}, {}]'.format(data_path,N_LAYERS, N_HEADS, HID_DIM, MAX_EPOCHES))
