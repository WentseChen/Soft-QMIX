import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Mixer(nn.Module):
    def __init__(self, args, abs=True, scheme=None):
        super(Mixer, self).__init__()

        self.args = args
        self.abs = abs
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_state_dim = int(np.prod(args.state_shape)) #+ self.n_agents * 11
        self.state_dim = int(np.prod(args.state_shape))
        
        # Qmix's mixer
        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim))
        
        self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, 1))
        
        # order preserving transformation
        self.hyper_w3 = nn.Sequential(
            nn.Linear(self.input_state_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim//2)
        )
        self.hyper_b3 = nn.Sequential(nn.Linear(self.input_state_dim, self.n_agents*self.embed_dim//2))
        self.hyper_w4 = nn.Sequential(
            nn.Linear(self.input_state_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents*self.embed_dim//2)
        )
        self.hyper_b4 = nn.Sequential(
            nn.Linear(self.input_state_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents)
        )
        
        # linear transformation
        self.hyper_w5 = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents)
        )
        self.hyper_b5 = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents)
        )
        
    def func_f(self, qvals, states, t_env, death_mask=None):
        
        qval_shape = qvals.shape
        states = states.reshape(-1, self.state_dim)
        
        if qval_shape[-2] == self.n_agents:
            self.dim_idx = -3
            qvals = qvals.reshape(-1, self.n_agents, qvals.shape[-1])
            w = self.hyper_w5(states).view(-1, self.n_agents, 1)
            b = self.hyper_b5(states).view(-1, self.n_agents, 1)
        if qval_shape[-1] == self.n_agents:
            self.dim_idx = -2
            qvals = qvals.reshape(-1, self.n_agents)
            w = self.hyper_w5(states).view(-1, self.n_agents)
            b = self.hyper_b5(states).view(-1, self.n_agents)
            
        if self.abs:
            w = w.abs()
        
        y = qvals * w + b 
        
        return y.reshape(qval_shape)
    
    def func_g(self, qvals, states, t_env, death_mask=None):
        
        qval_shape = qvals.shape
        states = states.reshape(-1, self.state_dim)
        
        qvals = qvals.reshape(-1, 1, self.n_agents, qval_shape[-1])
        w1 = self.hyper_w3(states).view(-1, self.embed_dim//2, self.n_agents, 1)
        b1 = self.hyper_b3(states).view(-1, self.embed_dim//2, self.n_agents, 1)
        w2 = self.hyper_w4(states).view(-1, self.embed_dim//2, self.n_agents, 1)
        b2 = self.hyper_b4(states).view(-1, 1, self.n_agents, 1)
        
        if self.abs:
            w1 = w1.abs()
            w2 = w2.abs()
        
        y = F.elu(qvals * w1 + b1)
        y = (y * w2).sum(dim=-3, keepdim=True) + b2
        y = y + qvals
        
        return y.reshape(qval_shape)

    def forward(self, qvals, states, death_mask):
        
        # reshape
        b, t, _ = qvals.size()
        
        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        
        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1) # b * t, emb, 1
        b2= self.hyper_b2(states).view(-1, 1, 1)
        
        if self.abs:
            w1 = w1.abs()
            w2 = w2.abs()
        
        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1) # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2 # b * t, 1, 1
        
        y = qvals.sum(-1)
        
        return y.view(b, t, -1)
    
