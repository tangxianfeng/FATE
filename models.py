import torch
import torch.nn as nn
from torch.nn import init, Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GCNConv, DenseGCNConv
from torch_geometric.nn.inits import glorot, zeros

from ipdb import set_trace



# based on torch_geometric.nn.GCNConv

class GCNTensor(nn.Module):
    # gcn that separate the operation on different features
    def __init__(self, in_channels, dim_per_channel, out_channels, improved=False, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(dim_per_channel, self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(self.in_channels, out_channels))
        else:
            self.register_parameter('bias', None)

        self.user_attn_fn = nn.Linear(in_channels * out_channels, 1).cuda()

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj, add_loop=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x # x: batch_size x num_node x in_channels
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        x = x.unsqueeze(-1) if x.dim() == 3 else x # x: batch_size x num_node x in_channels x 1
        out = torch.einsum('bnci,ick->bnck', x, self.weight) # x: batch_size x num_node x in_channels x out_channels
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.einsum('bni, bick -> bnck', adj, out)

        if self.bias is not None:
            out = out + self.bias # bias: c x k

        attn = torch.tanh(self.user_attn_fn(out.reshape((out.shape[0], out.shape[1], -1))))
        attn = torch.exp(attn)
        attn = attn/torch.sum(attn, dim=1, keepdim=True) # batch_size x num_node x 1
        attn = attn.squeeze(-1) # batch_size x num_node

        return out, attn # return user attention

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# based on torch_geometric.nn.GCNConv

class GCNTensorInteraction(nn.Module):
    # gcn that separate the operation on different features
    def __init__(self, in_channels, dim_per_channel, out_channels, interaction_ftr_dim, improved=False, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, dim_per_channel, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(self.in_channels, out_channels))
        else:
            self.register_parameter('bias', None)

        self.user_attn_fn = nn.Linear(in_channels * out_channels + out_channels, 1).cuda()

        self.inter_fn = nn.Linear(interaction_ftr_dim, out_channels).cuda()

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj, interaction, add_loop=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x # x: batch_size x num_node x in_channels
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        interaction = interaction.unsqueeze(0) if interaction.dim() == 2 else interaction
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        x = x.unsqueeze(-1) if x.dim() == 3 else x # x: batch_size x num_node x in_channels x 1
        out = torch.einsum('bnci,cik->bnck', x, self.weight) # x: batch_size x num_node x in_channels x out_channels
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.einsum('bni, bick -> bnck', adj, out)

        if self.bias is not None:
            out = out + self.bias # bias: c x k

        rep_ints = self.inter_fn(interaction) # batch_size x num_node x hidden dim

        reps = out.reshape((out.shape[0], out.shape[1], -1))[:,1:,:] # exclude self
        reps = torch.cat([rep_ints, reps], dim = -1)

        attn = torch.tanh(self.user_attn_fn(reps))
        attn = torch.exp(attn)
        attn = attn/torch.sum(attn, dim=1, keepdim=True) # batch_size x num_node x 1
        attn = attn.squeeze(-1) # batch_size x num_node

        return out, attn # return user attention

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




# alpha: temporal attention
# beta: action attention

# adopted from https://github.com/KurochkinAlexey/IMV_LSTM
class IMVTensorLSTM(nn.Module):

    def __init__(self, input_dim, output_dim, n_units, n_units_in, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(n_units_in, input_dim, n_units)*init_std)
        self.U_i = nn.Parameter(torch.randn(n_units_in, input_dim, n_units)*init_std)
        self.U_f = nn.Parameter(torch.randn(n_units_in, input_dim, n_units)*init_std)
        self.U_o = nn.Parameter(torch.randn(n_units_in, input_dim, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(n_units, input_dim, n_units)*init_std)
        self.W_i = nn.Parameter(torch.randn(n_units, input_dim, n_units)*init_std)
        self.W_f = nn.Parameter(torch.randn(n_units, input_dim, n_units)*init_std)
        self.W_o = nn.Parameter(torch.randn(n_units, input_dim, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(n_units, input_dim, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim

    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        outputs = []
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,jik->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bik", x[:,t,...], self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,jik->bik", h_tilda_t, self.W_i) + \
                                torch.einsum("bij,jik->bik", x[:,t,...], self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,jik->bik", h_tilda_t, self.W_f) + \
                                torch.einsum("bij,jik->bik", x[:,t,...], self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,jik->bik", h_tilda_t, self.W_o) + \
                                torch.einsum("bij,jik->bik", x[:,t,...], self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t*f_tilda_t + i_tilda_t*j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t*torch.tanh(c_tilda_t))
            outputs += [h_tilda_t,]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,jik->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)
        
        return mean, alphas, betas

class FATE(nn.Module):
    def __init__(self, in_dim, hidden_dim, x_num_day):
        super().__init__()
        self.fc_weight = Parameter(torch.Tensor(in_dim, 1, hidden_dim))
        self.fc_bias = Parameter(torch.Tensor(in_dim, hidden_dim))
        glorot(self.fc_weight)
        zeros(self.fc_bias)
        self.conv1s = [GCNTensorInteraction(in_dim, 1, hidden_dim, interaction_ftr_dim = 3).cuda() for _ in range(x_num_day)]
        self.conv2s = [GCNTensorInteraction(in_dim, hidden_dim, hidden_dim, interaction_ftr_dim = 3).cuda() for _ in range(x_num_day)]

        # add more if necessary
        self.rnn = IMVTensorLSTM(in_dim, 1, hidden_dim, hidden_dim *2).cuda()
        self.x_num_day = x_num_day

    def forward(self, As, Xs, Is):
        batch_hidden = []
        user_attn = []
        for ix, A in enumerate(As):
            X = Xs[ix]
            I = Is[ix] # num_days x num_friend x 3
            self_ftr = X[:,0,:].unsqueeze(-1)
            X_emb = torch.einsum('tij,ijo->tio', self_ftr, self.fc_weight) + self.fc_bias # num_days x num_var x hidden_dim
            H = []
            uat = []
            for i in range(self.x_num_day):
                node_emb, _ = self.conv1s[i](X[i], A, I[i])
                node_emb = F.elu(node_emb)
                node_emb2, fnd_attn = self.conv2s[i](node_emb, A, I[i])
                uat.append(fnd_attn)
                # 1 x num_node x in_channels x out_channels

                node_emb2 = node_emb2[:, 1:, :, :] # remove self representation
                weighted_node_emb = torch.einsum('bijk,bi->bijk', F.elu(node_emb2), fnd_attn)
                H.append(weighted_node_emb)

            avg_node_emb = torch.cat(H) # num_days x node_num x num_var x hidden_dim
            avg_node_emb = torch.mean(avg_node_emb, dim = 1, keepdim=False) # num_days x num_var x hidden_dim

            H_rnn = torch.cat([X_emb, avg_node_emb], dim=-1) # num_days x num_var x (hidden_dim * 2)
            batch_hidden.append(H_rnn)

            uat = torch.cat(uat) # num_days x num_users
            user_attn.append(uat)

        batch_rnn_in = torch.stack(batch_hidden) # batch_size x num_days x num_var x (hidden_dim * 2)
        pred, temporal_attn, action_attn = self.rnn(batch_rnn_in)
        
        return pred.squeeze(-1), user_attn, temporal_attn.squeeze(-1), action_attn.squeeze(-1)
