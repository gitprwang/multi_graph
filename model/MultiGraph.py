import torch
from torch import nn 

class TemporalSim(nn.Module):
    def __init__(self, in_dim, embed_dim, ksize=3):
        super(TemporalSim, self).__init__()
        self.conv = nn.Conv2d(in_dim, embed_dim, kernel_size=(ksize,1), padding='same')

    def forward(self, x):
        # x:B,T,N,d
        x = torch.permute(x, [0,3,1,2])
        x = self.conv(x)  # B,d,T,N
        x = torch.permute(x, [0,2,3,1]) # B,T,N,d
        adj_mx = torch.sigmoid(torch.matmul(x, x.transpose(2,3))) # B,T,N,N
        return adj_mx

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act=torch.relu):
        super(GCN, self).__init__()
        self.w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim))
        self.act = act

    def forward(self, x, g):
        # x : B,T,N,d
        # g : B,T,N,N   or B, N, N
        x = self.act(self.b + torch.matmul(x, self.w))
        x = torch.matmul(g, x)
        return x

class GraphBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act=torch.relu, norm=None, use_g=True, use_t=True):
        super(GraphBlock, self).__init__()
        self.use_g = use_g
        self.use_t = use_t
        if use_g:
            self.global_gcn = GCN(in_dim, out_dim, act=act)
        if use_t:
            self.temporal_gcn = GCN(in_dim, out_dim, act=act)
        if use_g and use_t:
            self.fc = nn.Linear(out_dim*2, out_dim)
        else:
            self.fc = nn.Linear(out_dim, out_dim)
        self.act = act
        if norm is None:
            self.norm = None
        else:
            self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, global_graph, temporal_graph):
        # x : B,T,N,d
        cat_list = []
        if self.use_g:
            g_x = self.global_gcn(x, global_graph)      # B,T,N,d
            cat_list.append(g_x)
        if self.use_t:
            t_x = self.temporal_gcn(x, temporal_graph)  # B,T,N,d
            cat_list.append(t_x)
        out = torch.cat(cat_list, dim=-1)
        out = self.act(self.fc(out))
        if self.norm is None:
            return out
        else:
            return self.norm(out)


class TemporalBlock(nn.Module):
    def __init__(self, in_dim, out_dim, in_len, out_len, act=torch.relu, use_diff=True):
        super(TemporalBlock, self).__init__()
        self.t_w = nn.Parameter(torch.randn(in_len, out_len))
        self.use_diff = use_diff
        if self.use_diff:
            self.d_w = nn.Parameter(torch.randn(2*in_dim, out_dim))
        else:
            self.d_w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.d_b = nn.Parameter(torch.zeros(out_dim))
        self.act = act 

    def forward(self, x):
        # x:B,T,N,d
        if self.use_diff:
            dif = x - torch.cat([x[:,0:1,:,:], x[:,0:-1,:,:]], dim=1)
            x = torch.cat([x, dif], dim=-1)
        x = torch.einsum('abcd,de->abce', x, self.d_w)
        x = x + self.d_b
        x = self.act(x)
        x = torch.einsum('abcd,be->aecd', x, self.t_w)
        return x


class MultiGraphBlock(nn.Module):
    def __init__(self, in_dim, out_dim, in_len, out_len, num_nodes, act=torch.relu, use_diff=True, use_g=True, use_t=True):
        super(MultiGraphBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_len = in_len 
        self.out_len = out_len
        # self.t1 = TemporalBlock(in_dim, out_dim, in_len, out_len, act=act)
        self.g = GraphBlock(in_dim, out_dim, act=act, use_g=use_g, use_t=use_t)
        self.t2 = TemporalBlock(out_dim, out_dim, in_len, out_len, act=act, use_diff=use_diff)
        self.ln = nn.LayerNorm([num_nodes, out_dim])
        self.align = nn.Linear(in_dim, out_dim) if in_dim!=out_dim else lambda x:x

    def forward(self, x, global_graph, temporal_graph):
        # x:B,T,N,d
        align = self.align(x)
        # x = self.t1(x)
        x = self.g(x, global_graph, temporal_graph)
        x = self.t2(x)
        x = x + align
        x = self.ln(x)
        return x

class Model(nn.Module):
    def __init__(self, in_dim, out_dim, in_len, out_len, num_nodes, embed_dim=10, ksize=3, layers=[64,128,64], use_diff=True, use_g=True, use_t=True):
        super().__init__()
        self.TemporalGraph = TemporalSim(in_dim, embed_dim=embed_dim, ksize=ksize)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.blocks = nn.ModuleList([])
        self.layers = [in_dim] + layers
        self.use_g = use_g
        self.use_t = use_t
        for i in range(len(self.layers)-1):
            if i==0:
                self.blocks.append(MultiGraphBlock(self.layers[i], self.layers[i+1], in_len, out_len, num_nodes, use_diff=use_diff, use_g=use_g, use_t=use_t))
            else:
                self.blocks.append(MultiGraphBlock(self.layers[i], self.layers[i+1], out_len, out_len, num_nodes, use_diff=use_diff, use_g=use_g, use_t=use_t))
        self.out_ln1 = nn.Linear(self.layers[-1], out_dim)
        self.out_ln2 = nn.Parameter(torch.randn(out_len, out_len))

    def forward(self, x):
        # x:B,T,N,d
        global_graph = torch.sigmoid(torch.matmul(self.node_embeddings, self.node_embeddings.T)) if self.use_g else None
        temporal_graph = self.TemporalGraph(x) if self.use_t else None

        for block in self.blocks:
            x = block(x, global_graph, temporal_graph)

        out = torch.relu(self.out_ln1(x))
        out = torch.einsum('abcd,be->aecd', out, self.out_ln2)
        return out







if __name__=='__main__':
    x = torch.randn([32, 12, 307, 1])
    model = Model(1,1,12,12,307,embed_dim=10,layers=[3])
    y = model(x)
    print(y.shape)

