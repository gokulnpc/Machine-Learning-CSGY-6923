# models/edsr.py
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias))
            if i == 0:
                m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=4):
        super(EDSR, self).__init__()
        kernel_size = 3
        self.scale = scale
        
        # Define head module
        m_head = [nn.Conv2d(1, n_feats, kernel_size, padding=(kernel_size//2))]

        # Define body module
        m_body = [
            ResBlock(n_feats, kernel_size) \
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)))

        # Define tail module
        m_tail = [
            nn.Conv2d(n_feats, 1, kernel_size, padding=(kernel_size//2))
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x