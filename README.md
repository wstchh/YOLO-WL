# YOLO-WL

This is the official code implementation of YOLO-WL for the *Sensors* journal paper: **"YOLO-WL: A Lightweight and Efficient Framework for UAV-Based Wildlife Detection"**

<div align="center">
    <img src="Fig. 1-Network structure of YOLO-WL.png" width="600">
</div>

<div align="center">​ 
Network structure of YOLO-WL
</div>	

## 1-MSDDSC module
<div align="center">
    <img src="Fig. 2-The structure of C2f-MSDDSC module.png" width="600">
</div>

<div align="center">​ 
The structure of C2f-MSDDSC module
</div>	

```python
class MSDDSC(nn.Module):
    "Multi-Scale Dilated Depthwise Separable Convolution"
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3,3), e=0.5, dilation_rates=(1,3,5,7,9)):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)

        self.dilated_convs = nn.ModuleList()
        for rate in dilation_rates:
            self.dilated_convs.append(
                nn.Sequential(
                    nn.Conv2d(c_, c_, kernel_size=k[1], stride=1, padding=rate, dilation=rate, groups=c_, bias=False),
                    nn.Conv2d(c_, c2, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.SiLU()
                )
            )
        
        self.cv2 = Conv(5*c2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        x = self.cv1(x)

        features = []
        for conv in self.dilated_convs:
            features.append(conv(x))
        x_concat = torch.cat(features, dim=1)
        y = self.cv2(x_concat)
        
        if self.add:
            y = y + identity
        return y
    

class C2f_MSDDSC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  
        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.cv2 = Conv((2+n) * self.c, c2, 1)  
        self.m = nn.Sequential(*(MSDDSC(self.c, self.c, shortcut, g, k=(3,3), e=0.5) for _ in range(n)))
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```


## 2-MLKSA mechanism

<div align="center">
    <img src="Fig. 3-Illustration of MLKSA mechanism.png" width="600">
</div>

<div align="center">​ 
Illustration of MLKSA mechanism
</div>	


```python
class MLKSA(nn.Module):
    """
    Multi-scale Large Kernel Spatial Attention
    """
    def __init__(self, channels, kernel_sizes=[5, 7, 9, 11, 13]):
        super().__init__()
        self.conv_branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.conv_branches.append(
                nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=k, padding=padding, bias=False),
                    nn.BatchNorm2d(1),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.fusion = nn.Sequential(
            # 1x1 convolution for multi-scale feature fusion
            nn.Conv2d(len(kernel_sizes), len(kernel_sizes), kernel_size=1),  
            nn.BatchNorm2d(len(kernel_sizes)),
            nn.ReLU(inplace=True),
            # 3×3 convolution for further fusion
            nn.Conv2d(len(kernel_sizes), 1, kernel_size=3, padding=1),  
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            # Ultimately, this process generates a channel spatial attention map.
            nn.Conv2d(1, 1, kernel_size=1),  
            nn.Sigmoid()
        )
        
        nn.init.constant_(self.fusion[-2].weight, 0)
        nn.init.constant_(self.fusion[-2].bias, 0)
    
    def forward(self, x):
        # Compute the mean and maximum along the channel dimension.
        mean_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        concat_feat = torch.cat([mean_feat, max_feat], dim=1)
        
        # Multi-scale feature extraction
        scale_outputs = []
        for conv in self.conv_branches:
            scale_outputs.append(conv(concat_feat))
        
        # Fuse multi-scale features
        fused = torch.cat(scale_outputs, dim=1)
        weights = self.fusion(fused)
        return x * weights    
```

## 3-SSA-PAN network
