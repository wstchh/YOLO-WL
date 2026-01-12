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

<div align="center">
    <img src="Fig. 4-SSA-PAN network schematic.png" width="600">
</div>

<div align="center">​ 
SSA-PAN network schematic
</div>	

```python
class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=None, fusion_type='learnable'):
        super(MultiScaleSpatialAttention, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [5, 7, 9, 11, 13]  
        
        self.kernel_sizes = kernel_sizes
        self.fusion_type = fusion_type
        
        # Multi-branch convolution: each branch processes the concatenated [avg, max] features
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=k, padding=padding, bias=False),
                    nn.BatchNorm2d(1)
                )
            )
        
        # Learnable fusion weights
        if fusion_type == 'learnable':
            self.scale_weights = nn.Parameter(torch.ones(len(kernel_sizes)), requires_grad=True)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Global channel compression
        avg_out = torch.mean(x, dim=1, keepdim=True)     # (b, 1, h, w)
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # (b, 1, h, w)
        attention_input = torch.cat([avg_out, max_out], dim=1)  # (b, 2, h, w)

        scale_outputs = []
        for i, branch in enumerate(self.branches):
            out = branch(attention_input)
            if self.fusion_type == 'learnable':
                out = self.scale_weights[i] * out
            scale_outputs.append(out)
        
        if self.fusion_type == 'learnable':
            attention_map = torch.sum(torch.stack(scale_outputs, dim=0), dim=0)
        else:
            attention_map = torch.mean(torch.stack(scale_outputs, dim=0), dim=0)  
        
        attention_map = self.sigmoid(attention_map)  # (b, 1, h, w)
        return x * attention_map

    def get_fusion_weights(self):
        if self.fusion_type == 'learnable':
            return F.softmax(self.scale_weights, dim=0).detach().cpu().numpy()
        else:
            return None


class SGF-2(nn.Module):
    def __init__(self, dimension=1, init_value=1.0):
        super(SGF-2, self).__init__()
        self.dim = dimension

        # Learnable weight parameters
        self.w = nn.Parameter(torch.full((2,), init_value, dtype=torch.float32), requires_grad=True)

        self.spatial_attention = MultiScaleSpatialAttention()


    def forward(self, x_list):
        assert len(x_list) == 2, "Input should be a list of 2 feature maps."
        # Weighted summation
        x0 = self.w[0] * x_list[0]
        x1 = self.w[1] * x_list[1]
        x_cat = torch.cat([x0 ,x1], dim=self.dim)

        # Spatial Attention enhancement
        fused = self.spatial_attention(x_cat)
        return fused


class SGF-3(nn.Module):
    def __init__(self, dimension=1, init_value=1.0):
        super(SGF-3, self).__init__()
        self.dim = dimension

        # Learnable weight parameters
        self.w = nn.Parameter(torch.full((3,), init_value, dtype=torch.float32), requires_grad=True)

        self.spatial_attention = MultiScaleSpatialAttention()


    def forward(self, x_list):
        assert len(x_list) == 3, "Input should be a list of 3 feature maps."
        # Weighted summation
        x0 = self.w[0] * x_list[0]
        x1 = self.w[1] * x_list[1]
        x2 = self.w[2] * x_list[2]
        x_cat = torch.cat([x0, x1, x2], dim=self.dim)

        # Spatial Attention enhancement
        fused = self.spatial_attention(x_cat)
        return fused

```
