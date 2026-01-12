# YOLO-WL
A Lightweight and Efficient Framework for UAV Based Wildlife Detection

## 2. MLKSA mechanism
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
