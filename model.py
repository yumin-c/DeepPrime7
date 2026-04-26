# Model definition inspired from PrimeNet (2025), DeepPrime (2023).

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        return x * self.sigmoid(avg_out)

class Conv_Attention(nn.Module):
    def __init__(self, in_channels, kernel_size=5):
        super(Conv_Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()
        # self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        # return self.conv2(x) * self.sigmoid(self.conv1(x))
        return x * self.sigmoid(self.conv1(x)) # original PrimeNet implementation
        # return x * self.sigmoid(self.bn(self.conv1(x)))

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 1), stride=(1, 1)):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(1, 0))
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))       
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        return out + residual

class MultiScaleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv2d, self).__init__()
        self.conv1x2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2), padding=(0, 0))
        self.attn1x2 = Conv_Attention(out_channels)
        self.conv3x2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2), padding=(1, 0))
        self.attn3x2 = Conv_Attention(out_channels)
        self.conv9x2 = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 2), padding=(4, 0))
        self.attn9x2 = Conv_Attention(out_channels)
        conv_num = 3    
        self.attn_cat = ChannelAttention(conv_num * out_channels)
        
        self.adjust_channels = nn.Conv2d(conv_num * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out1x2 = self.conv1x2(x)
        out1x2 = self.attn1x2(out1x2)
        out3x2 = self.conv3x2(x)
        out3x2 = self.attn3x2(out3x2)
        out9x2 = self.conv9x2(x)
        out9x2 = self.attn9x2(out9x2)
        
        concat_out = torch.cat([out1x2, out3x2, out9x2], dim=1)
        concat_out = self.attn_cat(concat_out)
        return self.adjust_channels(concat_out)
    
class DeepPrime7(nn.Module):
    def __init__(
        self,
        conv1_out,
        conv2_out,
        conv3_out,
        dropout1,
        dropout2,
        dropout3,
        fc1_hidden,
        fc2_hidden,
        fc_dropout,
        num_heads=4,
        use_attention=True,
        use_additional_features=[]
    ):
        super(DeepPrime7, self).__init__()

        self.use_attention = use_attention
        self.use_additional_features = use_additional_features
        self.n_additional_features = len(use_additional_features)
        self.num_heads = num_heads

        self.conv1 = MultiScaleConv2d(4, conv1_out)
        self.conv2 = ResidualConvBlock(conv1_out, conv2_out)
        self.conv3 = ResidualConvBlock(conv2_out, conv3_out)

        self.norm1 = nn.LayerNorm([conv1_out, 128, 1])
        self.norm2 = nn.LayerNorm([conv2_out, 128, 1])
        self.norm3 = nn.LayerNorm([conv3_out, 64, 1])
        
        self.gelu = nn.GELU()

        self.attn0 = Conv_Attention(4) if use_attention else nn.Identity()
        self.attn1 = Conv_Attention(conv1_out) if use_attention else nn.Identity()
        self.attn2 = Conv_Attention(conv2_out) if use_attention else nn.Identity()
        self.attn3 = Conv_Attention(conv3_out) if use_attention else nn.Identity()
        self.attn4 = ChannelAttention(conv3_out) if use_attention else nn.Identity()

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout3 = nn.Dropout(dropout3)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        # Optional MLP for additional features
        if self.n_additional_features > 0:
            self.additional_mlp = nn.Sequential(
                nn.Linear(self.n_additional_features, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            fc1_input_dim = conv3_out * 32 + 32
        else:
            self.additional_mlp = None
            fc1_input_dim = conv3_out * 32

        self.fc1_shared = nn.Sequential(
            nn.Linear(fc1_input_dim, fc1_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout)
        )

        self.fc_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fc1_hidden, fc2_hidden),
                nn.ReLU(),
                nn.Dropout(fc_dropout),
                nn.Linear(fc2_hidden, 1)
            ) for _ in range(num_heads)
        ])

    def forward(self, x, additional_features=None):
        x = self.attn0(x) # Initial conv-attention
        
        x = self.conv1(x) # Multi-scale convolution
        x = self.norm1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.attn1(x) # Convolutional attention

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu(x)
        x = self.pool(x)
        x = self.dropout2(x)
        x = self.attn2(x) # Convolutional attention

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.gelu(x)
        x = self.pool(x)
        x = self.dropout3(x)
        x = self.attn3(x) # Convolutional attention
        x = self.attn4(x) # Channel attention

        x = x.view(x.size(0), -1)

        # Concatenate additional features if used
        if self.n_additional_features > 0 and additional_features is not None:
            feat = self.additional_mlp(additional_features)
            x = torch.cat([x, feat], dim=1)

        x = self.fc1_shared(x)

        # Apply each FC branch and collect outputs
        outputs = [branch(x) for branch in self.fc_branches]

        return tuple(outputs)  # Return as tuple of outputs

# Legacy models from PrimeNet (2025) for compatibility
class PrimeNet(nn.Module):
    def __init__(
        self,
        conv1_out,
        conv2_out,
        conv3_out,
        dropout1,
        dropout2,
        dropout3,
        fc1_hidden,
        fc2_hidden,
        fc_dropout,
        use_attention=True,
        use_additional_features=[]
    ):
        super(PrimeNet, self).__init__()

        self.use_attention = use_attention
        self.use_additional_features = use_additional_features
        self.n_additional_features = len(use_additional_features)

        self.conv1 = MultiScaleConv2d(4, conv1_out)
        self.conv2 = ResidualConvBlock(conv1_out, conv2_out)
        self.conv3 = ResidualConvBlock(conv2_out, conv3_out)

        self.norm1 = nn.LayerNorm([conv1_out, 128, 1])
        self.norm2 = nn.LayerNorm([conv2_out, 64, 1])
        self.norm3 = nn.LayerNorm([conv3_out, 32, 1])

        self.attn0 = Conv_Attention(4) if use_attention else nn.Identity()
        self.attn1 = Conv_Attention(conv1_out) if use_attention else nn.Identity()
        self.attn2 = Conv_Attention(conv2_out) if use_attention else nn.Identity()
        self.attn3 = Conv_Attention(conv3_out) if use_attention else nn.Identity()
        self.attn4 = ChannelAttention(conv3_out) if use_attention else nn.Identity()

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout3 = nn.Dropout(dropout3)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        # Optional MLP for additional features
        if self.n_additional_features > 0:
            self.additional_mlp = nn.Sequential(
                nn.Linear(self.n_additional_features, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            fc1_input_dim = conv3_out * 32 + 32
        else:
            self.additional_mlp = None
            fc1_input_dim = conv3_out * 32

        self.fc1_shared = nn.Sequential(
            nn.Linear(fc1_input_dim, fc1_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout)
        )

        self.fc1_branch1 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )
        self.fc1_branch2 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )
        self.fc1_branch3 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )

    def forward(self, x, additional_features=None):
        x = self.attn0(x)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.attn1(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.attn2(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.norm3(x)
        x = self.dropout3(x)
        x = self.attn3(x)
        x = self.attn4(x)

        x = x.view(x.size(0), -1)

        # Concatenate additional features if used
        if self.n_additional_features > 0 and additional_features is not None:
            feat = self.additional_mlp(additional_features)
            x = torch.cat([x, feat], dim=1)

        x = self.fc1_shared(x)

        out1 = self.fc1_branch1(x)
        out2 = self.fc1_branch2(x)
        out3 = self.fc1_branch3(x)

        return torch.cat([out1, out2, out3], dim=1)

class PrimeNet_6(nn.Module):
    def __init__(
        self,
        conv1_out,
        conv2_out,
        conv3_out,
        dropout1,
        dropout2,
        dropout3,
        fc1_hidden,
        fc2_hidden,
        fc_dropout,
        use_attention=True,
        use_additional_features=[]
    ):
        super(PrimeNet_6, self).__init__()

        self.use_attention = use_attention
        self.use_additional_features = use_additional_features
        self.n_additional_features = len(use_additional_features)

        self.conv1 = MultiScaleConv2d(4, conv1_out)
        self.conv2 = ResidualConvBlock(conv1_out, conv2_out)
        self.conv3 = ResidualConvBlock(conv2_out, conv3_out)

        self.norm1 = nn.LayerNorm([conv1_out, 128, 1])
        self.norm2 = nn.LayerNorm([conv2_out, 64, 1])
        self.norm3 = nn.LayerNorm([conv3_out, 32, 1])

        self.attn0 = Conv_Attention(4) if use_attention else nn.Identity()
        self.attn1 = Conv_Attention(conv1_out) if use_attention else nn.Identity()
        self.attn2 = Conv_Attention(conv2_out) if use_attention else nn.Identity()
        self.attn3 = Conv_Attention(conv3_out) if use_attention else nn.Identity()
        self.attn4 = ChannelAttention(conv3_out) if use_attention else nn.Identity()

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout3 = nn.Dropout(dropout3)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        # Optional MLP for 11 additional features
        if self.n_additional_features > 0:
            self.additional_mlp = nn.Sequential(
                nn.Linear(self.n_additional_features, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            fc1_input_dim = conv3_out * 32 + 32
        else:
            self.additional_mlp = None
            fc1_input_dim = conv3_out * 32

        self.fc1_shared = nn.Sequential(
            nn.Linear(fc1_input_dim, fc1_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout)
        )

        self.fc1_branch1 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )
        self.fc1_branch2 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )
        self.fc1_branch3 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )
        self.fc1_branch4 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )
        self.fc1_branch5 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )
        self.fc1_branch6 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )

    def forward(self, x, additional_features=None):
        x = self.attn0(x)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.attn1(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.attn2(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.norm3(x)
        x = self.dropout3(x)
        x = self.attn3(x)
        x = self.attn4(x)

        x = x.view(x.size(0), -1)

        # Concatenate additional features if used
        if self.n_additional_features > 0 and additional_features is not None:
            feat = self.additional_mlp(additional_features[:, self.use_additional_features])
            x = torch.cat([x, feat], dim=1)

        x = self.fc1_shared(x)

        out1 = self.fc1_branch1(x)
        out2 = self.fc1_branch2(x)
        out3 = self.fc1_branch3(x)
        out4 = self.fc1_branch4(x)
        out5 = self.fc1_branch5(x)
        out6 = self.fc1_branch6(x)

        return torch.cat([out1, out2, out3, out4, out5, out6], dim=1)

class PrimeNet_4(nn.Module):
    def __init__(
        self,
        conv1_out,
        conv2_out,
        conv3_out,
        dropout1,
        dropout2,
        dropout3,
        fc1_hidden,
        fc2_hidden,
        fc_dropout,
        use_attention=True,
        use_additional_features=[]
    ):
        super(PrimeNet_4, self).__init__()

        self.use_attention = use_attention
        self.use_additional_features = use_additional_features
        self.n_additional_features = len(use_additional_features)

        self.conv1 = MultiScaleConv2d(4, conv1_out)
        self.conv2 = ResidualConvBlock(conv1_out, conv2_out)
        self.conv3 = ResidualConvBlock(conv2_out, conv3_out)

        self.norm1 = nn.LayerNorm([conv1_out, 128, 1])
        self.norm2 = nn.LayerNorm([conv2_out, 64, 1])
        self.norm3 = nn.LayerNorm([conv3_out, 32, 1])

        self.attn0 = Conv_Attention(4) if use_attention else nn.Identity()
        self.attn1 = Conv_Attention(conv1_out) if use_attention else nn.Identity()
        self.attn2 = Conv_Attention(conv2_out) if use_attention else nn.Identity()
        self.attn3 = Conv_Attention(conv3_out) if use_attention else nn.Identity()
        self.attn4 = ChannelAttention(conv3_out) if use_attention else nn.Identity()

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout3 = nn.Dropout(dropout3)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        # Optional MLP for 11 additional features
        if self.n_additional_features > 0:
            self.additional_mlp = nn.Sequential(
                nn.Linear(self.n_additional_features, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            fc1_input_dim = conv3_out * 32 + 32
        else:
            self.additional_mlp = None
            fc1_input_dim = conv3_out * 32

        self.fc1_shared = nn.Sequential(
            nn.Linear(fc1_input_dim, fc1_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout)
        )

        self.fc1_branch1 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )
        self.fc1_branch2 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )
        self.fc1_branch3 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )
        self.fc1_branch4 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc2_hidden, 1)
        )

    def forward(self, x, additional_features=None):
        x = self.attn0(x)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.attn1(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.attn2(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.norm3(x)
        x = self.dropout3(x)
        x = self.attn3(x)
        x = self.attn4(x)

        x = x.view(x.size(0), -1)

        # Concatenate additional features if used
        if self.n_additional_features > 0 and additional_features is not None:
            feat = self.additional_mlp(additional_features[:, self.use_additional_features])
            x = torch.cat([x, feat], dim=1)

        x = self.fc1_shared(x)

        out1 = self.fc1_branch1(x)
        out2 = self.fc1_branch2(x)
        out3 = self.fc1_branch3(x)
        out4 = self.fc1_branch4(x)

        return torch.cat([out1, out2, out3, out4], dim=1)