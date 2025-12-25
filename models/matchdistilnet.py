import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchvision.models import (
    efficientnet_b5, EfficientNet_B5_Weights,
    swin_t, Swin_T_Weights
)
from swca_helper import (
    MultiHeadCrossWindowAttention, 
    positional_encoding
)

class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_planes, out_planes=256, memory=96, sync_bn=False, dilations=(12, 24, 36)):
        super(ASPP, self).__init__()

        norm_layer = nn.GroupNorm(num_groups=8, num_channels=memory)
        act = nn.GELU()
        self.dilations = nn.ModuleList()
        self.conv1 = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Conv2d(in_planes, memory, kernel_size=1, padding=0, dilation=1, bias=False),
                        norm_layer,
                        act
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_planes, memory, kernel_size=1, padding=0, dilation=1, bias=False),
                        norm_layer,
                        act
                    )
        
        for i in range(len(dilations)):
            self.dilations.append(
                nn.Sequential(
                    nn.Conv2d(in_planes, memory, kernel_size=3, padding=dilations[i], dilation=dilations[i], bias=False),
                    norm_layer,
                    act
                )
            )
        
        self.out_aspp = nn.Sequential(
            nn.Conv2d(memory*(len(dilations) + 2), out_planes, kernel_size=1, padding="same", dilation=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = self.conv1(x)
        feat1 = F.interpolate(feat1, size=[h, w], mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        
        feats = [feat1, feat2]
        for layer in self.dilations:
            feats.append(layer(x))
        
        aspp_out = torch.cat(feats, 1)
        
        return self.out_aspp(aspp_out)

class ConvMatcher(nn.Module):
    """Some Information about ConvMatcher"""
    def __init__(self, input_channels, output_channels, dilations):
        super(ConvMatcher, self).__init__()
        window_size = 7
        num_heads = 8
        qkv_bias = True
        drop_prob = 0.1
        device = torch.device('cuda')
        self.act = nn.GELU()
        
        self.conv_init = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=input_channels, stride=1, kernel_size=1, padding='same'),
                nn.GroupNorm(num_groups=8, num_channels=input_channels),
                self.act, 
            )
        
        self.attention = nn.ModuleList([
            MultiHeadCrossWindowAttention(
                    skip_channels=input_channels, 
                    cyclic_shift=False, 
                    window_size=window_size, 
                    num_heads=num_heads, 
                    qkv_bias=qkv_bias, 
                    attn_drop_prob=drop_prob, 
                    lin_drop_prob=drop_prob, 
                    device=device
                ),
            MultiHeadCrossWindowAttention(
                    skip_channels=input_channels, 
                    cyclic_shift=True, 
                    window_size=window_size, 
                    num_heads=num_heads, 
                    qkv_bias=qkv_bias, 
                    attn_drop_prob=drop_prob, 
                    lin_drop_prob=drop_prob, 
                    device=device
                ),
        ])
        self.mlp_norm = nn.ModuleList((
            nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=input_channels, stride=1, kernel_size=1, padding='same'),
                nn.GroupNorm(num_groups=8, num_channels=input_channels),
                self.act, 
            ),
            
            nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=input_channels, stride=1, kernel_size=1, padding='same'),
                nn.GroupNorm(num_groups=8, num_channels=input_channels),
                self.act,
            ),
        ))
        
        self.conv_matcher = ASPP(in_planes=input_channels, out_planes=output_channels, dilations=dilations)
        
        
    def forward(self, x):
        attn_weights = []
        x_normal, _ = self.attention[0](x, x)
        x_mlp1 = self.mlp_norm[0](x_normal) + x_normal
        attn_weights.append(_)
        
        x_shifted, _ = self.attention[1](x, x_mlp1)
        x_mlp2 = self.mlp_norm[1](x_shifted) + x_shifted
        attn_weights.append(_)
        
        if x_mlp2.shape != x.shape:
            x_mlp2 = F.interpolate(x_mlp2, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)
            
        out = self.conv_matcher(x_mlp2)
        
        return out, attn_weights

class EncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""

    def __init__(self, backbone_name, freeze=False):
        super(EncoderBlock, self).__init__()
        self.cnn = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1).features
        self.transformer = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).features[:6]
        self.act = nn.GELU()
        
        self.tf_conv1 = nn.Conv2d(in_channels=96, out_channels=40, kernel_size=1, stride=1, padding='same', bias=False)
        self.tf_conv2 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding='same', bias=False)
        self.tf_conv3 = nn.Conv2d(in_channels=384, out_channels=176, kernel_size=1, stride=1, padding='same', bias=False)
        
        self.conv_stabilizer1 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=1, stride=1, padding='same', bias=False)
        self.conv_stabilizer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding='same', bias=False)
        self.conv_stabilizer3 = nn.Conv2d(in_channels=176, out_channels=176, kernel_size=1, stride=1, padding='same', bias=False)
        
        self.conv_matcher1 = ConvMatcher(40+40, 40, dilations=(3, 5))
        self.conv_matcher2 = ConvMatcher(64+64, 64, dilations=(3, 5, 7))
        self.conv_matcher3 = ConvMatcher(176+176, 176, dilations=(7, 12, 24))
        
        self.train_purpose = [
            self.transformer,
            self.tf_conv1, self.tf_conv2, self.tf_conv3, 
            self.conv_matcher1, self.conv_matcher2, self.conv_matcher3
        ]
        self.param_train_enabled = False
        self.param_test_enabled = False
        
    def forward(self, x, is_train):
        features_cnn = [x]
        for layer1 in self.cnn: 
            pred = layer1(features_cnn[-1])
            features_cnn.append(pred)
            
        eff1, eff2, eff3, eff4, eff5 = features_cnn[2], features_cnn[3], features_cnn[4], features_cnn[6], features_cnn[9]
        
        if is_train:
            
            if self.param_train_enabled == False:
                self.enable_params(True)
                self.param_train_enabled = True
                self.param_test_enabled = False
            
            eff2 = self.conv_stabilizer1(eff2)
            eff3 = self.conv_stabilizer2(eff3)
            eff4 = self.conv_stabilizer3(eff4)
            
            features_transformer = [x]
            for layer2 in self.transformer:
                pred = layer2(features_transformer[-1])
                features_transformer.append(pred)
            
            tf1, tf2, tf3 = features_transformer[2], features_transformer[4], features_transformer[6]
        
            tf1 = tf1.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
            tf2 = tf2.permute(0, 3, 1, 2)
            tf3 = tf3.permute(0, 3, 1, 2)
            
            # Stabilizing 
            tf1 = self.tf_conv1(tf1)
            tf2 = self.tf_conv2(tf2)
            tf3 = self.tf_conv3(tf3)
            
            features = [ 
                eff1, 
                self.conv_matcher1(self.stacking(eff2, tf1)), 
                self.conv_matcher2(self.stacking(eff3, tf2)), 
                self.conv_matcher3(self.stacking(eff4, tf3)), 
                eff5
            ]
            
            # features = [ 
            #     eff1, 
            #     self.conv_matcher1(torch.cat([eff2, tf1], dim=1)), # self.stacking(eff2, tf1) 
            #     self.conv_matcher2(torch.cat([eff3, tf2], dim=1)),  # self.stacking(eff3, tf2)
            #     self.conv_matcher3(torch.cat([eff4, tf3], dim=1)),  # self.stacking(eff4, tf3)
            #     eff5
            # ]
        
        else: 
            
            if self.param_test_enabled == False:
                self.enable_params(False)
                self.param_test_enabled = True
                self.param_train_enabled = False
                
            
            features = [ 
                eff1, 
                eff2, 
                eff3, 
                eff4, 
                eff5
            ]

        return features
    
    def stacking(self, feat1, feat2): 
        # feat1 and feat2 : B, C, H, W 
        B, C, H, W = feat1.shape
        group_cat = torch.stack([feat1, feat2], dim=2)    # (B, C, 2, H, W)
        group_cat = group_cat.view(B, C*2, H, W)  # Interleave per channel
        
        return group_cat
    
    def enable_params(self, condition): 
        for module in self.train_purpose:
            for p in module.parameters():
                p.requires_grad = condition
        

class DecoderBLock(nn.Module):
    """Some Information about DecoderBLock"""

    def __init__(self, x_channels, skip_channels, desired_channels, layer, window_size, num_heads, 
                    qkv_bias, attn_drop_prob, lin_drop_prob, device, dilations=(3, 5, 7)):
        super(DecoderBLock, self).__init__()
        self.act  = nn.GELU()
        self.attentions = nn.ModuleList()
        self.convMLP = nn.ModuleList()
        self.conv_stabilizer1 = nn.ModuleList()
        self.conv_stabilizer2 = nn.ModuleList()
        self.desired_channels = desired_channels
        self.layer = layer
        self.device = device
        
        for _ in range(layer // 2):
            self.attentions.append(
                nn.ModuleList([
                        MultiHeadCrossWindowAttention(
                                skip_channels=desired_channels, cyclic_shift=False, window_size=window_size, num_heads=num_heads, 
                                qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, lin_drop_prob=lin_drop_prob, device=device
                            ),
                        MultiHeadCrossWindowAttention(
                                skip_channels=desired_channels, cyclic_shift=True, window_size=window_size, num_heads=num_heads, 
                                qkv_bias=qkv_bias, attn_drop_prob=attn_drop_prob, lin_drop_prob=lin_drop_prob, device=device
                            ),
                    ])
                )
            
            self.conv_stabilizer1.append(
                nn.Conv2d(in_channels=desired_channels, out_channels=desired_channels, stride=1, kernel_size=1, padding='same')
            )
            self.conv_stabilizer2.append(
                nn.Conv2d(in_channels=desired_channels, out_channels=desired_channels, stride=1, kernel_size=1, padding='same')
            )
            
            self.convMLP.append(
                ASPP(desired_channels, out_planes=desired_channels, memory=desired_channels//2, sync_bn=False, dilations=dilations)
            )
        
        self.x_feed2msa = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=desired_channels, stride=1, kernel_size=1, padding='same'), 
            nn.GroupNorm(num_groups=8, num_channels=desired_channels),
            self.act,
        )
        
        self.skip_feed2msa = nn.Sequential(
            nn.Conv2d(in_channels=skip_channels, out_channels=desired_channels, stride=1, kernel_size=1, padding='same'),
            nn.GroupNorm(num_groups=8, num_channels=desired_channels),
            self.act,
        )
        self.mlp_out = nn.Sequential(
            nn.Conv2d(in_channels=desired_channels*2, out_channels=desired_channels, stride=1, kernel_size=1, padding='same'),
            nn.GroupNorm(num_groups=8, num_channels=desired_channels),
            self.act,
        )
        self.pe = None
        

    def forward(self, skip, x):
        """
        Args: 
            skip    : B, C, H, W
            x       : B, 2C, H/2, W/2
        """
        x_msa = self.x_feed2msa(x)              # B, D, H, W
        skip_prop = self.skip_feed2msa(skip)    # B, D, H, W
        
        if x_msa.shape != skip_prop.shape:
            x_msa = F.interpolate(x_msa, size=[skip_prop.shape[2], skip_prop.shape[3]], mode='bilinear', align_corners=True)
        
        
        attn_weight = []
        
        for idx in range(self.layer // 2):
            x_msa_basic, attn_weight_normal = self.attentions[idx][0](skip_prop, x_msa)
            x_msa_basic = self.conv_stabilizer1[idx](x_msa_basic) + x_msa_basic
            x_msa_shifted, attn_weight_shifted = self.attentions[idx][1](skip_prop, x_msa_basic)
            x_msa_shifted = self.conv_stabilizer2[idx](x_msa_shifted) + x_msa_shifted
            
            x_msa_mlp = self.convMLP[idx](x_msa_shifted) + x_msa_shifted
            attn_weight.append(attn_weight_normal)
            attn_weight.append(attn_weight_shifted)
            
            x_msa = x_msa_mlp.clone()
            
        # Shape after SWCA can be differnt due to auto padding in window attention
        if x_msa.shape != skip_prop.shape:
            x_msa = F.interpolate(x_msa, size=[skip_prop.shape[2], skip_prop.shape[3]], mode='bilinear', align_corners=True)
        
        B, C, H, W = x_msa.shape
        # group_cat = torch.stack([x_msa, skip_prop], dim=2)    # (B, C, 2, H, W)
        # group_cat = group_cat.view(B, C*2, H, W)  # Interleave per channel
        group_cat = torch.cat([x_msa, skip_prop], dim=1)  # B, 2C, H, W
        out = self.mlp_out(group_cat)
        
        return out, attn_weight


class Head(nn.Module):
    """Some Information about Head"""
    def __init__(self, input_channel, n_bins, min_val, max_val, drop_prob, device):
        super(Head, self).__init__()
        self.min_val = min_val 
        self.max_val = max_val
        
        self.drop_prob = drop_prob
        self.new_features = 128 
        self.device = device
        
        self.act = nn.GELU()
        
        self.head1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=self.new_features, kernel_size=1, stride=1, padding="same"),
            self.act,
        ).to(device)
        
        self.final_head = nn.Sequential(
            nn.Conv2d(in_channels=self.new_features, out_channels=n_bins, kernel_size=1, stride=1, padding="same"),
            nn.Softmax(dim=1)
        ).to(device)
        
        self.patcher_size = 8
        self.img_patcher = nn.Sequential(
            nn.Conv2d(self.new_features, out_channels=self.new_features, kernel_size=self.patcher_size, stride=self.patcher_size//2),
            nn.GELU(),
            nn.Conv2d(self.new_features, out_channels=self.new_features, kernel_size=self.patcher_size, stride=self.patcher_size//2),
            nn.GELU(),
        )
        self.pe = None
        
        self.token_proj = nn.Sequential(
            nn.Linear(self.new_features, self.new_features, bias=False), 
            nn.GELU(),
            nn.Linear(self.new_features, n_bins, bias=False)
        )
        self.linear_reg = nn.Sequential(
            nn.Linear(n_bins, n_bins, bias=False),
            nn.ReLU()
        )
        
    def forward(self, x):
        head = self.head1(x)    # B, C, H, W
        
        # Preparing the size of the head
        head_b, head_C, head_H, head_W = head.shape
        target_h = math.ceil(head_H / self.patcher_size) * self.patcher_size
        target_w = math.ceil(head_W / self.patcher_size) * self.patcher_size
        head = F.interpolate(head, size=(target_h, target_w), mode='bilinear', align_corners=True)
        
        # Preparing input 
        shrinked_head = self.img_patcher(head)
        shrinked_head = shrinked_head.flatten(start_dim=2)  # B, C, T
        
        b, c, t = shrinked_head.shape
        if self.pe is None or self.pe.size(1) != t: 
            self.pe = positional_encoding(t, c, self.device, requires_grad=False).to(shrinked_head.device)   # 1, T, C
        
        # Original Information
        bin_preds = shrinked_head.permute(0, 2, 1) + self.pe    # B, T, C 
        
        # Taking features
        token_regression = self.token_proj(bin_preds)           # B, T, n_bins
        bin_widths_normed = self.linear_reg(token_regression.mean(dim=1))   # B, n_bins
        
        head2 = self.final_head(head)   # B, n_bins, H, W
        
        # Bin Calculation
        bin_widths_normed = bin_widths_normed + 0.1
        bin_widths_normed = bin_widths_normed / bin_widths_normed.sum(dim=1, keepdim=True)
        
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_widths = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_widths[:, :-1] + bin_widths[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)
        
        main_pred = torch.sum(head2 * centers, dim=1, keepdim=True)

        return bin_widths, main_pred


class MatchDistillNet(nn.Module):
    """Some Information about UNetResNet"""

    def __init__(self, 
                    device, 
                    backbone_name, 
                    n_bins:int, 
                    window_sizes:int, 
                    layers:int, 
                    qkv_bias:bool=True, 
                    drop_prob:float=0.15, 
                    min_val:float=0.001, 
                    max_val:float=10.0
                ):
        super(MatchDistillNet, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.encoder = EncoderBlock(self.backbone_name, freeze=False)#.to(device)
        dec_heads = 4
        self.min_val = min_val  # Minimum value for depth
        self.max_val = max_val  # Maximum value for depth
        
        # EfficientNet-B1, B3, B5, B6
        # Todo: EfficientNet Attention Head dimension = 8
        if self.backbone_name == 'eff_b5':
            self.block_idx = [2, 3, 4, 6, 9]
            features = [24, 40, 64, 176, 2048]
            # features = [192, 384, 768, 1536]
        
        else:
            print('Check your backbone again ^.^')
            return None
        
        
        self.decoder = nn.ModuleList([
            DecoderBLock(
                x_channels=features[-1], skip_channels=features[-2], desired_channels=features[-1]//4,
                layer=layers, num_heads=dec_heads, window_size=window_sizes, qkv_bias=qkv_bias, 
                attn_drop_prob=drop_prob, lin_drop_prob=drop_prob, device=device, dilations=(2, 3)
            ),
            DecoderBLock(
                x_channels=features[-1]//4, skip_channels=features[-3], desired_channels=features[-1]//8,
                layer=layers, num_heads=dec_heads, window_size=window_sizes, qkv_bias=qkv_bias, 
                attn_drop_prob=drop_prob, lin_drop_prob=drop_prob, device=device, dilations=(3, 5)
            ),
            DecoderBLock(
                x_channels=features[-1]//8, skip_channels=features[-4], desired_channels=features[-1]//16,
                layer=layers, num_heads=dec_heads, window_size=window_sizes, qkv_bias=qkv_bias, 
                attn_drop_prob=drop_prob, lin_drop_prob=drop_prob, device=device, dilations=(3, 5, 7)
            ),
            DecoderBLock(
                x_channels=features[-1]//16, skip_channels=features[-5], desired_channels=features[-1]//32,
                layer=layers, num_heads=dec_heads, window_size=window_sizes, qkv_bias=qkv_bias, 
                attn_drop_prob=drop_prob, lin_drop_prob=drop_prob, device=device, dilations=(3, 5, 7, 9)
            ),
        ]).to(device)
        self.head = Head(features[-1]//32, n_bins, min_val, max_val, drop_prob, device).to(device)
        self.act = nn.GELU()
        
        
    def forward(self, x, is_train=False):
        enc = self.encoder(x, is_train) 
        block1, block2_out, block3_out, block4_out, block5 = enc   
        
        if is_train:
            block2_out, match_weight2 = block2_out
            block3_out, match_weight3 = block3_out
            block4_out, match_weight4 = block4_out
        
        u1, attn_weight1 = self.decoder[0](block4_out, block5)
        u2, attn_weight2 = self.decoder[1](block3_out, u1)
        u3, attn_weight3 = self.decoder[2](block2_out, u2)
        u4, attn_weight4 = self.decoder[3](block1, u3)
        
        bin_widths, main_pred = self.head(u4)
        
        all_attn_weights = [attn_weight1, attn_weight2, attn_weight3, attn_weight4]
        if is_train:
            all_attn_weights.append(match_weight2)
            all_attn_weights.append(match_weight3)
            all_attn_weights.append(match_weight4)
        
        attn_weights = []
        for i in all_attn_weights: 
            attn_weights.extend(i)
        
        
        return bin_widths, main_pred, attn_weights
    
    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.head] # , self.atrous_3, self.atrous_6, self.atrous_9, self.final]
        for m in modules:
            yield from m.parameters()

if __name__ == '__main__': 
    img = torch.randn((1, 3, 416, 544)).to('cuda')
    model = MatchDistillNet(
        device='cuda', 
        backbone_name='eff_b5', 
        n_bins=128,
        window_sizes=7, 
        layers=2,
    ).to('cuda')
    
    def count_parameters(model):
        from prettytable import PrettyTable
        
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        total_encoder, total_decoder = 0, 0
        
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            
            params = parameter.numel()
            table.add_row([name, f"{params:,}"])
            total_params += params
            
            first_name = name.split('.')[0]
            if first_name == 'encoder':
                total_encoder += params
            elif first_name == 'decoder':
                total_decoder += params
            
                
        # print(table)
        print(f"Total Params \t\t\t: {total_params:,}")
        print(f"Total Encoder Params \t: {total_encoder:,}")
        print(f"Total Decoder Params \t: {total_decoder:,}")
        print(f"Total Heads Params \t\t: {(total_params - total_encoder - total_decoder):,}")
    
    def fvcore(model, img, is_train):
        from fvcore.nn import FlopCountAnalysis, flop_count_str

        # Hitung FLOPs
        flops = FlopCountAnalysis(model, (img, is_train))

        # Total FLOPs (dalam satuan GFLOPs)
        total_flops = flops.total()
        print("=== FLOPs Summary ===")
        print(f"Total FLOPs   : {total_flops / 1e9:.2f} GFLOPs")
        print("=" * 30)

        # Rincian berdasarkan operator (ops)
        print("=== FLOPs by Operator ===")
        flop_by_op = flops.by_operator()
        for op, val in flop_by_op.items():
            print(f"{op:<30}: {val / 1e9:.4f} GFLOPs")
        print("=" * 30)
        
    
    print("-----"*25)
    print(f"Full Model : Train")
    preds = model(img, is_train=True)
    print(f"bin_widths : {preds[0].shape}, Depths : {preds[1].shape}")
    count_parameters(model)
    fvcore(model, img, is_train=True)
    
    print("-----"*25)
    print(f"Testing time")
    preds = model(img, is_train=False)
    print(f"bin_widths : {preds[0].shape}, Depths : {preds[1].shape}")
    count_parameters(model)
    fvcore(model, img, is_train=False)
    
    
    
    
