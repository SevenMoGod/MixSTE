import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            
        attn = (q @ k.transpose(-2, -1)) * self.scale   
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     
        x = self.proj(x)                                   
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(                                 
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MixsteBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, frames, joints, pos_embed=False, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):    
        super().__init__()

        self.is_embed = pos_embed
        self.frames = frames
        self.joints = joints
        self.pos_drop = nn.Dropout(p=drop)

        self.spatial_pos_embed = nn.Parameter(torch.zeros([1, joints, embed_dim]))
        self.temporal_pos_embed = nn.Parameter(torch.zeros([1, frames, embed_dim]))

        # self.layer_norm = norm_layer(embed_dim)

        self.spatial_block = TransformerBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)

        self.temporal_block = TransformerBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)

    def spatial_forward_features(self, x):
        # input size: [batch_size*frames, joints, embed_dim]
        if self.is_embed:
            x += self.spatial_pos_embed
        x = self.pos_drop(x)
        x = self.spatial_block(x)
        # x = self.layer_norm(x)      # not sure if layer normalization is used here
        return x

    def temporal_forward_features(self, x):
        # input size: [batch_size*frames, joints, embed_dim]
        # joint separation
        x = rearrange(x, '(b f) j c -> (b j) f c', f=self.frames)
        if self.is_embed:
            x += self.temporal_pos_embed
        x = self.pos_drop(x)
        x = self.temporal_block(x)
        # x = self.layer_norm(x)        # not sure if layer normalization is used here
        x = rearrange(x, '(b j) f c -> (b f) j c', j=self.joints)
        return x

    def forward(self, x):
        x = self.spatial_forward_features(x)

        x = self.temporal_forward_features(x)
        return x


class Model(nn.Module):
    def __init__(self, num_frame=243, num_joints=17, in_chans=2, embed_dim=512, loop=8,
                 num_heads=8, mlp_ratio=1., qkv_bias=True, qk_scale=None,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.1, norm_layer=None):    
        """
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim (int): embedding dimension ratio
            loop (int): number of stacked mixste
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        out_dim = 3
        
        # patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim)

        is_embed = torch.zeros([loop])
        is_embed[0] = 1

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, loop)] 

        self.mixste_blocks = nn.ModuleList([
            MixsteBlock(
                embed_dim=embed_dim, num_heads=num_heads, frames=num_frame, joints=num_joints, pos_embed=is_embed[i],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(loop)])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x):
        # input size: [batch_size, frames, joints, 2 channels]
        _, f, j, _ = x.shape
        x = rearrange(x, 'b f j c -> (b f) j c',)
        x = self.patch_to_embedding(x)      
        for blk in self.mixste_blocks:
            x = blk(x)

        x = rearrange(x, '(b f) j c -> b f j c', f=f)
        x = self.head(x)
        return x

