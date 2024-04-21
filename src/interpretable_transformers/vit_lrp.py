# Source: https://github.com/hila-chefer/Transformer-Explainability
# Vision Transformer (ViT) in PyTorch Hacked together by / Copyright 2020 Ross Wightman

from src.interpretable_transformers.custom_layers import *
from src.interpretable_transformers.layer_helpers import to_2tuple
from src.interpretable_transformers.weight_init import trunc_normal_

from typing import Optional, Dict, List

from einops import rearrange

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # FIXME: look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam: torch.Tensor, **kwargs) -> torch.Tensor:
        cam = cam.transpose(1, 2)
        cam = cam.reshape(
            cam.shape[0],
            cam.shape[1],
            (self.img_size[0] // self.patch_size[0]),
            (self.img_size[1] // self.patch_size[1]),
        )
        return self.proj.relprop(cam, **kwargs)


class Attention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads

        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.matmul1 = einsum("bhid,bhjd->bhij")  # A = Q*K^T
        self.matmul2 = einsum("bhij,bhjd->bhid")  # attn = A*V

        self.qkv = Linear(in_features=embed_dim, out_features=embed_dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(p=attn_drop)
        self.proj = Linear(in_features=embed_dim, out_features=embed_dim)
        self.proj_drop = Dropout(p=proj_drop)
        self.softmax = Softmax(dim=-1)

        self._attn = None
        self._attn_cam = None
        self._attn_gradients = None
        self._v = None
        self._v_cam = None

    @property
    def attn(self) -> torch.Tensor:
        return self._attn

    @attn.setter
    def attn(self, attn: torch.Tensor):
        self._attn = attn

    @property
    def attn_cam(self) -> torch.Tensor:
        return self._attn_cam

    @attn_cam.setter
    def attn_cam(self, attn_cam: torch.Tensor):
        self._attn_cam = attn_cam

    @property
    def attn_gradients(self) -> torch.Tensor:
        return self._attn_gradients

    @attn_gradients.setter
    def attn_gradients(self, attn_gradients: torch.Tensor):
        self._attn_gradients = attn_gradients

    # Callback for register_hook
    def set_attn_gradients(self, attn_gradients: torch.Tensor):
        self.attn_gradients = attn_gradients

    @property
    def v(self) -> torch.Tensor:
        return self._v

    @v.setter
    def v(self, v: torch.Tensor):
        self._v = v

    @property
    def v_cam(self) -> torch.Tensor:
        return self._v_cam

    @v_cam.setter
    def v_cam(self, v_cam: torch.Tensor):
        self._v_cam = v_cam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=h)

        self.v = v

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)
        self.attn = attn

        if attn.requires_grad:
            attn.register_hook(self.set_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam: torch.Tensor, **kwargs) -> torch.Tensor:
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, "b n (h d) -> b h n d", h=self.num_heads)

        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)  # attn = A*V
        cam1 /= 2
        cam_v /= 2

        self.v_cam = cam_v
        self.attn_cam = cam1

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)  # A = Q*K^T
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange(
            [cam_q, cam_k, cam_v],
            "qkv b h n d -> b n (qkv h d)",
            qkv=3,
            h=self.num_heads,
        )

        return self.qkv.relprop(cam_qkv, **kwargs)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = Linear(in_features=in_features, out_features=hidden_features)
        self.act = GELU()
        self.fc2 = Linear(in_features=hidden_features, out_features=out_features)
        self.drop = Dropout(p=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam: torch.Tensor, **kwargs) -> torch.Tensor:
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Block(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()

        self.norm1 = LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.mlp = Mlp(
            in_features=embed_dim, 
            hidden_features=int(embed_dim * mlp_ratio), 
            drop=drop
        )
        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam: torch.Tensor, **kwargs) -> torch.Tensor:
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        mlp_head: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )

        self.norm = LayerNorm(normalized_shape=embed_dim, eps=1e-6)

        if mlp_head:
            self.head = Mlp(
                in_features=embed_dim,
                hidden_features=int(embed_dim * mlp_ratio),
                out_features=num_classes,
            )
        else:
            self.head = Linear(
                in_features=embed_dim, 
                out_features=num_classes
            )

        # FIXME: not quite sure what the proper weight init is supposed to be,
        # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        trunc_normal_(self.pos_embed, std=0.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

        self._input_grad = None

    @property
    def input_grad(self) -> Optional[torch.Tensor]:
        return self._input_grad

    @input_grad.setter
    def input_grad(self, input_grad: torch.Tensor):
        self._input_grad = input_grad

    # Callback for register_hook
    def set_input_grad(self, input_grad: torch.Tensor):
        self.input_grad = input_grad

    @property
    def no_weight_decay(self) -> Dict[str, str]:
        return {"pos_embed", "cls_token"}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def compute_rollout_attention(
        all_layer_matrices: List[torch.Tensor], 
        start_layer: int = 0
    ) -> torch.Tensor:
        # adding residual consideration
        batch_size = all_layer_matrices[0].shape[0]
        num_tokens = all_layer_matrices[0].shape[1]

        eye = (
            torch.eye(num_tokens)
            .expand(batch_size, num_tokens, num_tokens)
            .to(all_layer_matrices[0].device)
        )

        all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
        # all_layer_matrices = [
        #     all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
        #     for i in range(len(all_layer_matrices))
        # ]

        joint_attention = all_layer_matrices[start_layer]
        for i in range(start_layer + 1, len(all_layer_matrices)):
            joint_attention = all_layer_matrices[i].bmm(joint_attention)

        return joint_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        # cls_tokens implementation from Phil Wang
        cls_tokens = self.cls_token.expand(B, -1, -1) 

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        # Small fix to use this in validation and testing
        if x.requires_grad:
            x.register_hook(self.set_input_grad)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def relprop(
        self,
        cam: torch.Tensor = None,
        method: str = "transformer_attribution",
        is_ablation: bool = False,
        start_layer: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)

        for block in reversed(self.blocks):
            cam = block.relprop(cam, **kwargs)

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            attn_cams = []
            for block in self.blocks:
                attn_heads = block.attn.attn_cam.clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)

            cam = self.compute_rollout_attention(all_layer_matrices=attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam

        elif method == "transformer_attribution":
            cams = []
            for block in self.blocks:
                grad = block.attn.attn_gradients
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])

                cam = block.attn.attn_cam
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)

                cams.append(cam.unsqueeze(0))

            rollout = self.compute_rollout_attention(all_layer_matrices=cams, start_layer=start_layer)

            cam = rollout[:, 0, 1:]
            return cam

        elif method == "last_layer":
            cam = self.blocks[-1].attn.attn_cam
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])

            if is_ablation:
                grad = self.blocks[-1].attn.attn_gradients
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam

            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.attn
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.attn_cam
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])

            if is_ablation:
                grad = self.blocks[1].attn.attn_gradients
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam

            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam
