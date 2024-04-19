# Source: https://github.com/hila-chefer/Transformer-Explainability

import numpy as np

import torch


class LRP:
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device = device

    def generate_attribution(
        self,
        input_img,
        index=None,
        method="transformer_attribution",
        is_ablation=False,
        start_layer=0,
    ):
        output = self.model(input_img)
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        # one_hot = torch.sum(one_hot.cuda() * output)
        one_hot = torch.sum(one_hot.to(self.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(
            torch.tensor(one_hot_vector).to(self.device),
            method=method,
            is_ablation=is_ablation,
            start_layer=start_layer,
            **kwargs
        )


class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam_attn(self, input, index=None):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grad = self.model.blocks[-1].attn.attn_gradients
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)

        cam = self.model.blocks[-1].attn.attn_cam
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
    
    @staticmethod
    def compute_rollout_attention(all_layer_matrices, start_layer=0):
        batch_size = all_layer_matrices[0].shape[0]
        num_tokens = all_layer_matrices[0].shape[1]
        eye = (
            torch.eye(num_tokens)
            .expand(batch_size, num_tokens, num_tokens)
            .to(all_layer_matrices[0].device)
        )
        all_layer_matrices = [
            all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))
        ]
        matrices_aug = [
            all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
            for i in range(len(all_layer_matrices))
        ]
        joint_attention = matrices_aug[start_layer]
        for i in range(start_layer + 1, len(matrices_aug)):
            joint_attention = matrices_aug[i].bmm(joint_attention)

        return joint_attention

    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        
        all_layer_attentions = []
        blocks = self.model.blocks
        for block in blocks:
            attn_heads = block.attn.attn_cam
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        
        rollout = self.compute_rollout_attention(all_layer_matrices=all_layer_attentions, start_layer=start_layer)
        
        return rollout[:, 0, 1:]
