"""
An implementation of the paper https://arxiv.org/abs/1911.05722
DelinQu, 2022.12.20
"""
import torch
import torch.nn as nn
import einops

@torch.no_grad()
def concat_all_gather(tensor):
    """
    将 group 中的 tensor 集中到 tensor_list 中.
    ! Warning: torch.distributed.all_gather has no gradient: https://zhuanlan.zhihu.com/p/76638962
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class MOCO(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        """
        dim: dimension of feature
        K: the size of queue
        m: momentum
        T: softmax temperature
        """
        super(MOCO, self).__init__()
        self.K, self.m, self.T = K, m, T
        self.f_q, self.f_k = base_encoder(num_classes=dim), base_encoder(num_classes=dim)

        """
        initialize, q <= random, k <= q.data wo grad  
        https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
        """
        for param_q, param_k in zip(self.f_q.parameters(), self.f_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        """
        create the queue use register_buffer:
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=register_buffer#torch.nn.Module.register_buffer
        """
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_theta_k(self):
        for param_q, param_k in zip(self.f_q.parameters(), self.f_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _update_dictionary(self, keys):
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = einops.rearrange(keys, "b c -> c b")
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, x_q, x_k):
        """
        Args:
            x_q (tensor): a batch of query image
            x_k (tensor): a batch of key image
        Returns:
            logits, labels: onehot vec and labels
        """
        q = self.f_q(x_q)                       # qurey: (b c) 
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._update_theta_k()
            # shuffle for making use of BN
            x_k, idx_unshuffle = self._batch_shuffle_ddp(x_k)

            k = self.f_k(x_k)                   # keys: (b c)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        l_pos = torch.einsum("bc,bc->b", [q, k]).unsqueeze(-1)  # positive logits: (b 1)
        l_neg = torch.einsum("bc,ck->bk", [q, self.queue.clone().detach()]) # negative logits: (b k)
        logits = torch.cat([l_pos, l_neg], dim=1) # logits: bx(1+k)

        # contrastive loss, Eqn.(1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # update dictionary
        self._update_dictionary(k)
        return logits, labels



