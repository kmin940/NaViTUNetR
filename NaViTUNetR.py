from functools import partial
from typing import List, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

from collections.abc import Sequence

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import deprecated_arg, ensure_tuple_rep



#helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def group_images_by_max_seq_len(
    images: List[Tensor],
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = 2048

) -> List[List[Tensor]]:

    calc_token_dropout = default(calc_token_dropout, always(0.))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-2:]
        ph, pw = map(lambda t: t // patch_size, image_dims)

        image_seq_len = (ph * pw)
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)

    #print("group len", len(groups))  # 2
    #print("group first element", len(groups[0]))  # 9
    #print("s", groups[0][0].shape)  # torch.Size([3, 384, 640])

    return groups

#normalization
#they use layernorm without bias, something that pytorch does not offer
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

#they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

#feedforward
def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

#attention
# self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)
# x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask) + queries
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None
    ):
        x = self.norm(x)
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        # print("q dim", q.shape) # 2,13,9,64 ##############
        # print("k dim", k.transpose(-1, -2).shape) # 2,13,64,1944 ############3
        dots = torch.matmul(q, k.transpose(-1, -2))
        # print("dots dim", dots.shape) # 2,13,9,1944
        # print("queries", queries.shape) # 2,9,32

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # print("attn dim", attn.shape)# 2,13,9,1944
        # print("v dim", v.shape) #2,13,1944,64
        out = torch.matmul(attn, v)
        # print("out dim", out.shape) # 2,13,9,64
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) # # torch.Size([2, 9, 1024])

#transformer block
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        attn_mask = None
    ):
        encoder_outputs = []
        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_mask = attn_mask) + x
            x = ff(x) + x
            # x = self.norm(x)
            encoder_outputs.append(x)
        return encoder_outputs  # self.norm(x)

class NaViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        token_dropout_prob = None
    ):
        super().__init__()
        # print(image_size) #[40, 56, 40]
        #image_height, image_width = pair(image_size)
        image_height, image_width = image_size
        #image_depth, image_height, image_width = image_size

        #what percent of tokens to dropout
        #if int or float given, then assume constant dropout prob
        #otherwise accept a callback that in turn calculates dropout prob from height and width

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0. < token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

        #calculate patching related stuff

        print(image_height)  #384
        print(patch_size)  # 32
        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        self.from_patch_embedding = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, patch_size * patch_size * num_classes),
            LayerNorm(patch_size * patch_size * num_classes),
        )

        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        #final attention pooling queries

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        #output to logits

        self.to_latent = nn.Identity()
        self.num_classes = num_classes

        # self.mlp_head = nn.Sequential(
        #     LayerNorm(dim),
        #     nn.Linear(dim, num_classes, bias = False)
        # )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        batched_images: Union[List[Tensor], List[List[Tensor]]], #assume different resolution images already grouped correctly
        group_images = True,
        group_max_seq_len = 2048
    ):
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout)
        # print("batch shape", batched_images.shape) # torch.Size([13, 3, 384, 640])

        arange = partial(torch.arange, device = device)
        pad_sequence = partial(orig_pad_sequence, batch_first = True)

        #auto pack if specified

        #print("batch shape before", batched_images.shape) # torch.Size([13, 3, 384, 640])
        if group_images:
            batched_images = group_images_by_max_seq_len(
                batched_images,
                patch_size = self.patch_size,
                calc_token_dropout = self.calc_token_dropout,
                max_seq_len = group_max_seq_len
            )
        # print("batch shape after", batched_images.shape)  # torch.Size([13, 3, 384, 640])
        #process images into variable lengthed sequences with attention mask

        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for images in batched_images:
            num_images.append(len(images))

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device = device, dtype = torch.long)
            #print("initial image ids", image_ids.shape)

            for image_id, image in enumerate(images):
                #print(image.ndim) # 2 if group_images False, if true, 3
                #print(image.shape) # torch.Size([384, 640]) if group_images False, if true torch.Size([3, 384, 640])
                assert image.ndim ==3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                # print("image_dims", image_dims)
                assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}'

                ph, pw = map(lambda dim: dim // p, image_dims)
                # print("ph, pw", ph, pw) # 12, 20

                pos = torch.stack(torch.meshgrid((
                    arange(ph),
                    arange(pw)
                ), indexing = 'ij'), dim = -1)

                pos = rearrange(pos, 'h w c -> (h w) c') #h, w, 2d positions
                seq = rearrange(image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1 = p, p2 = p) # torch.Size([240, 3072])

                seq_len = seq.shape[-2]
                #print("seq_len", seq_len) # 240

                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices
                    #print("keep_indices", keep_indices.shape) # torch.Size([216])
                    ##### print("seq before", seq.shape) # torch.Size([240, 3072]) ############# (num patches per image, else) one patch
                    seq = seq[keep_indices]
                    #print("seq after", seq.shape) # torch.Size([216, 3072])
                    pos = pos[keep_indices]

                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value = image_id)
                #print('ids',image_ids)
                #print("image_ids", image_ids.shape) # torch.Size([216])  torch.Size([432])
                sequences.append(seq)
                # print("seq", seq.shape) # torch.Size([216, 3072])
                positions.append(pos)
                #print("pos", pos.shape) # torch.Size([216, 2])

            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(sequences, dim = 0))
            # torch.Size([1944, 3072]), torch.Size([864, 3072]) #num images 9 * num patc"h per image 216, length per patch
            # 1944/216 = 9
            # print('batched_sequences element size', torch.cat(sequences, dim = 0).shape)

            batched_positions.append(torch.cat(positions, dim = 0))
            # print('batched_positions element size', torch.cat(positions, dim=0).shape)
            # torch.Size([1944, 2]), torch.Size([864, 2])

        #derive key padding mask

        lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device = device, dtype = torch.long)
        #print('lengths', lengths) # tensor([1944,  864], device='cuda:0')  # 216*9 = 1944
        max_length = arange(lengths.amax().item())
        #print("max_length", max_length)  # tensor([   0,    1,    2,  ..., 1941, 1942, 1943], device='cuda:0')
        key_pad_mask = rearrange(lengths, 'b -> b 1') <= rearrange(max_length, 'n -> 1 n')

        #derive attention mask, and combine with key padding mask from above
        # print(batched_image_ids[0].shape, batched_image_ids[1].shape)
        batched_image_ids = pad_sequence(batched_image_ids) # 1944 864->1944
        attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j')
        #print("rearrage", rearrange(batched_image_ids, 'b i -> b 1 i 1').shape) # torch.Size([2, 1, 1944, 1]) #1944 num of patches 216*9
        #print("rearrage", rearrange(batched_image_ids, 'b j -> b 1 1 j').shape) # torch.Size([2, 1, 1, 1944])
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')
        # print("attn_mask", attn_mask.shape) # torch.Size([2, 1, 1944, 1944])

        #combine patched images as well as the patched width / height positions for 2d positional embedding

        patches = pad_sequence(batched_sequences) # 2,1944, 3072 #batched_sequences  # torch.Size([1944, 3072]), torch.Size([864, 3072]) #num images 9 * num patch per image 216, length per patch
        # print("patches", patches.shape)
        patch_positions = pad_sequence(batched_positions)

        #need to know how many images for final attention pooling

        num_images = torch.tensor(num_images, device = device, dtype = torch.long)
        #print("num_images", num_images)  # tensor([9, 4], device='cuda:0')
        images_count = sum(num_images)

        #to patches

        x = self.to_patch_embedding(patches)  # 2,1944, 3072->1024      # patch_dim -> dim
        # print("x test0", x.shape) # torch.Size([2, 1944, 1024])

        #factorized 2d absolute positional embedding

        h_indices, w_indices = patch_positions.unbind(dim = -1)

        h_pos = self.pos_embed_height[h_indices]  # self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        w_pos = self.pos_embed_width[w_indices]

        x = x + h_pos + w_pos

        #embed dropout

        x = self.dropout(x)

        #attention

        encoder_outputs = self.transformer(x, attn_mask = attn_mask)
        #print("x test1", x.shape)  # torch.Size([2, 1944, 1024])
        # print("last layer", encoder_outputs[-1].shape) # torch.Size([2, 1920, 32])

        encoder_outputs = [rearrange(encoder_output, 'b (b2 ph pw) d -> (b b2) d ph pw', ph=ph, pw=pw)[:images_count, :, :, :] \
                            for encoder_output in encoder_outputs]
        # print("last layer2", encoder_outputs[-1].shape) # torch.Size([13, 12, 20, 32])
        # last layer torch.Size([7, 1920, 32])
        # last layer2 torch.Size([13, 24, 40, 32])
        return encoder_outputs

        # x = self.from_patch_embedding(x)
        # print("final1", x.shape) # final1 torch.Size([2, 1944, 3072]) #9*216(b ph pw)=1944, channel*p*p= 3072
        # # want : 13, c=13, h, w
        # x = rearrange(x, 'b (b2 pc1 pc2) (c p1 p2) -> (b b2) c (pc1 p1) (pc2 p2)', c=self.num_classes, p1 = p, p2=p, pc1=ph, pc2=pw)
        # print("x FINAL shape", x.shape) # x FINAL shape torch.Size([16, 3, 384, 640])
        # #x = rearrange(x, 'b (c h w) -> b c h w', h=image_dims[0], w=image_dims[1])
        # return x[:images_count, :, :, :]


        #
        # #do attention pooling at the end
        #
        # max_queries = num_images.amax().item()
        #
        # queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])
        # print("queries", queries.shape) # 2,9,32
        #
        # #attention pool mask
        #
        # image_id_arange = arange(max_queries)
        #
        # attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')
        #
        # attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j')
        #
        # attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')
        #
        # #attention pool
        #
        # x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask) + queries
        # #print("x test2", x.shape) # torch.Size([2, 9, 1024])
        #
        # x = rearrange(x, 'b n d -> (b n) d')
        # #print("x test3", x.shape) # torch.Size([18, 1024])
        #
        # #each batch element may not have same amount of images
        #
        # is_images = image_id_arange < rearrange(num_images, 'b -> b 1')
        # is_images = rearrange(is_images, 'b n -> (b n)')
        # print("is_images", is_images.shape) # 18
        # print("x", x.shape) # 18,32
        # x = x[is_images]
        # print("x images", x.shape) # 13,32
        # #project out to logits

        # x = self.to_latent(x)
        # #print(x.shape) # torch.Size([13, 1024]) 1024=dim
        #
        # #print("final",self.mlp_head(x).shape) # final torch.Size([13, 13])
        # # nn.Linear(dim, num_classes * image_height * image_width)
        # x = self.from_patch_embedding(x) # 13, 3072=(c p1 p2) !=(c ph pw p1 p2)
        # #print("final1", x.shape) # final1 torch.Size([13, 737280])
        # # want : 13, c, h, w
        # x = rearrange(x, 'b (c h w) -> b c h w', h=image_dims[0], w=image_dims[1])
        # #print("final2", x.shape) # final2 torch.Size([13, 3, 384, 640])


        #return x
        #return self.mlp_head(x)


class NaViTUNetR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.

        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        # self.vit = ViT(
        #     in_channels=in_channels,
        #     img_size=img_size,
        #     patch_size=self.patch_size,
        #     hidden_size=hidden_size,
        #     mlp_dim=mlp_dim,
        #     num_layers=self.num_layers,
        #     num_heads=num_heads,
        #     proj_type=proj_type,
        #     classification=self.classification,
        #     dropout_rate=dropout_rate,
        #     spatial_dims=spatial_dims,
        #     qkv_bias=qkv_bias,
        #     save_attn=save_attn,
        # )
        self.vit = NaViT(
            channels = in_channels,
            heads = num_heads,
            image_size = img_size,
            patch_size = self.patch_size[0],
            num_classes = num_heads,
            dim = hidden_size,
            depth = 4,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1,
            token_dropout_prob=None
            )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # RuntimeError: Given transposed=1, weight of size [32, 32, 2, 2](hiddensize, featuresize*2, upsample_kernel),
        # expected input [13, 24, 40, 32]
        # to have 32 channels, but got 24 channels instead
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[0]
        enc2 = self.encoder2(x2)
        x3 = hidden_states_out[1]
        enc3 = self.encoder3(x3)
        x4 = hidden_states_out[2]
        enc4 = self.encoder4(x4)
        dec4 = hidden_states_out[3]
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)
