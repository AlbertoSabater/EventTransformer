from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import normal

import sys
sys.path.append('../')
from models.positional_encoding import fourier_features

import os
import copy




# =============================================================================
# Token processing blocks
# =============================================================================

# =============================================================================
# Processing blocks
# X-attention input
#   Q/z_input         -> (#latent_embs, batch_size, embed_dim)
#   K/V/x             -> (#events, batch_size, embed_dim)
#   key_padding_mask  -> (batch_size, #event)
# output -> (#latent_embs, batch_size, embed_dim)
# =============================================================================
class AttentionBlock(nn.Module):    # PerceiverAttentionBlock
    def __init__(self, opt_dim, heads, dropout, att_dropout, **args):
        super(AttentionBlock, self).__init__()

        self.layer_norm_x = nn.LayerNorm([opt_dim])
        self.layer_norm_1 = nn.LayerNorm([opt_dim])
        self.layer_norm_att = nn.LayerNorm([opt_dim])
        
        self.attention = nn.MultiheadAttention(
            opt_dim,            # embed_dim
            heads,              # num_heads
            dropout=att_dropout,
            bias=True,
            add_bias_kv=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(opt_dim, opt_dim)
        self.layer_norm_2 = nn.LayerNorm([opt_dim])
        self.linear2 = nn.Linear(opt_dim, opt_dim)
        self.linear3 = nn.Linear(opt_dim, opt_dim)


    def forward(self, x, z_input, mask=None, q_mask=None, **args):
        x = self.layer_norm_x(x)
        z = self.layer_norm_1(z_input)
        
        z_att, _ = self.attention(z, x, x, key_padding_mask=mask, attn_mask=q_mask)  # Q, K, V
        
        z_att = z_att + z_input
        z = self.layer_norm_att(z_att)

        z = self.dropout(z)
        z = self.linear1(z)
        z = torch.nn.GELU()(z)

        z = self.layer_norm_2(z)
        z = self.linear2(z)
        z = torch.nn.GELU()(z)
        z = self.dropout(z)
        z = self.linear3(z)
        
        return z + z_att


class TransformerBlock(nn.Module):
    def __init__(self, opt_dim, latent_blocks, dropout, att_dropout, heads, cross_heads, **args):
        super(TransformerBlock, self).__init__()

        self.cross_attention = AttentionBlock(opt_dim=opt_dim, heads=cross_heads, dropout=dropout, att_dropout=att_dropout)
        self.latent_attentions = nn.ModuleList([
            AttentionBlock(opt_dim=opt_dim, heads=heads, dropout=dropout, att_dropout=att_dropout) for _ in range(latent_blocks)
        ])

    def forward(self, x_input, z, mask=None, q_mask=None, **args):
        z = self.cross_attention(x_input, z, mask=mask, q_mask=q_mask)
        for latent_attention in self.latent_attentions:
            z = latent_attention(z, z, q_mask=q_mask)
        return z



# Feed Forward Net
class MLPBlock(nn.Module):
    def __init__(self, ipt_dim, embed_dim, init_layers, 
                 add_x_input=False, dropout=0.0, **args):   #, num_layers
        super(MLPBlock, self).__init__()
        self.embed_dim = embed_dim
        self.add_x_input = add_x_input
        self.dropout = dropout
        if self.dropout > 0.0: self.dropout = nn.Dropout(p=dropout)
        self.seq_init = self._get_sequential_block(init_layers, ipt_dim)
            
            
            
    def _get_sequential_block(self, layers, ipt_dim):
        seq = []
        for l in layers:
            l_name, opt_dim, activation = l.split('_'); opt_dim = int(opt_dim)
            if opt_dim == -1: opt_dim = self.embed_dim
            if self.dropout: seq.append(self.dropout)
            # Layer type
            if l_name == 'ff': seq.append(nn.Linear(ipt_dim, opt_dim))
            else: raise ValueError(f'MLPBlock l_name [{l_name}] not handled')
            # Layer activation
            if activation == 'rel': seq.append(nn.ReLU())
            elif activation == 'gel': seq.append(nn.GELU())
            else: raise ValueError(f'MLPBlock activation [{activation}] not handled')
            ipt_dim = opt_dim
        return nn.Sequential(*seq)
        
    # x_input -> (#events, batch_size, emb_dim)
    # mask -> (batch_size, #events) -> (#events, batch_size)
    def forward(self, x_input, mask=None, pos_embs=None, **args):
        x = self.seq_init(x_input)
        if mask is not None: 
            mask = mask.reshape(mask.shape[1], mask.shape[0])
            
        if self.add_x_input: x = x + x_input
        
        return x
    
    
  
def get_block(name, params):
    if name == 'MLP': return MLPBlock(**params)
    elif name == 'TransformerBlock': return TransformerBlock(**params)
    else: raise ValueError(f'Block [{name}] not implemented')




# Transforms a set of latent vectors/Q into a single descriptor
class LatentEmbsCompressor(nn.Module):
    
    # Linear + compressor
    def __init__(self, opt_dim, clf_mode, params, embs_norm):
        super(LatentEmbsCompressor, self).__init__()
        self.clf_mode = clf_mode
        self.linear1 = nn.Linear(opt_dim, opt_dim)
        if embs_norm: self.layer_norm = nn.LayerNorm([opt_dim])
        self.embs_norm = embs_norm
    
    # batch_size x num_latent x emb_dim
    def forward(self, z):
        if self.embs_norm: z = self.layer_norm(z)
        z = self.linear1(z)
        z = F.relu(z)
        if self.clf_mode == 'gap':
            # Average every latent
            z = z.mean(dim=0)
        else: raise ValueError('clf_mode [{}] nor handled'.format(self.clf_mode))

        return z    

  
# =============================================================================
# Backbone
# =============================================================================
class EvNetBackbone(nn.Module):
    def __init__(self, 
                 pos_encoding,  # Positional encoding params -> {name, {params}}
                 token_dim,                     # Size of the flattened patch_tokens
                 embed_dim,                     # Dimensionality of the latent_vectors and patch tokens
                 num_latent_vectors,            # Number of latent vectors
                 event_projection,              # Event Pre-processing Net {name, {params}}
                 preproc_events,                # Event Pre-processing Net (after positional encoding) {name, {params}}
                 proc_events,                   # Event Processing Net (with skip connection) {name, {params}}
                 proc_memory,                   # Attention Layers {name, {params}}
                 return_last_q,                 # Return processed latent vectors or updated ones
                 proc_embs,                     # Latent Vectors summarization Net {clf_mode, params}
                 downsample_pos_enc,            # Minimize positional encoding size
                 pos_enc_grad=False,            # Learnable latent vectors?
                 model_version=None,     # Dummy
                 # **qargs,        # Remove
                 ):
        super(EvNetBackbone, self).__init__()
        
        self.return_last_q = return_last_q

        self.num_latent_vectors = num_latent_vectors
        self.downsample_pos_enc = downsample_pos_enc
        self.memory_vertical = nn.Parameter(normal.Normal(0.0, 0.2).sample((num_latent_vectors, embed_dim)).clip(-2,2), requires_grad=True)

        # Positional encodings
        if pos_encoding is not None:
            if pos_encoding['name'] == 'fourier':
                if pos_encoding['params']['bands'] == -1: pos_encoding['params']['bands'] = embed_dim//4
                # frame_shape, fourier_bands
                pos_enc_params = copy.deepcopy(pos_encoding['params'])
                pos_enc_params['shape'] = (pos_encoding['params']['shape'][0]//downsample_pos_enc, pos_encoding['params']['shape'][1]//downsample_pos_enc)
                self.pos_encoding = nn.Parameter(fourier_features(**pos_enc_params).permute(1,2,0), requires_grad=pos_enc_grad)
                pos_emb_dim = self.pos_encoding.shape[2]
                self.pos_encoding = self.pos_encoding.type_as(self.memory_vertical)
            else:
                raise ValueError('Positional Encoding[{}] not implemented'.format(pos_encoding['name']))
        else: 
            self.pos_encoding = None
            print(' ** Not using pos_encoding')
            
        
        # Event pre-proc block -> Linear transformation on tokens
        event_projection['params']['embed_dim'] = embed_dim
        self.event_projection = get_block(event_projection['name'], {**event_projection['params'], **{'ipt_dim': token_dim}})

        # Events preprocessing -> Linear transformation on tokens
        self.preproc_events = preproc_events
        preproc_events['params']['embed_dim'] = embed_dim
        self.preproc_block_events = get_block(preproc_events['name'], {**preproc_events['params'], 
                                           **{'ipt_dim': int(event_projection['params']['init_layers'][-1].split('_')[1])+pos_emb_dim}})

        # Transforms events at each level
        proc_events['params']['opt_dim'] = embed_dim
        proc_events['params']['ipt_dim'] = embed_dim
        proc_events['params']['embed_dim'] = embed_dim
        self.proc_event_blocks = nn.ModuleList([ get_block(proc_events['name'], proc_events['params']) ])

        # Transforms latent embeddings at each level
        proc_memory['params']['opt_dim'] = embed_dim
        proc_memory['params']['embed_dim'] = embed_dim
        self.proc_memory_blocks = nn.ModuleList([ get_block(proc_memory['name'], proc_memory['params']) ])

        proc_embs['opt_dim'] = embed_dim
        proc_embs['params']['embed_dim'] = embed_dim
        self.proc_embs_block = LatentEmbsCompressor(**proc_embs)


    # input -> (#timesteps, batch_size, #events, token_dim/event_xy)
    def forward(self, kv, pixels):
        
        pixels = pixels // self.downsample_pos_enc
        
        batch_size = kv.shape[1]
        num_time_steps = kv.shape[0]
        # True to ignore empty patches    |   (#timesteps, batch_size, #events)    
        samples_mask = kv.sum(-1) == 0          #  to ignore in the attention block   
        samples_mask_time = kv.sum(-1).sum(-1) == 0     # to ignore when there is no events at some time-step -> in a short clip when it is left-padded

        # Linear projection
        kv = self.event_projection(kv)              # (num_timesteps, batch_size, num_events, token_dim)

        #  Add pos. encodings
        if self.pos_encoding is not None:
            pos_embs = self.pos_encoding[pixels[:,:,:,1], pixels[:,:,:,0],:]
            kv = torch.cat([kv, pos_embs], dim=-1)
        else: pos_embs = None
        kv = kv.permute(0,2,1,3)                     # (num_timesteps, num_events, batch_size, token_dim)
        
        # To embedding size
        kv = self.preproc_block_events(kv)              # (num_timesteps, batch_size, token_dim, num_events)
        
        # Initial latent vectors
        latent_vectors = self.memory_vertical.unsqueeze(1)
        latent_vectors = latent_vectors.expand(-1, batch_size, -1)    # (num_latent_vectors, batch_size, embed_dim)
        
        # Initialize inp_q
        inp_q = latent_vectors
        
        for t in range(num_time_steps):
            inp_kv = kv[t]                                              # (num_events, batch_size, token_dim)
            mask_t = samples_mask[t]
            mask_time_t = samples_mask_time[t]
            pos_embs_t = pos_embs[t].reshape(pos_embs.shape[2], pos_embs.shape[1], pos_embs.shape[3]) if pos_embs is not None else None
                
            # Process events
            proc_event_block = self.proc_event_blocks[0]
            inp_kv = proc_event_block(x_input=inp_kv, z=inp_kv, mask=mask_t, pos_embs=pos_embs_t)

            # Process memory
            proc_memory_block = self.proc_memory_blocks[0]
            inp_q = proc_memory_block(inp_kv, inp_q, mask=mask_t)
            inp_q[:, mask_time_t] = latent_vectors[:, mask_time_t]
                
            # Update latent_vectors
            latent_vectors = inp_q + latent_vectors
        
        embs = inp_q if self.return_last_q else latent_vectors
        res = self.proc_embs_block(embs)
        
        return res


# Classification backbone
class CLFBlock(nn.Module):
    def __init__(self, ipt_dim, opt_classes, **args):
        super(CLFBlock, self).__init__()
        self.linear_1 = nn.Linear(ipt_dim, ipt_dim)
        self.linear_2 = nn.Linear(ipt_dim, opt_classes)
    
    # z -> (batch_size, emb dim)
    def forward(self, z):
        z = F.relu(self.linear_1(z))
        z = self.linear_2(z)
        clf = F.log_softmax(z, dim=1)
        return clf
    



