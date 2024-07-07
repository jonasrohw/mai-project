import torch
import torch.nn as nn
import clip
import utils
import math
import torch.nn.functional as F
from models import CrossAttention
class SingleBlock(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """
    def __init__(self, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size         = utils.output_features    # 512
        self.t_size         = utils.text_features      # 512
        self.b_size         = utils.spatial_features   # 4
        self.output_size    = utils.hidden_features    # 512
        self.num_inter_head = utils.num_inter_head     # 8
        self.num_intra_head = utils.num_intra_head     # 8
        self.num_block      = utils.num_block          # 2
        self.spa_block      = utils.spa_block          # 2
        self.que_block      = utils.que_block          # 2

        self.iteration      = utils.iteration

        self.v_lin = nn.Linear(self.v_size, self.output_size)
        self.b_lin = nn.Linear(self.b_size, self.output_size)
        self.t_lin = nn.Linear(self.t_size, self.output_size)

        self.interBlock       = InterModalityUpdate(self.output_size, self.output_size, self.output_size, self.num_inter_head, drop)
        self.intraBlock       = DyIntraModalityUpdate(self.output_size, self.output_size, self.output_size, self.num_intra_head, drop)
        self.lin1             = nn.Linear(self.output_size, self.output_size)

        if utils.exp_id == 2:
            self.n_relations    = 8
            self.dim_g          = int(self.output_size / self.n_relations)

            self.RelationModule   = RelationModule(n_relations      = self.n_relations, 
                                                hidden_dim       = self.output_size, 
                                                key_feature_dim  = self.dim_g, 
                                                geo_feature_dim  = self.dim_g,
                                                drop             = drop)

        self.drop = nn.Dropout(drop)

    def forward(self, v, t, v_mask, t_mask):
        """
            v: visual feature      [batch, num_obj, feat_size]
            t: text                [batch, max_len, feat_size]
            v_mask                 [batch, num_obj]
            t_mask                 [batch, max_len]
        """
        # transfor features
        v = self.v_lin(self.drop(v))
        #b = self.b_lin(self.drop(b))
        t = self.t_lin(self.drop(t))

        v_init = v.clone()
        #b_init = b.clone()
        t_init = t.clone()

        for i in range(self.num_block):
            v, t = self.interBlock(v, t, v_mask, t_mask)
            v, t = self.intraBlock(v, t, v_mask, t_mask)

            # if utils.exp_id == 2:
            #     v = self.RelationModule(v, b, v_mask, t, t_mask)

        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        t_mean = (t * t_mask.unsqueeze(2)).sum(1) / t_mask.sum(1).unsqueeze(1)
        
        out = self.lin1(self.drop(v_mean * t_mean))

        return out


class RelationModule(nn.Module):
    def __init__(self, n_relations=8, hidden_dim=512, key_feature_dim=64, geo_feature_dim=64, drop=0.0):
        super(RelationModule, self).__init__()

        self.Nr        = n_relations
        self.dim_g     = geo_feature_dim
        self.relation  = nn.ModuleList()

        self.relu      = nn.ReLU(inplace=True)
        self.drop      = nn.Dropout(drop)
        self.sigmoid   = nn.Sigmoid()

        # text
        self.t_compress = nn.Linear(hidden_dim, 1)
        self.t_map_obj  = nn.Linear(hidden_dim, utils.output_size**2)

        # for N in range(self.Nr):
        self.relation = RelationUnit(n_relations, hidden_dim, key_feature_dim, geo_feature_dim)

    def forward(self, v, b, v_mask, t, t_mask):

        # v: torch.Size([bs, num_obj, 512])
        # t: torch.Size([bs, t_len, 512])
        # b: torch.Size([bs, num_obj, 512])

        t = t * t_mask.unsqueeze(2)

        # summarize text
        t_compress = self.t_compress(t)
        t_mask     = (t_mask == 0).unsqueeze(-1).expand_as(t_compress)      # (bs, t_len, 1)
        t_compress = t_compress.masked_fill_(t_mask, -float('inf'))         # (bs, t_len, 1)
        t_sscore   = self.sigmoid(t_compress)                               # (bs, t_len, 1)
        t_summary  = torch.bmm(t.transpose(1,2), t_sscore).squeeze(-1)      # (bs, 512)
        t_map_obj  = self.relu(self.t_map_obj(t_summary))                   # (bs, 36*36)

        concat = self.relation(v, b, t_map_obj, v_mask)

        return concat + v

class RelationUnit(nn.Module):
    def __init__(self, n_relations=8, hidden_dim=512, key_feature_dim=64, geo_feature_dim=64):
        super(RelationUnit, self).__init__()

        self.n_relations = n_relations

        self.dim_g  = geo_feature_dim
        self.dim_k  = key_feature_dim

        self.WG_1   = nn.Linear(hidden_dim, hidden_dim)
        self.WG_2   = nn.Linear(hidden_dim, hidden_dim)

        self.WK     = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.WQ     = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.WV     = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.relu   = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()

    def forward(self, v, b, t_map_obj, v_mask):

        # v: torch.Size([256, 36, 512])
        # b: torch.Size([256, 36, 512])

        bs, num_obj, dim  = v.size()
        t_map_obj         = t_map_obj.view(bs, num_obj, -1).unsqueeze(1)

        dim_per_rel = dim // self.n_relations
        mask_reshape = (bs, 1, 1, num_obj)

        def shape(x)  : return x.view(bs, -1, self.n_relations, dim_per_rel).transpose(1, 2)
        def unshape(x): return x.transpose(1, 2).contiguous().view(bs, -1, self.n_relations * dim_per_rel)

        w_g_1 = self.WG_1(b) * v_mask.unsqueeze(2)                      # (bs, n_rel, num_obj, 64)
        w_g_2 = self.WG_2(b) * v_mask.unsqueeze(2)                      # (bs, n_rel, num_obj, 64)

        w_g_1 = shape(w_g_1)
        w_g_2 = shape(w_g_2)

        w_g = self.relu(torch.matmul(w_g_1, w_g_2.transpose(2,3)))      # (bs, n_rel, num_obj, num_obj)
        w_g = w_g + t_map_obj                                           # (bs, n_rel, num_obj, num_obj)

        w_k = shape(self.WK(v))                                         # (bs, n_rel, num_obj, 64)
        w_k = w_k.view(bs, -1, num_obj, 1, self.dim_k)                  # (bs, n_rel, num_obj, 1, 64)

        w_q = shape(self.WQ(v))                                         # (bs, n_rel, num_obj, 64)
        w_q = w_q.view(bs, -1, 1, num_obj, self.dim_k)                  # (bs, n_rel, 1, num_obj, 64)

        scaled_dot = torch.sum((w_k * w_q), dim=-1)                     # (bs, n_rel, num_obj, num_obj)
        scaled_dot = scaled_dot / math.sqrt(self.dim_k)                 # (bs, n_rel, num_obj, num_obj)

        w_mn    = torch.log(torch.clamp(w_g, min = 1e-6)) + scaled_dot  # (bs, n_rel, num_obj, num_obj)
        v_mask  = (v_mask == 0).view(mask_reshape).expand_as(w_mn)      # (bs, n_rel, num_obj, num_obj)
        w_mn    = w_mn.masked_fill_(v_mask, -float('inf'))              # (bs, n_rel, num_obj, num_obj)
        w_mn    = F.softmax(w_mn, dim=-1)                               # (bs, n_rel, num_obj, num_obj)

        w_v = shape(self.WV(v))                     # (bs, n_rel, num_obj, 64)

        output = torch.matmul(w_mn, w_v)     # (bs, n_rel, num_obj, 64)
        output = unshape(output)

        return output


class InterModalityUpdate(nn.Module):
    """
        Inter-modality Attention Flow
    """
    def __init__(self, v_size, t_size, output_size, num_head, drop=0.0):

        super(InterModalityUpdate, self).__init__()
        self.v_size      = v_size        # 512
        self.t_size      = t_size        # 512
        self.output_size = output_size   # 512
        self.num_head    = num_head      # 8

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.t_lin = nn.Linear(t_size, output_size * 3)

        self.v_output = nn.Linear(output_size + v_size, output_size)
        self.t_output = nn.Linear(output_size + t_size, output_size)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, v, t, v_mask, t_mask):
        """
            v: visual feature      [batch, num_obj, feat_size]
            t: text                [batch, max_len, feat_size]
            v_mask                 [batch, num_obj]
            t_mask                 [batch, max_len]
        """

        batch_size, num_obj = v_mask.shape
        _         , max_len = t_mask.shape

        dim_per_head = self.output_size // self.num_head

        def shape(x)    : return x.view(batch_size, -1, self.num_head, dim_per_head).transpose(1, 2)
        def unshape(x)  : return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head*dim_per_head)

        v = v * v_mask.unsqueeze(2)
        t = t * t_mask.unsqueeze(2)

        # transfor features
        v_trans = self.v_lin(self.drop(self.relu(v)))
        t_trans = self.t_lin(self.drop(self.relu(t)))

        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        t_k, t_q, t_v = torch.split(t_trans, t_trans.size(2) // 3, dim=2)

        # transform features
        v_k = shape(v_k)  # (batch_size, num_head, num_obj, dim_per_head)
        v_q = shape(v_q)  # (batch_size, num_head, num_obj, dim_per_head)
        v_v = shape(v_v)  # (batch_size, num_head, num_obj, dim_per_head)

        t_k = shape(t_k)  # (batch_size, num_head, max_len, dim_per_head)
        t_q = shape(t_q)  # (batch_size, num_head, max_len, dim_per_head)
        t_v = shape(t_v)  # (batch_size, num_head, max_len, dim_per_head)

        # inner product
        t2v = torch.matmul(v_q, t_k.transpose(2,3)) / math.sqrt(dim_per_head)   # (batch_size, num_head, num_obj, max_len)
        v2t = torch.matmul(t_q, v_k.transpose(2,3)) / math.sqrt(dim_per_head)   # (batch_size, num_head, max_len, num_obj)

        t_mask = (t_mask == 0).unsqueeze(1).unsqueeze(1).expand_as(t2v)
        v_mask = (v_mask == 0).unsqueeze(1).unsqueeze(1).expand_as(v2t)

        # set padding object/word attention to negative infinity & normalized by square root of hidden dimension
        t2v = t2v.masked_fill(t_mask, -float('inf'))   # (batch_size, num_head, num_obj, max_len)
        v2t = v2t.masked_fill(v_mask, -float('inf'))   # (batch_size, num_head, max_len, num_obj)

        # softmax attention
        interMAF_t2v = F.softmax(t2v.float(), dim=-1).type_as(t2v) # (batch_size, num_head, num_obj, max_len) over max_len
        interMAF_v2t = F.softmax(v2t.float(), dim=-1).type_as(v2t) # (batch_size, num_head, max_len, num_obj) over num_obj

        v_update = unshape(torch.matmul(interMAF_t2v, t_v))
        t_update = unshape(torch.matmul(interMAF_v2t, v_v))

        # update new feature
        cat_v = torch.cat((v, v_update), dim=2)
        cat_t = torch.cat((t, t_update), dim=2)

        updated_v = self.v_output(self.drop(cat_v))
        
        if utils.exp_id == 2:
            updated_t = t
        else:
            updated_t = self.t_output(self.drop(cat_t))

        return updated_v, updated_t


class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """
    def __init__(self, v_size, t_size, output_size, num_head, drop=0.0):

        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.t_size = t_size
        self.output_size = output_size
        self.num_head = num_head

        self.v4t_gate_lin = nn.Linear(v_size, output_size)
        self.t4v_gate_lin = nn.Linear(t_size, output_size)

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.t_lin = nn.Linear(t_size, output_size * 3)

        self.v_output = nn.Linear(output_size, output_size)
        self.t_output = nn.Linear(output_size, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(drop)
    
    def forward(self, v, t, v_mask, t_mask):

        """
        v: visual feature      [batch, num_obj, feat_size]
        t: text                [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        t_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = t_mask.shape

        dim_per_head = self.output_size // self.num_head

        # average pooling
        v_mean      = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        t_mean      = (t * t_mask.unsqueeze(2)).sum(1) / t_mask.sum(1).unsqueeze(1)

        # conditioned gating vector
        v4t_gate    = self.sigmoid(self.v4t_gate_lin(self.drop(self.relu(v_mean)))).unsqueeze(1) #[batch, 1, feat_size]
        t4v_gate    = self.sigmoid(self.t4v_gate_lin(self.drop(self.relu(t_mean)))).unsqueeze(1) #[batch, 1, feat_size]

        # key, query, value
        v_trans = self.v_lin(self.drop(self.relu(v)))
        t_trans = self.t_lin(self.drop(self.relu(t)))

        # mask all padding object/word features
        v_trans = v_trans * v_mask.unsqueeze(2)
        t_trans = t_trans * t_mask.unsqueeze(2)

        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        t_k, t_q, t_v = torch.split(t_trans, t_trans.size(2) // 3, dim=2)

        def shape_gate(x)   : return x.view(batch_size, self.num_head, dim_per_head).unsqueeze(2)
        def shape(x)        : return x.view(batch_size, -1, self.num_head, dim_per_head).transpose(1, 2)
        def unshape(x)      : return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head*dim_per_head)

        # transform features
        v_k = shape(v_k)  # (batch_size, num_head, num_obj, dim_per_head)
        v_q = shape(v_q)  # (batch_size, num_head, num_obj, dim_per_head)
        v_v = shape(v_v)  # (batch_size, num_head, num_obj, dim_per_head)

        t_k = shape(t_k)  # (batch_size, num_head, max_len, dim_per_head)
        t_q = shape(t_q)  # (batch_size, num_head, max_len, dim_per_head)
        t_v = shape(t_v)  # (batch_size, num_head, max_len, dim_per_head)

        # apply conditioned gate
        new_vq = (1 + shape_gate(t4v_gate)) * v_q   # (batch_size, nhead, num_obj, dim_per_head)
        new_vk = (1 + shape_gate(t4v_gate)) * v_k   # (batch_size, nhead, num_obj, dim_per_head)
        new_tq = (1 + shape_gate(v4t_gate)) * t_q   # (batch_size, nhead, max_len, dim_per_head)
        new_tk = (1 + shape_gate(v4t_gate)) * t_k   # (batch_size, nhead, max_len, dim_per_head)

        # multi-head attention
        v2v = torch.matmul(new_vq, new_vk.transpose(2,3)) / math.sqrt(dim_per_head)
        t2t = torch.matmul(new_tq, new_tk.transpose(2,3)) / math.sqrt(dim_per_head)

        # masking
        v_mask = (v_mask == 0).unsqueeze(1).unsqueeze(1).expand_as(v2v)
        t_mask = (t_mask == 0).unsqueeze(1).unsqueeze(1).expand_as(t2t)

        # set padding object/word attention to negative infinity & normalized by square root of hidden dimension
        v2v = v2v.masked_fill(v_mask, -float('inf'))   # (batch_size, num_head, num_obj, max_len)
        t2t = t2t.masked_fill(t_mask, -float('inf'))   # (batch_size, num_head, max_len, num_obj)

        # attention score
        dyIntraMAF_v2v = F.softmax(v2v, dim=-1).type_as(v2v)
        dyIntraMAF_t2t = F.softmax(t2t, dim=-1).type_as(t2t)

        v_update = unshape(torch.matmul(dyIntraMAF_v2v, v_v))
        t_update = unshape(torch.matmul(dyIntraMAF_t2t, t_v))

        # update
        updated_v = self.v_output(self.drop(v + v_update))
        updated_t = self.t_output(self.drop(t + t_update))

        return updated_v, updated_t


class TokenClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.1):
        super(TokenClassifier, self).__init__()
        
        self.layer_norm = nn.LayerNorm(input_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(input_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):

        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.gelu(x)
        x = self.dropout(x)        
        x = self.output_layer(x)
                
        return x
    
class RED_DOT_DYNAMIC(nn.Module):
    def __init__(
        self,
        device,
        emb_dim=512,
        tf_layers=1,
        tf_head=2,
        tf_dim=128,
        activation="gelu",
        dropout=0.1,
        use_features=["images", "texts"],
        pre_norm=True,
        num_classes=1,
        skip_tokens=0,
        use_evidence=1,
        use_neg_evidence=1,
        model_version="baseline",
        fuse_evidence=[False]
    ):

        super().__init__()

        self.use_features = use_features
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.fuse_evidence = fuse_evidence
        
        self.device = device
        
        self.cls_token = nn.Parameter(torch.randn(self.emb_dim))
        self.cls_token.requires_grad = True
        
        self.skip_tokens = skip_tokens
        self.use_evidence = use_evidence
        self.use_neg_evidence = use_neg_evidence
        
        self.DynamicFusion = SingleBlock(drop=0.1)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=tf_head,
                dim_feedforward=tf_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=pre_norm,                
            ),
            num_layers=tf_layers,
        )
        
        self.num_evidence_tokens = len(use_features) * (use_evidence + use_neg_evidence) * len(fuse_evidence)
        
        self.binary_classifier = TokenClassifier(self.emb_dim, self.emb_dim)
        
        self.model_version = model_version
        
        if "guided" not in self.model_version and self.model_version != "baseline":
            self.evidence_classifier = TokenClassifier(self.emb_dim, self.emb_dim)
            
        if "two_transformers" in self.model_version:
            self.transformer2 = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=tf_head,
                    dim_feedforward=tf_dim,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    norm_first=pre_norm,                
                ),
                num_layers=tf_layers,
            )
        
        
    def forward(self, images, texts, X_all, cross_attention_module, inference=False, X_labels=None, eval_verite = False):
        device = images.device 
        batch_size, _ = images.size() 
        # Calculate v_mask (object mask)
        v_mask = torch.ones(batch_size, 1, device = device).float()  # Assuming all images are valid
        # Calculate t_mask (text mask)
        t_mask = torch.ones(batch_size, 1, device = device).float()  # Assuming all texts are valid 
        x = self.DynamicFusion(images, texts, v_mask, t_mask).unsqueeze(1)
        if X_all is not None:
            X_all = X_all.to(device)
            all_attentions = []
            for i in range(X_all.shape[1]):
                evidence = X_all[:, i, :].unsqueeze(1)
                # print('x.shape is ', x.shape)
                # print('evidence shape is ', evidence.shape)
                attention_output = cross_attention_module(x, evidence, evidence)
                all_attentions.append(attention_output)
            x = torch.cat([x, X_all], axis=1)   
            x = torch.cat([x] + all_attentions, dim=1)
        #only for Verite benchmark
        if eval_verite:
            total_tokens = 9
            if x.shape[1] < total_tokens:
                pad_zeros = torch.zeros((x.shape[0], total_tokens - x.shape[1], x.shape[-1])).to(device)
                x = torch.concat([x, pad_zeros], axis=1)

        b_size = x.shape[0]
        num_tokens = x.shape[1]
        dimensions = x.shape[2]
        y_relevance = None
        
        if not inference and "dual_stage" in self.model_version and self.model_version != "baseline":
            
            # Forward x with all evidence
            cls_token = self.cls_token.expand(b_size, 1, -1)        
            x2 = torch.cat((cls_token, x), dim=1)    
            x2 = self.transformer(x2) 
            
            if "guided" in self.model_version:
                scale_factor = torch.tensor(dimensions, dtype=torch.float32)
                y_relevance = torch.bmm(x2, x2.transpose(1,2))           
                y_relevance = y_relevance[:,0,:][:, -self.num_evidence_tokens:]
                y_relevance = y_relevance / scale_factor
                
            else:
                results = []
                for i in range(self.num_evidence_tokens):
                    token_result = self.evidence_classifier(x2[:, self.skip_tokens+i+1:self.skip_tokens+i+2, :])                
                    results.append(token_result)     
                y_relevance = torch.stack(results, dim=1).view(b_size, -1)
            
            # Teacher enforcing only during training
            mask = torch.zeros(b_size, num_tokens, dtype=torch.bool)
            
            if not self.training:           
                x_labels = torch.sigmoid(y_relevance)
                x_labels = torch.round(x_labels)
            
            # Mask labeled/predicted negative evidence
            positions_to_mask = x_labels == 0
            mask[:, -self.num_evidence_tokens:] = positions_to_mask[:, -self.num_evidence_tokens:]
            mask = mask.to(self.device)
            x = x * (1 - mask.to(torch.float32).unsqueeze(2))
                            
        cls_token = self.cls_token.expand(b_size, 1, -1)        
        x = torch.cat((cls_token, x), dim=1)
        
        if "two_transformers" in self.model_version:
            x = self.transformer2(x)
            
        else:
            x = self.transformer(x) 
        
        x_truth = x[:,0,:]
        y_truth = self.binary_classifier(x_truth)
        
        if not inference and "dual_stage" not in self.model_version and self.model_version != "baseline": 
                
            if "guided" in self.model_version: 
                scale_factor = torch.tensor(dimensions, dtype=torch.float32)

                y_relevance = torch.bmm(x, x.transpose(1,2))           
                y_relevance = y_relevance[:,0,:][:, -self.num_evidence_tokens:]
                y_relevance = y_relevance / scale_factor
            
            else:
                results = []
                for i in range(self.num_evidence_tokens):                
                    token_result = self.evidence_classifier(x[:, self.skip_tokens+i+1:self.skip_tokens+i+2, :])                
                    results.append(token_result)    

                y_relevance = torch.stack(results, dim=1).view(b_size, -1) 

                    
        return y_truth, y_relevance
    




"""
class CrossAttentionWithCLIP(nn.Module):
    def __init__(self, model, device, embed_dim, num_heads=4, dropout=0.2):
        super(CrossAttentionWithCLIP, self).__init__()
        self.clip_model = model
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # self.embed_dim = self.clip_model.config.projection_dim  # Use projection_dim for CLIP
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(self.embed_dim)
        self.layernorm2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.activation = nn.GELU()

    def forward(self, query, key, value):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        attn_output = self.layernorm1(query + attn_output)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(attn_output))))
        ff_output = self.dropout(ff_output)
        ff_output = self.layernorm2(attn_output + ff_output)
        return ff_output.transpose(0, 1)
"""
"""
class CrossAttention(nn.Module):
    def __init__(self, device, embed_dim, num_heads=4, dropout=0.2):
        super(CrossAttention, self).__init__()
        self.device = device
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(self.embed_dim)
        self.layernorm2 = nn.LayerNorm(self.embed_dim)
        self.batchnorm1 = nn.BatchNorm1d(self.embed_dim)
        self.batchnorm2 = nn.BatchNorm1d(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.activation = nn.GELU()

    def forward(self, query, key, value):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        attn_output = self.layernorm1(query + attn_output)
        attn_output = attn_output.transpose(1, 2)  # Change to (batch_size, embed_dim, sequence_length) for BatchNorm1d
        attn_output = self.batchnorm1(attn_output)
        attn_output = attn_output.transpose(1, 2)  # Change back to (sequence_length, batch_size, embed_dim)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(attn_output))))
        ff_output = self.dropout(ff_output)
        ff_output = self.layernorm2(attn_output + ff_output)
        ff_output = ff_output.transpose(1, 2)  # Change to (batch_size, embed_dim, sequence_length) for BatchNorm1d
        ff_output = self.batchnorm2(ff_output)
        ff_output = ff_output.transpose(1, 2)  # Change back to (sequence_length, batch_size, embed_dim)
        return ff_output.transpose(0, 1)
"""

class StackedCrossAttention(nn.Module):
    def __init__(self, device, embed_dim, num_heads=4, dropout=0.2, num_layers=2):
        super(StackedCrossAttention, self).__init__()
        self.layers = nn.ModuleList([CrossAttention(device, embed_dim, num_heads, dropout) for _ in range(num_layers)])
    
    def forward(self, query, key, value):
        for layer in self.layers:
            query = layer(query, key, value)
        return query