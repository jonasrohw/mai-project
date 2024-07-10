import torch
import torch.nn as nn
import clip

class TokenClassifier(nn.Module):
    """
    TokenClassifier model used for classifying individual tokens.

    This model uses layer normalization, GELU activation, dropout,
    and linear layers to perform classification on input tokens.
    """
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
    
class RED_DOT(nn.Module):
    """
    RED_DOT model, a transformer-based model for binary classification and relevance scoring
    using multiple evidence sources.

    This model integrates various components such as token classification, evidence handling,
    and multi-layer transformer encoders to achieve its objectives.
    """
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
        
        
    def forward(self, x, inference=False, x_labels=None):

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


class CrossAttention(nn.Module):
    """
    CrossAttention model using multi-head attention mechanism.

    This model includes layer normalization, dropout, and linear layers to process the
    input query, key, and value tensors using multi-head attention.
    """
    def __init__(self, device, embed_dim, num_heads=4, dropout=0.2):
        super(CrossAttention, self).__init__()
        self.device = device
        self.num_heads = num_heads
        self.embed_dim = embed_dim
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
multihead attention model with more methods implemented for mitigating overfitting

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
    """
    StackedCrossAttention model for capturing complex relationships using multiple CrossAttention layers.

    This model stacks several CrossAttention layers to process input query, key, and value tensors
    through multiple layers of multi-head attention.
    """
    def __init__(self, device, embed_dim, num_heads=4, dropout=0.2, num_layers=2):
        super(StackedCrossAttention, self).__init__()
        self.layers = nn.ModuleList([CrossAttention(device, embed_dim, num_heads, dropout) for _ in range(num_layers)])
    
    def forward(self, query, key, value):
        for layer in self.layers:
            query = layer(query, key, value)
        return query
