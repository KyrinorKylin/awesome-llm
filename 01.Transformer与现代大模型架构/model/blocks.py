from src.attention_rope import *
from src.ffn import *
from src.ffn_pro import *
from src.norm import RMSNorm

class DecoderBlock(nn.Module):
    def __init__(
        self, d_model, d_ff, num_heads, num_groups, d_c_kv, d_c_q, d_rope, dropout, 
        attention: Literal['MHA','MQA','GQA','MLA'], 
        mla: Literal['ffn','glu', 'topk','shared','finegrained','ds'],
        act: Literal['relu', 'gelu', 'silu']
        eps=1e-6
        ):

        super().__init__()
        attn = {
            'MHA': MHA(d_model, num_heads, dropout),
            'MQA': MQA(d_model, num_heads, dropout),
            'GQA': GQA(d_model, num_heads, num_groups, dropout),
            'MLA': MLA(d_model, num_heads, d_c_kv, d_c_q, d_rope, dropout)
        }

        self.norm = RMSNorm(d_model, eps)
        self.mha = attn[attention]
        self.mla = ...