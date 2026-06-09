"""
FCmodel — 全连接模型, 统一输入 [bs, seq_len, feature_dim]
Backbone + Classifier 分离, 支持换头/冻结/提取
"""
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class MLPBlock(nn.Module):
    """RMSNorm + LinearUp + ReLU + LinearDown + residual"""
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.up = nn.Linear(hidden_size, hidden_size * 2)
        self.act = nn.ReLU()
        self.down = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.up(x)
        x = self.act(x)
        x = self.down(x)
        return x + residual


class FCmodel(nn.Module):
    """统一输入 [bs, seq_len, feature_dim], 输出 [bs, num_class]"""
    def __init__(self, feature_dim=4, num_class=4, hidden_size=256, num_layers=4, seq_len=30):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_class = num_class

        # Embed: flatten + project
        self.embed = nn.Linear(seq_len * feature_dim, hidden_size)
        # Backbone: MLP blocks with residual
        self.layers = nn.ModuleList([MLPBlock(hidden_size) for _ in range(num_layers)])
        # Classifier
        self.classifier = nn.Linear(hidden_size, num_class, bias=False)

    def forward(self, x):
        # x: [bs, seq_len, feature_dim] = [bs, 30, 4]
        # 转置为 [bs, 4, 30] 再 flatten → 和旧 concat 顺序一致
        x = x.transpose(1, 2).reshape(x.size(0), -1)  # [bs, 120]
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    # ─── Backbone / Classifier 分离工具 ───
    def backbone_state_dict(self):
        """返回 backbone (embed + layers) 的 state_dict, 不含 classifier"""
        full = self.state_dict()
        return {k: v for k, v in full.items() if not k.startswith('classifier.')}

    def load_backbone_state_dict(self, backbone_state):
        """加载 backbone 权重, classifier 保持当前"""
        own = self.state_dict()
        for k, v in backbone_state.items():
            if k in own and not k.startswith('classifier.'):
                own[k] = v
        self.load_state_dict(own, strict=False)

    def freeze_backbone(self, unfreeze_last=0):
        """冻结 backbone. unfreeze_last: 解冻最后 N 层 mlp block"""
        freeze_until = self.num_layers - unfreeze_last  # 前 freeze_until 层冻结
        for name, param in self.named_parameters():
            if name.startswith('classifier.') or name.startswith('embed.'):
                param.requires_grad = not name.startswith('embed.')
                continue
            # layers.N.* → 提取 N
            if name.startswith('layers.'):
                layer_idx = int(name.split('.')[1])
                if layer_idx >= freeze_until:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻全部"""
        for param in self.parameters():
            param.requires_grad = True

    @staticmethod
    def extract_backbone_from_pt(pt_path, map_location='cpu'):
        """从已保存 .pt 提取 backbone 权重 (兼容旧 head→embed 改名)"""
        state = torch.load(pt_path, map_location=map_location)
        result = {}
        for k, v in state.items():
            if k.startswith('classifier.'):
                continue
            # 旧模型 head.* → 新模型 embed.*
            if k.startswith('head.'):
                k = 'embed.' + k[5:]
            result[k] = v
        return result
