import torch
import torch.nn as nn

class DINOv2SelfAttentionModel(nn.Module):
    """
    Siamese-style model using a shared DINOv2 backbone per image via PyTorch Hub,
    followed by a stack of self-attention blocks over the concatenated embeddings,
    and a final linear layer to produce a logit.
    """
    def __init__(self,
                 model_name: str = 'dinov2_vits14',
                 num_layers: int = 3,
                 num_heads: int = 8,
                 attn_dropout: float = 0.0):
        super(DINOv2SelfAttentionModel, self).__init__()
        # Load pretrained DINOv2 backbone from PyTorch Hub
        # Available names include 'dinov2_vits14', 'dinov2_vitb14',
        # 'dinov2_vitl14', 'dinov2_vitg14'
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=True
        ).eval()
        # Determine feature dimension from backbone
        feat_dim = getattr(self.backbone, 'embed_dim', None)
        if feat_dim is None:
            # fallback if embed_dim not set, infer from head weight
            feat_dim = self.backbone.head.weight.shape[1]
        # After concatenation, attention dimension = 2 * feat_dim
        attn_dim = feat_dim * 2

        # Define a sequence of Transformer encoder layers (self-attention blocks)
        self.attn_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=attn_dim,
                nhead=num_heads,
                dropout=attn_dropout,
                batch_first=False
            ) for _ in range(num_layers)
        ])

        # Final classifier: embedding dimension after attention -> logit
        self.classifier = nn.Linear(attn_dim, 1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # img1, img2: [B, 3, H, W]
        # Extract features for each image via hub model
        f1 = self.backbone(img1)  # [B, feat_dim]
        f2 = self.backbone(img2)  # [B, feat_dim]

        # Concatenate features along channel dim: [B, 2*feat_dim]
        x = torch.cat([f1, f2], dim=1)

        # Apply each self-attention block: reshape -> [1, B, attn_dim]
        for block in self.attn_blocks:
            x = block(x.unsqueeze(0)).squeeze(0)  # [B, attn_dim]

        # Final linear layer to produce logit per pair
        logit = self.classifier(x).squeeze(1)  # [B]
        return logit


def main():
    # Example usage to verify DINOv2 loading and concatenation-based attention
    model = DINOv2SelfAttentionModel(model_name='dinov2_vits14', num_layers=3)
    model.eval()

    # Dummy inputs: batch of 2 RGB images of size 224x224
    batch_size = 2
    img1 = torch.randn(batch_size, 3, 224, 224)
    img2 = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        outputs = model(img1, img2)

    # Print results for verification
    print(f"Input batch size: {batch_size}")
    print(f"Output logits shape: {outputs.shape}")
    print(f"Output logits: {outputs}")

if __name__ == "__main__":
    main()
