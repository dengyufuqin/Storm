import torch
import torch.nn as nn
from torchvision.transforms import Resize, CenterCrop, Compose, InterpolationMode

class Dinov2Encoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_backbone: bool,
        device: str,
        img_size: int = 224,
    ):
        super().__init__()

        # Load DINO/DINOv2 pretrained backbone
        self.dino_version, _ = model_name.split("_", 1)
        self.dino = torch.hub.load(
            f"facebookresearch/{self.dino_version}:main", model_name
        ).to(device)

        # Optionally freeze backbone
        if freeze_backbone:
            for p in self.dino.parameters():
                p.requires_grad = False
        self.dino.eval()

        # Precompute patch map size and embedding dim
        dummy = torch.zeros(1, 3, img_size, img_size).to(device)
        with torch.no_grad():
            if self.dino_version == "dinov2":
                feats = self.dino.forward_features(dummy)
                tokens = feats["x_norm_patchtokens"]  # [1, num_patches, embed_dim]
            else:
                raw = self.dino.get_intermediate_layers(dummy)[0][0]
                tokens = raw[:, 1:, :]
        _, num_patches, embed_dim = tokens.shape
        dim_hw = int(num_patches ** 0.5)
        assert dim_hw * dim_hw == num_patches, "patch count not a square"

        # Store config
        self.patch_dim = dim_hw
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.device = device

        # Simple preprocessing: resize + center crop
        self.img_preprocessor = Compose([
            Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(img_size),
        ])
        
    @torch.no_grad()
    def _encode_with_dino(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] normalized via dataset pipeline
        Returns: [B, embed_dim, patch_dim, patch_dim]
        """
        # Resize and crop
        x = self.img_preprocessor(x)
        # Extract patch tokens
        if self.dino_version == "dinov2":
            feats = self.dino.forward_features(x)
            tokens = feats["x_norm_patchtokens"]
        else:
            raw = self.dino.get_intermediate_layers(x)[0][0]
            tokens = raw[:, 1:, :]
        # Reshape to spatial map
        B, N, D = tokens.shape
        h_w = self.patch_dim
        # print(f"Patch dim: {h_w}, Embed dim: {D}")
        return tokens.transpose(1, 2).reshape(B, D, h_w, h_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return raw patch embeddings from DINO:
        Tensor [B, embed_dim, patch_dim, patch_dim]
        """
        return self._encode_with_dino(x)

