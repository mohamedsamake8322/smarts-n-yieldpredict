"""
model_def.py
------------
Définitions des classes du modèle DINOv2 multitâche, extraites telles
quelles du script d'entraînement (Dinov2_v6_smartagri_fusionne_KAGGLE).

À placer à la racine du projet et uploader sur le repo Hugging Face
mohamedsamake8322/maladie-plantes-dinov2, pour que model_core.py puisse
les importer au chargement du checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Multitask(nn.Module):
    """
    DINOv2 ViT-L/14 avec :
      Stage 0 → backbone gelé, têtes actives
      Stage 1 → last N blocs + norm
      Stage 2 → backbone entier
    """
    def __init__(self, backbone_name, num_classes, num_crops,
                 num_categories, embed_dim, total_blocks=24):
        super().__init__()
        print(f"  📥 Chargement {backbone_name} depuis torch.hub...")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', backbone_name, pretrained=True)
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()

        self.embed_dim    = embed_dim
        self.total_blocks = total_blocks
        self._freeze_all_backbone()

        # Bottleneck partagé
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.head_main = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        self.head_crop     = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Linear(128, num_crops))
        self.head_category = nn.Sequential(
            nn.Linear(512, 64),  nn.GELU(), nn.Linear(64, num_categories))

        # Uncertainty weighting (Kendall & Gal, 2018) : log-variance appris
        # par tâche (main, crop, category) pour pondérer automatiquement la
        # loss multi-tâche, à la place de poids fixes LOSS_WEIGHTS. Initialisé
        # à 0 (précision=1 au départ, comme les poids fixes actuels).
        self.log_vars = nn.Parameter(torch.zeros(3))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

        total  = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  ✅ {total/1e6:.1f}M params | {frozen/1e6:.1f}M gelés (stage 0)")

    def _freeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n):
        blocks = getattr(self.backbone, 'blocks', None)
        if blocks is None:
            self.unfreeze_full(); return
        start = max(0, len(blocks) - n)
        for i, blk in enumerate(blocks):
            if i >= start:
                for p in blk.parameters(): p.requires_grad = True
        for name, mod in self.backbone.named_modules():
            if 'norm' in name.lower() and isinstance(mod, (nn.LayerNorm, nn.BatchNorm1d)):
                for p in mod.parameters(): p.requires_grad = True
        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        free   = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"  🔓 Stage 1 — last {n} blocs | libre={free/1e6:.1f}M gelé={frozen/1e6:.1f}M")

    def unfreeze_full(self):
        for p in self.backbone.parameters(): p.requires_grad = True
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"  🔓 Stage 2 — backbone entier dégelé ({total/1e6:.1f}M)")

    def extract_backbone_feat(self, x):
        """CLS token (vision globale) + moyenne des patch tokens (zones de
        symptômes : taches, lésions, textures) -> représentation plus riche
        que le CLS seul."""
        if hasattr(self.backbone, 'forward_features'):
            out = self.backbone.forward_features(x)
            if isinstance(out, dict):
                cls_tok   = out.get('x_norm_clstoken')
                patch_tok = out.get('x_norm_patchtokens')
                if cls_tok is not None and patch_tok is not None:
                    feat = cls_tok + patch_tok.mean(dim=1)
                elif cls_tok is not None:
                    feat = cls_tok
                else:
                    feat = next(iter(out.values()))
                    if feat.dim() == 3: feat = feat[:, 0]
            else:
                feat = out
                if feat.dim() == 3:
                    feat = feat[:, 0] + feat[:, 1:].mean(dim=1)
        else:
            feat = self.backbone(x)
            if feat.dim() == 3:
                feat = feat[:, 0] + feat[:, 1:].mean(dim=1)
        return feat

    def forward_shared(self, x):
        """Embedding partagé (512-d) après shared_proj — utilisé pour les
        têtes de classification ET pour la détection OOD (Mahalanobis)."""
        return self.shared_proj(self.extract_backbone_feat(x))

    def forward(self, x):
        shared = self.forward_shared(x)
        return {
            'main':      self.head_main(shared),
            'crop':      self.head_crop(shared),
            'category':  self.head_category(shared),
            'embedding': shared,  # utilisé par la loss contrastive supervisée
        }


class CosineLinear(nn.Module):
    """Couche de classification "cosine" (façon LUCIR) : normalise features
    ET poids avant le produit scalaire, puis applique un facteur d'échelle
    scalaire appris `s`. Volontairement SANS biais additif : un biais par
    classe ajouté avant la mise à l'échelle réintroduirait exactement le
    biais de norme que la normalisation cosine est censée supprimer entre
    anciennes et nouvelles classes.
    """
    def __init__(self, in_features, out_features, scale_init=10.0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.s = nn.Parameter(torch.tensor(float(scale_init)))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x):
        x_n = F.normalize(x, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        return self.s * F.linear(x_n, w_n)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, cosine=True'
