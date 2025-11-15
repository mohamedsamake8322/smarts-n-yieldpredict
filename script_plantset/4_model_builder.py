#!/usr/bin/env python3
"""
model_builder.py - Architecture de modèle multimodal vision-langage
Définit le modèle multimodal (Vision Encoder + Text Encoder + Fusion)
Inspiré de Florence-2, CLIP, BLIP, Qwen-VL, FLAVA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Tuple, Optional, Any, Union
import math
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionEncoder(nn.Module):
    """
    Encodeur d'images basé sur ResNet/EfficientNet/ViT
    """
    
    def __init__(self,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 output_dim: int = 512,
                 dropout: float = 0.1):
        """
        Args:
            backbone: Architecture (resnet50, efficientnet_b0, vit_base_patch16)
            pretrained: Utiliser des poids pré-entraînés
            freeze_backbone: Geler les couches du backbone
            output_dim: Dimension de sortie
            dropout: Taux de dropout
        """
        super().__init__()
        self.backbone_name = backbone
        self.output_dim = output_dim
        
        # Charger le backbone
        self.backbone = self._create_backbone(backbone, pretrained)
        
        # Geler si demandé
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Adapter la dimension de sortie
        self.feature_dim = self._get_feature_dim()
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def _create_backbone(self, backbone: str, pretrained: bool):
        """Crée le backbone d'images"""
        if backbone.startswith("resnet"):
            if backbone == "resnet18":
                model = tvmodels.resnet18(pretrained=pretrained)
                model.fc = nn.Identity()
            elif backbone == "resnet50":
                model = tvmodels.resnet50(pretrained=pretrained)
                model.fc = nn.Identity()
            elif backbone == "resnet101":
                model = tvmodels.resnet101(pretrained=pretrained)
                model.fc = nn.Identity()
            else:
                raise ValueError(f"ResNet non supporté: {backbone}")
        
        elif backbone.startswith("efficientnet"):
            try:
                import timm
                model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            except ImportError:
                logger.warning("timm non disponible, fallback vers ResNet50")
                model = tvmodels.resnet50(pretrained=pretrained)
                model.fc = nn.Identity()
        
        elif backbone.startswith("vit"):
            try:
                import timm
                model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            except ImportError:
                logger.warning("timm non disponible, fallback vers ResNet50")
                model = tvmodels.resnet50(pretrained=pretrained)
                model.fc = nn.Identity()
        
        else:
            raise ValueError(f"Backbone non supporté: {backbone}")
        
        return model
    
    def _get_feature_dim(self) -> int:
        """Retourne la dimension des features du backbone"""
        if hasattr(self.backbone, 'num_features'):
            return self.backbone.num_features
        elif hasattr(self.backbone, 'head'):
            return self.backbone.head.in_features
        else:
            # Fallback pour ResNet
            return 2048 if "resnet50" in self.backbone_name else 512
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, height, width]
        Returns:
            features: [batch_size, output_dim]
        """
        # Extraire les features
        features = self.backbone(images)
        
        # Projeter vers la dimension de sortie
        features = self.projection(features)
        
        return features

class TextEncoder(nn.Module):
    """
    Encodeur de texte basé sur BERT/RoBERTa/LLaMA
    """
    
    def __init__(self,
                 model_name: str = "microsoft/DialoGPT-medium",
                 output_dim: int = 512,
                 freeze_backbone: bool = False,
                 dropout: float = 0.1):
        """
        Args:
            model_name: Nom du modèle HuggingFace
            output_dim: Dimension de sortie
            freeze_backbone: Geler les couches du backbone
            dropout: Taux de dropout
        """
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Charger le modèle de texte
        try:
            self.backbone = AutoModel.from_pretrained(model_name)
            self.config = AutoConfig.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Erreur chargement {model_name}: {e}")
            # Fallback vers un modèle simple
            self.backbone = self._create_simple_text_encoder()
            self.config = type('Config', (), {'hidden_size': 512})()
        
        # Geler si demandé
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Adapter la dimension de sortie
        hidden_size = getattr(self.config, 'hidden_size', 512)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def _create_simple_text_encoder(self):
        """Crée un encodeur de texte simple en fallback"""
        class SimpleTextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(10000, 512)
                self.lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
                self.pool = nn.AdaptiveAvgPool1d(1)
            
            def forward(self, input_ids, attention_mask=None):
                # Embedding
                x = self.embedding(input_ids)
                
                # LSTM
                lstm_out, _ = self.lstm(x)
                
                # Pooling global
                pooled = self.pool(lstm_out.transpose(1, 2)).squeeze(-1)
                
                return type('Output', (), {'last_hidden_state': pooled})()
        
        return SimpleTextEncoder()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            features: [batch_size, output_dim]
        """
        # Encoder le texte
        if hasattr(self.backbone, 'forward'):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            # Utiliser le dernier état caché
            if hasattr(outputs, 'last_hidden_state'):
                text_features = outputs.last_hidden_state
            else:
                text_features = outputs
        else:
            # Fallback
            text_features = self.backbone(input_ids, attention_mask)
        
        # Pooling global si nécessaire
        if len(text_features.shape) == 3:  # [batch, seq, hidden]
            if attention_mask is not None:
                # Pooling avec attention mask
                attention_mask = attention_mask.unsqueeze(-1).float()
                text_features = (text_features * attention_mask).sum(1) / attention_mask.sum(1)
            else:
                # Pooling simple
                text_features = text_features.mean(1)
        
        # Projeter vers la dimension de sortie
        text_features = self.projection(text_features)
        
        return text_features

class CrossAttentionFusion(nn.Module):
    """
    Module de fusion par attention croisée entre vision et texte
    Inspiré de CLIP et BLIP
    """
    
    def __init__(self,
                 vision_dim: int = 512,
                 text_dim: int = 512,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            vision_dim: Dimension des features visuelles
            text_dim: Dimension des features textuelles
            hidden_dim: Dimension cachée
            num_heads: Nombre de têtes d'attention
            num_layers: Nombre de couches
            dropout: Taux de dropout
        """
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Projections vers l'espace commun
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Couches d'attention croisée
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Projection finale
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, vision_dim]
            text_features: [batch_size, text_dim]
        Returns:
            fused_features: [batch_size, hidden_dim]
        """
        # Projeter vers l'espace commun
        v = self.vision_proj(vision_features)  # [batch, hidden_dim]
        t = self.text_proj(text_features)      # [batch, hidden_dim]
        
        # Attention croisée
        for layer in self.cross_attention_layers:
            v, t = layer(v, t)
        
        # Fusion finale
        fused = torch.cat([v, t], dim=-1)  # [batch, hidden_dim * 2]
        fused = self.fusion_proj(fused)    # [batch, hidden_dim]
        
        return fused

class CrossAttentionLayer(nn.Module):
    """Couche d'attention croisée"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, vision: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention pour chaque modalité
        v_attn, _ = self.self_attn(vision.unsqueeze(1), vision.unsqueeze(1), vision.unsqueeze(1))
        t_attn, _ = self.self_attn(text.unsqueeze(1), text.unsqueeze(1), text.unsqueeze(1))
        
        v = self.norm1(vision + v_attn.squeeze(1))
        t = self.norm1(text + t_attn.squeeze(1))
        
        # Cross-attention
        v_cross, _ = self.cross_attn(v.unsqueeze(1), t.unsqueeze(1), t.unsqueeze(1))
        t_cross, _ = self.cross_attn(t.unsqueeze(1), v.unsqueeze(1), v.unsqueeze(1))
        
        v = self.norm2(v + v_cross.squeeze(1))
        t = self.norm3(t + t_cross.squeeze(1))
        
        # FFN
        v_ffn = self.ffn(v)
        t_ffn = self.ffn(t)
        
        v = self.norm2(v + self.dropout(v_ffn))
        t = self.norm3(t + self.dropout(t_ffn))
        
        return v, t

class MultimodalClassifier(nn.Module):
    """
    Classifieur multimodal complet
    Combine vision, texte et fusion pour la classification
    """
    
    def __init__(self,
                 num_classes: int,
                 vision_backbone: str = "resnet50",
                 text_model: str = "microsoft/DialoGPT-medium",
                 vision_dim: int = 512,
                 text_dim: int = 512,
                 fusion_dim: int = 256,
                 num_attention_heads: int = 8,
                 num_attention_layers: int = 2,
                 dropout: float = 0.1,
                 freeze_vision: bool = False,
                 freeze_text: bool = False):
        """
        Args:
            num_classes: Nombre de classes de maladies
            vision_backbone: Architecture du backbone visuel
            text_model: Modèle de texte HuggingFace
            vision_dim: Dimension des features visuelles
            text_dim: Dimension des features textuelles
            fusion_dim: Dimension de la fusion
            num_attention_heads: Nombre de têtes d'attention
            num_attention_layers: Nombre de couches d'attention
            dropout: Taux de dropout
            freeze_vision: Geler l'encodeur visuel
            freeze_text: Geler l'encodeur textuel
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Encodeurs
        self.vision_encoder = VisionEncoder(
            backbone=vision_backbone,
            output_dim=vision_dim,
            freeze_backbone=freeze_vision,
            dropout=dropout
        )
        
        self.text_encoder = TextEncoder(
            model_name=text_model,
            output_dim=text_dim,
            freeze_backbone=freeze_text,
            dropout=dropout
        )
        
        # Fusion
        self.fusion = CrossAttentionFusion(
            vision_dim=vision_dim,
            text_dim=text_dim,
            hidden_dim=fusion_dim,
            num_heads=num_attention_heads,
            num_layers=num_attention_layers,
            dropout=dropout
        )
        
        # Têtes de classification
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Tête de génération de texte (optionnelle)
        self.text_generator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, text_dim)
        )
    
    def forward(self, images: torch.Tensor, texts: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [batch_size, 3, height, width]
            texts: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            Dict avec logits, features, etc.
        """
        # Encoder les modalités
        vision_features = self.vision_encoder(images)  # [batch, vision_dim]
        text_features = self.text_encoder(texts, attention_mask)  # [batch, text_dim]
        
        # Fusion
        fused_features = self.fusion(vision_features, text_features)  # [batch, fusion_dim]
        
        # Classification
        logits = self.classifier(fused_features)  # [batch, num_classes]
        
        # Génération de texte (pour description)
        text_embedding = self.text_generator(fused_features)  # [batch, text_dim]
        
        return {
            'logits': logits,
            'vision_features': vision_features,
            'text_features': text_features,
            'fused_features': fused_features,
            'text_embedding': text_embedding
        }
    
    def get_embeddings(self, images: torch.Tensor, texts: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Retourne seulement les embeddings fusionnés"""
        with torch.no_grad():
            vision_features = self.vision_encoder(images)
            text_features = self.text_encoder(texts, attention_mask)
            fused_features = self.fusion(vision_features, text_features)
        return fused_features

def create_multimodal_model(num_classes: int,
                           vision_backbone: str = "resnet50",
                           text_model: str = "microsoft/DialoGPT-medium",
                           **kwargs) -> MultimodalClassifier:
    """
    Fonction utilitaire pour créer un modèle multimodal
    """
    return MultimodalClassifier(
        num_classes=num_classes,
        vision_backbone=vision_backbone,
        text_model=text_model,
        **kwargs
    )

if __name__ == "__main__":
    # Test du modèle
    import argparse
    
    parser = argparse.ArgumentParser(description="Test du MultimodalClassifier")
    parser.add_argument("--num-classes", type=int, default=133, help="Nombre de classes")
    parser.add_argument("--batch-size", type=int, default=4, help="Taille du batch")
    parser.add_argument("--image-size", type=int, default=224, help="Taille des images")
    parser.add_argument("--text-length", type=int, default=128, help="Longueur du texte")
    
    args = parser.parse_args()
    
    # Créer le modèle
    model = create_multimodal_model(
        num_classes=args.num_classes,
        vision_backbone="resnet50",
        text_model="microsoft/DialoGPT-medium"
    )
    
    # Test forward pass
    print("Test du modèle...")
    batch_size = args.batch_size
    images = torch.randn(batch_size, 3, args.image_size, args.image_size)
    texts = torch.randint(0, 1000, (batch_size, args.text_length))
    
    with torch.no_grad():
        outputs = model(images, texts)
    
    print(f"Images shape: {images.shape}")
    print(f"Texts shape: {texts.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Vision features shape: {outputs['vision_features'].shape}")
    print(f"Text features shape: {outputs['text_features'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParamètres totaux: {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,}")
    
    print("Test terminé!")
