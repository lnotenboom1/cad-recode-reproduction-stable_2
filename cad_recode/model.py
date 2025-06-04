# cad_recode/model.py
"""
CAD-Recode model: point-cloud encoder + causal language model decoder.

This module defines the core model class `CADRecodeModel`, combining:

1. A `PointCloudProjector` that transforms a 3D point cloud into a sequence
   of query token embeddings using Fourier positional encoding.
2. A HuggingFace causal decoder (e.g. Qwen-1.5B) that processes the tokenized
   CadQuery code, optionally using the point-token embeddings as a prefix.
3. A loss function for training, using standard CausalLM language modeling.

The model supports:
- freezing the decoder weights (for memory-limited training)
- generating sequences via `.prepare_prefix()` for use with `generate()`
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List, Union


# ---------------------------------------------------------------------------
#  Positional Encoding + MLP Projector for point clouds
# ---------------------------------------------------------------------------
class PointCloudProjector(nn.Module):
    """
    Maps a point cloud (B, N, 3) → (B, N, E) using MLP.
    Optional Fourier positional encoding on XYZ coordinates.
    """
    def __init__(self, output_dim: int, pos_enc: bool = False):
        super().__init__()
        self.pos_enc = pos_enc

        # If enabled, add 4 frequencies (sin/cos) per axis → (3 + 3×8 = 27)
        in_dim = 3 * (1 + 4 * 2) if pos_enc else 3

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Input: (B, N, 3) → Output: (B, N, E)"""
        if self.pos_enc:
            B, N, _ = points.shape
            freqs = 2 ** torch.arange(4, device=points.device).float()  # [1,2,4,8]
            pts_freq = (points.unsqueeze(-1) * freqs).view(B, N, -1)     # (B,N,12)
            pts_pe = torch.cat([pts_freq.sin(), pts_freq.cos()], dim=-1)  # (B,N,24)
            x = torch.cat([points, pts_pe], dim=-1)  # (B,N,27)
        else:
            x = points  # (B,N,3)
        return self.mlp(x)  # (B,N,E)


# ---------------------------------------------------------------------------
#  CAD‑Recode Model
# ---------------------------------------------------------------------------
class CADRecodeModel(nn.Module):
    """
    Combines:
    - A 3D point-cloud projector (encoder producing token embeddings)
    - A HuggingFace CausalLM decoder

    Args:
        llm_name (str): HuggingFace model name (e.g. Qwen/Qwen2-1.5B)
        freeze_decoder (bool): If True, decoder weights are frozen
        pos_enc (bool): If True, Fourier positional encoding is used in projector
    """
    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2-1.5B",
        freeze_decoder: bool = False,
        pos_enc: bool = False,
    ) -> None:
        super().__init__()

        # 1. Load pretrained causal decoder
        try:
            self.decoder = AutoModelForCausalLM.from_pretrained(llm_name)
        except Exception as e:
            raise RuntimeError(f"Could not load decoder model '{llm_name}': {e}")

        embed_dim = self.decoder.get_input_embeddings().embedding_dim

        # 2. Point-token projector
        self.projector = PointCloudProjector(embed_dim, pos_enc=pos_enc)

        # 3. Tokenizer and special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|start|>", "<|end|>"]
        })
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.start_id = self.tokenizer.convert_tokens_to_ids("<|start|>")
        self.end_id   = self.tokenizer.convert_tokens_to_ids("<|end|>")

        # 4. Optionally freeze decoder weights
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    # -----------------------------------------------------------------------
    #  Forward pass with optional labels for loss
    # -----------------------------------------------------------------------
    def forward(
        self,
        points: torch.Tensor,                         # (B, N, 3)
        code: Optional[Union[List[str], torch.Tensor]] = None,
        labels: Optional[Union[List[str], torch.Tensor]] = None,
    ) -> torch.nn.modules.module.Module:
        """
        Args:
            points (Tensor): point cloud (B, N, 3)
            code (str[] or Tensor): tokenized input strings (for decoding)
            labels (str[] or Tensor): tokenized target strings (for LM loss)
        Returns:
            HuggingFace `CausalLMOutputWithCrossAttentions` (includes loss)
        """
        device = points.device
        B, N_pts, _ = points.shape

        # -- 1) point → token embeddings ----------------------------------
        pt_tokens = self.projector(points)  # (B, N_pts, E)

        # -- 2) tokenise code strings (if not tensor already) -------------
        if code is not None and not torch.is_tensor(code):
            tok = self.tokenizer(list(code), return_tensors="pt", padding=True)
            input_ids = tok["input_ids"].to(device)
        else:
            input_ids = code or torch.full((B, 1), self.start_id, dtype=torch.long, device=device)

        txt_embeds = self.decoder.get_input_embeddings()(input_ids)  # (B,T,E)

        # -- 3) concat:  [point-tokens] + [text-tokens] -------------------
        inputs_embeds = torch.cat([pt_tokens, txt_embeds], dim=1)  # (B,N+T,E)
        attn_mask = torch.ones((B, inputs_embeds.size(1)), dtype=torch.long, device=device)

        # -- 4) label masking for loss ------------------------------------
        new_labels = None
        if labels is not None:
            if not torch.is_tensor(labels):
                lbl = self.tokenizer(list(labels), return_tensors="pt", padding=True)
                lbl_ids = lbl["input_ids"]
            else:
                lbl_ids = labels
            lbl_ids = lbl_ids.to(device)
            ignore = torch.full((B, N_pts), -100, dtype=torch.long, device=device)
            new_labels = torch.cat([ignore, lbl_ids], dim=1)

        # -- 5) run decoder -----------------------------------------------
        return self.decoder(
            inputs_embeds  = inputs_embeds,
            attention_mask = attn_mask,
            labels         = new_labels
        )

    # -----------------------------------------------------------------------
    #  Prepare prefix for generation: return (inputs_embeds, attention_mask)
    # -----------------------------------------------------------------------
    def prepare_prefix(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the prefix (point embeddings + <|start|>) for decoder.generate()

        Args:
            points: (B, N, 3)
        Returns:
            inputs_embeds:   (B, N+1, E)
            attention_mask:  (B, N+1)
        """
        with torch.no_grad():
            pt_emb = self.projector(points)  # (B, N, E)
            B, N, E = pt_emb.shape
            start_ids = torch.full((B, 1), self.start_id, dtype=torch.long, device=points.device)
            start_emb = self.decoder.get_input_embeddings()(start_ids)  # (B,1,E)
            combined = torch.cat([pt_emb, start_emb], dim=1)  # (B, N+1, E)
            attn = torch.ones((B, N+1), dtype=torch.long, device=points.device)
        return combined, attn
