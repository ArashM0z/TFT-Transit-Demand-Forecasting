"""Temporal Fusion Transformer (Lim et al., 2021).

This implementation faithfully reproduces the architectural elements from the
paper: Variable Selection Networks, Gated Residual Networks, an LSTM
encoder/decoder for local processing, interpretable multi-head attention for
long-range dependencies, and quantile output heads.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """GLU as used throughout the paper."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class AddNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(x + skip)


class GatedResidualNetwork(nn.Module):
    """Gated residual network (eq. 2–4) with optional context."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1, context_size: int | None = None):
        super().__init__()
        self.skip = (nn.Identity() if input_size == output_size
                     else nn.Linear(input_size, output_size))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context = (nn.Linear(context_size, hidden_size, bias=False)
                        if context_size is not None else None)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GatedLinearUnit(hidden_size, output_size, dropout)
        self.add_norm = AddNorm(output_size)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.context is not None and context is not None:
            if context.dim() == h.dim() - 1:
                context = context.unsqueeze(-2).expand(*h.shape[:-1], -1)
            h = h + self.context(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.glu(h)
        return self.add_norm(h, self.skip(x))


class VariableSelectionNetwork(nn.Module):
    """Selects relevant inputs at each step (paper §4.2)."""

    def __init__(self, n_inputs: int, hidden_size: int, dropout: float = 0.1,
                 context_size: int | None = None):
        super().__init__()
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.flat_grn = GatedResidualNetwork(
            input_size=n_inputs * hidden_size,
            hidden_size=hidden_size,
            output_size=n_inputs,
            dropout=dropout,
            context_size=context_size,
        )
        self.per_input_grn = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
            for _ in range(n_inputs)
        ])

    def forward(self, inputs: torch.Tensor,
                context: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs: (..., n_inputs, hidden_size)
        flat = inputs.flatten(start_dim=-2)
        weights = F.softmax(self.flat_grn(flat, context), dim=-1)  # (..., n_inputs)
        processed = torch.stack(
            [grn(inputs[..., i, :]) for i, grn in enumerate(self.per_input_grn)],
            dim=-2,
        )  # (..., n_inputs, hidden_size)
        combined = (weights.unsqueeze(-1) * processed).sum(dim=-2)
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Shared-V attention used in the paper for interpretability."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = hidden_size // n_heads
        self.hidden_size = hidden_size
        self.q_layers = nn.ModuleList(
            [nn.Linear(hidden_size, self.d_head, bias=False) for _ in range(n_heads)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(hidden_size, self.d_head, bias=False) for _ in range(n_heads)]
        )
        self.v_layer = nn.Linear(hidden_size, self.d_head, bias=False)
        self.out = nn.Linear(self.d_head, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k, v: (batch, seq, hidden)
        v_shared = self.v_layer(v)  # (batch, seq, d_head)
        scale = 1.0 / math.sqrt(self.d_head)

        head_outputs = []
        attn_weights = []
        for i in range(self.n_heads):
            qi = self.q_layers[i](q)
            ki = self.k_layers[i](k)
            scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn_weights.append(attn)
            head_outputs.append(torch.matmul(self.dropout(attn), v_shared))

        # average across heads (shared-V interpretable attention)
        out = torch.stack(head_outputs, dim=0).mean(dim=0)
        attn = torch.stack(attn_weights, dim=0).mean(dim=0)
        return self.out(out), attn


class TemporalFusionTransformer(nn.Module):
    """Full TFT model.

    Input batch dictionary:
        static:   (B, n_static_inputs, hidden_size)
        known:    (B, encoder_length + decoder_length, n_known_inputs, hidden_size)
        observed: (B, encoder_length, n_observed_inputs, hidden_size)

    Output:
        predictions: (B, decoder_length, n_quantiles)
        interpretation: dict of attention/variable-selection weights
    """

    def __init__(self, n_static_inputs: int, n_known_inputs: int,
                 n_observed_inputs: int, hidden_size: int = 64, n_heads: int = 4,
                 encoder_length: int = 168, decoder_length: int = 24,
                 quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
                 dropout: float = 0.1):
        super().__init__()
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.hidden_size = hidden_size

        # 1. variable selection
        self.static_vsn = VariableSelectionNetwork(n_static_inputs, hidden_size, dropout)
        self.encoder_vsn = VariableSelectionNetwork(
            n_known_inputs + n_observed_inputs, hidden_size, dropout,
            context_size=hidden_size,
        )
        self.decoder_vsn = VariableSelectionNetwork(
            n_known_inputs, hidden_size, dropout, context_size=hidden_size,
        )

        # 2. static context generators
        self.ctx_var_selection = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.ctx_init_h = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.ctx_init_c = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.ctx_enrichment = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        # 3. local encoder / decoder LSTMs
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.lstm_add_norm = AddNorm(hidden_size)

        # 4. static enrichment
        self.static_enrichment = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout, context_size=hidden_size,
        )

        # 5. attention
        self.attention = InterpretableMultiHeadAttention(hidden_size, n_heads, dropout)
        self.attn_glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.attn_add_norm = AddNorm(hidden_size)

        # 6. position-wise feedforward
        self.pos_ff = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.final_glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.final_add_norm = AddNorm(hidden_size)

        # 7. output heads
        self.output = nn.Linear(hidden_size, len(quantiles))
        self.quantiles = quantiles

    @staticmethod
    def _decoder_mask(decoder_length: int, encoder_length: int,
                      device: torch.device) -> torch.Tensor:
        total = encoder_length + decoder_length
        mask = torch.ones(decoder_length, total, dtype=torch.bool, device=device)
        # Causal mask over the decoder portion (allow attending to all encoder steps
        # but only to past+current decoder steps).
        causal = torch.tril(torch.ones(decoder_length, decoder_length,
                                       dtype=torch.bool, device=device))
        mask[:, encoder_length:] = causal
        return mask

    def forward(self, static: torch.Tensor, known: torch.Tensor,
                observed: torch.Tensor) -> dict[str, torch.Tensor]:
        # --- static processing ---
        static_emb, static_w = self.static_vsn(static)
        ctx_vs = self.ctx_var_selection(static_emb)
        ctx_h = self.ctx_init_h(static_emb)
        ctx_c = self.ctx_init_c(static_emb)
        ctx_e = self.ctx_enrichment(static_emb)

        # --- temporal variable selection ---
        encoder_inputs = torch.cat([
            known[:, :self.encoder_length], observed,
        ], dim=-2)
        encoder_features, enc_w = self.encoder_vsn(encoder_inputs, ctx_vs)
        decoder_inputs = known[:, self.encoder_length:]
        decoder_features, dec_w = self.decoder_vsn(decoder_inputs, ctx_vs)

        # --- local LSTM processing ---
        h0 = ctx_h.unsqueeze(0)
        c0 = ctx_c.unsqueeze(0)
        enc_out, (hn, cn) = self.encoder_lstm(encoder_features, (h0, c0))
        dec_out, _ = self.decoder_lstm(decoder_features, (hn, cn))
        lstm_out = torch.cat([enc_out, dec_out], dim=1)
        lstm_in = torch.cat([encoder_features, decoder_features], dim=1)
        lstm_out = self.lstm_add_norm(self.lstm_glu(lstm_out), lstm_in)

        # --- static enrichment over full sequence ---
        enriched = self.static_enrichment(lstm_out, ctx_e)

        # --- masked interpretable multi-head attention ---
        attn_out, attn_weights = self.attention(
            enriched[:, self.encoder_length:],  # queries: decoder steps only
            enriched, enriched,
            mask=self._decoder_mask(self.decoder_length, self.encoder_length,
                                    enriched.device),
        )
        attn_out = self.attn_add_norm(
            self.attn_glu(attn_out),
            enriched[:, self.encoder_length:],
        )

        # --- position-wise feedforward + final skip ---
        ff = self.pos_ff(attn_out)
        ff = self.final_add_norm(
            self.final_glu(ff),
            lstm_out[:, self.encoder_length:],
        )

        # --- output quantiles ---
        predictions = self.output(ff)  # (B, decoder_length, n_quantiles)

        return {
            "predictions": predictions,
            "static_weights": static_w,
            "encoder_weights": enc_w,
            "decoder_weights": dec_w,
            "attention": attn_weights,
        }
