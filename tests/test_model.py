import torch

from tft.model import TemporalFusionTransformer


def _tiny_batch(batch=3, enc=24, dec=6, hidden=16,
                n_static=4, n_known=6, n_observed=5):
    return {
        "static": torch.randn(batch, n_static, hidden),
        "known": torch.randn(batch, enc + dec, n_known, hidden),
        "observed": torch.randn(batch, enc, n_observed, hidden),
    }


def test_forward_shape_matches_decoder_length_and_quantiles():
    model = TemporalFusionTransformer(
        n_static_inputs=4, n_known_inputs=6, n_observed_inputs=5,
        hidden_size=16, n_heads=2, encoder_length=24, decoder_length=6,
        quantiles=(0.1, 0.5, 0.9),
    )
    batch = _tiny_batch()
    out = model(**batch)
    assert out["predictions"].shape == (3, 6, 3)


def test_attention_weights_have_expected_shape():
    model = TemporalFusionTransformer(
        n_static_inputs=4, n_known_inputs=6, n_observed_inputs=5,
        hidden_size=16, n_heads=2, encoder_length=24, decoder_length=6,
    )
    out = model(**_tiny_batch())
    assert out["attention"].shape == (3, 6, 24 + 6)


def test_variable_selection_weights_sum_to_one():
    model = TemporalFusionTransformer(
        n_static_inputs=4, n_known_inputs=6, n_observed_inputs=5,
        hidden_size=16, n_heads=2, encoder_length=24, decoder_length=6,
    )
    out = model(**_tiny_batch())
    s = out["static_weights"].sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_gradient_flows_end_to_end():
    model = TemporalFusionTransformer(
        n_static_inputs=4, n_known_inputs=6, n_observed_inputs=5,
        hidden_size=16, n_heads=2, encoder_length=24, decoder_length=6,
    )
    out = model(**_tiny_batch())
    loss = out["predictions"].pow(2).mean()
    loss.backward()
    has_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                   for p in model.parameters() if p.requires_grad)
    assert has_grad > 0


def test_invalid_head_size_raises():
    import pytest
    with pytest.raises(ValueError):
        TemporalFusionTransformer(
            n_static_inputs=4, n_known_inputs=6, n_observed_inputs=5,
            hidden_size=15, n_heads=4,
        )
