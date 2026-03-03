"""
Model Tests — Smoke tests for the PyTorch model (CPU, no training data needed).
"""

import pytest
import torch
import numpy as np


class TestModelArchitecture:
    @pytest.fixture(scope="class")
    def model(self):
        from ml.model import CFUDetectorModel
        return CFUDetectorModel(pretrained=False)

    def test_forward_pass_shapes(self, model):
        dummy = torch.randn(2, 3, 512, 512)
        hm, sz = model(dummy)
        assert hm.shape == (2, 1, 128, 128)
        assert sz.shape == (2, 1, 128, 128)

    def test_heatmap_range(self, model):
        dummy = torch.randn(1, 3, 512, 512)
        hm, _ = model(dummy)
        assert hm.min().item() >= 0.0
        assert hm.max().item() <= 1.0

    def test_size_map_non_negative(self, model):
        dummy = torch.randn(1, 3, 512, 512)
        _, sz = model(dummy)
        assert sz.min().item() >= 0.0

    def test_parameter_count_reasonable(self, model):
        total = sum(p.numel() for p in model.parameters()) / 1e6
        assert 1.0 < total < 20.0

    def test_save_load_roundtrip(self, model, tmp_path):
        from ml.model import CFUDetectorModel
        path = str(tmp_path / "test_model.pt")
        model.save(path, extra_info={"test": True})
        loaded = CFUDetectorModel.load(path, device="cpu")
        dummy = torch.randn(1, 3, 512, 512)
        hm1, sz1 = model(dummy)
        hm2, sz2 = loaded(dummy)
        assert torch.allclose(hm1, hm2, atol=1e-5)


class TestLossFunctions:
    def test_focal_loss_range(self):
        from ml.model import FocalLoss
        loss_fn = FocalLoss()
        pred   = torch.rand(2, 1, 128, 128)
        target = torch.zeros(2, 1, 128, 128)
        target[0, 0, 64, 64] = 1.0
        loss = loss_fn(pred, target)
        assert loss.item() > 0.0
        assert not torch.isnan(loss)

    def test_combined_loss_returns_dict(self):
        from ml.model import CFUDetectorLoss
        loss_fn = CFUDetectorLoss()
        hm_pred = torch.sigmoid(torch.randn(2, 1, 128, 128))
        sz_pred = torch.relu(torch.randn(2, 1, 128, 128))
        hm_tgt  = torch.zeros(2, 1, 128, 128)
        sz_tgt  = torch.zeros(2, 1, 128, 128)
        losses = loss_fn(hm_pred, sz_pred, hm_tgt, sz_tgt)
        assert "total" in losses
        assert "focal" in losses
        assert "size"  in losses


class TestHeatmapDecoding:
    def test_decode_single_peak(self):
        from ml.evaluate import decode_heatmap_predictions
        hm = np.zeros((1, 128, 128), dtype=np.float32)
        sz = np.zeros((1, 128, 128), dtype=np.float32)
        hm[0, 64, 64] = 0.9
        sz[0, 64, 64] = 5.0
        preds = decode_heatmap_predictions(hm, sz, stride=4, score_thresh=0.5)
        assert len(preds) >= 1
        cx, cy, r, score = preds[0]
        assert abs(cx - 258.0) < 10
        assert score >= 0.9

    def test_decode_empty_heatmap(self):
        from ml.evaluate import decode_heatmap_predictions
        hm = np.zeros((1, 128, 128), dtype=np.float32)
        sz = np.zeros((1, 128, 128), dtype=np.float32)
        preds = decode_heatmap_predictions(hm, sz, score_thresh=0.5)
        assert preds == []