import numpy as np
from kws.features import FeatureConfig, extract_log_mel

def test_feature_shape_and_finite_values():
    config = FeatureConfig()
    output = extract_log_mel(np.zeros(config.window_samples), config)
    assert output.shape == (99, 40, 1)
    assert np.isfinite(output).all()

def test_short_audio_is_zero_padded():
    config = FeatureConfig()
    assert extract_log_mel(np.ones(10), config).shape == (config.frame_count, config.mel_bins, 1)
