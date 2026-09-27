import pytest
import torch as th

from ai.rl.raw_features import _normalize_raw_bank_in_place
from env.observation import RAW_MISSING_VALUE
from ai.rl.device import release_idle_cuda_memory


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_chunked_normalization_is_bitwise_equal_to_full_table(device):
    if device == 'cuda' and not th.cuda.is_available():
        pytest.skip('CUDA unavailable')
    raw = th.randn(37, 11, 5, device=device)
    raw[::3, :, 1] = float(RAW_MISSING_VALUE)
    raw[::4, :, 2] = 0
    scales = th.tensor([.01, .1, 1., 10., 100.], device=device)
    pit = th.ones((37, 11), dtype=th.bool, device=device)
    pit[::2, 0] = False
    valid = th.ones(37, dtype=th.bool, device=device)
    valid[:2] = False
    expected = raw.clone()
    missing = expected == float(RAW_MISSING_VALUE)
    negative = (expected < 0) & ~missing
    expected.masked_fill_(missing, 0).div_(scales)
    expected.sub_(negative.to(expected.dtype) * 2).masked_fill_(missing, -1)
    expected.mul_((pit & valid[:, None]).unsqueeze(-1))
    original_pointer = raw.data_ptr()
    _normalize_raw_bank_in_place(raw, scales, pit, valid, chunk_bytes=3 * 11 * 5 * 4)
    assert raw.data_ptr() == original_pointer
    assert th.equal(raw, expected)


def test_cuda_normalization_temporary_allocation_is_bounded():
    if not th.cuda.is_available():
        pytest.skip('CUDA unavailable')
    raw = th.randn(256, 128, 32, device='cuda')
    scales = th.ones(32, device='cuda')
    pit = th.ones((256, 128), dtype=th.bool, device='cuda')
    valid = th.ones(256, dtype=th.bool, device='cuda')
    th.cuda.synchronize()
    before = th.cuda.memory_allocated()
    th.cuda.reset_peak_memory_stats()
    _normalize_raw_bank_in_place(raw, scales, pit, valid, chunk_bytes=64 * 1024)
    th.cuda.synchronize()
    assert th.cuda.max_memory_allocated() - before < 1024 * 1024


def test_releasing_idle_cuda_cache_preserves_live_tensors():
    if not th.cuda.is_available():
        pytest.skip('CUDA unavailable')
    live = th.arange(1024, device='cuda', dtype=th.float32)
    expected = live.cpu()
    temporary = th.ones(8 * 1024 * 1024, device='cuda')
    del temporary
    allocated = th.cuda.memory_allocated()
    result = release_idle_cuda_memory()
    assert result['allocated_after'] == allocated
    assert result['reserved_after'] < result['reserved_before']
    assert th.equal(live.cpu(), expected)
