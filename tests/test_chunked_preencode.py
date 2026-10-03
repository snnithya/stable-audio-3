"""Chunked pre-encoding: `ChunkedSampleDataset`, `--chunk_seconds` through
`pre_encode_dataset.encode_dataset`, and the control loaders reading at the chunk offset.
Builds on the WJD fixture tree in test_wjd_metadata.py (3 s tone stems, no GPU).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from stable_audio_3.data.dataset import ChunkedSampleDataset, LocalDatasetConfig
from stable_audio_3.data.features import LATENT_HOP, rms_envelope_control
from test_wjd_metadata import (  # noqa: F401  (fixtures are registered by import)
    ADD_FEATURES,
    MD_MODULE,
    PRE_ENCODE,
    REPO,
    SR,
    FakeAE,
    _info,
    _preencode_args,
    _tone,
    _write,
    load,
    md,
    mirror_tree,
)

SLAKH_MODULE = REPO / "stable_audio_3/configs/dataset_configs/custom_metadata/custom_md_slakh_streamgen.py"


@pytest.fixture(scope="module")
def pe():
    return load(PRE_ENCODE, "pre_encode_dataset_chunked")


def one_second_window(pe):
    return pe.resolve_chunking(1.0, 0.5, SR)  # 11 frames = 45056 samples, hop 6 frames


def chunked(md, root, window, hop, **kw):
    return ChunkedSampleDataset(
        [LocalDatasetConfig(id="t", path=str(root), custom_metadata_fn=md.get_custom_metadata)],
        sample_size=window, hop_size=hop, sample_rate=SR, force_channels="stereo",
        random_crop=False, resample_on_reject=False, **kw,
    )


# ---------------------------------------------------------------------------
# Window arithmetic
# ---------------------------------------------------------------------------


def test_resolve_chunking_rounds_to_whole_frames(pe):
    window, hop = pe.resolve_chunking(13.37, 0.5, SR)
    assert window == 144 * LATENT_HOP and hop == 72 * LATENT_HOP
    assert pe.resolve_chunking(1.0, 1.0, SR) == (11 * LATENT_HOP, 11 * LATENT_HOP)
    assert pe.resolve_chunking(1.0, 0.01, SR)[1] == LATENT_HOP  # never below one frame


def test_chunk_offsets_cover_the_file_and_handle_tails(pe, md, mirror_tree):
    window, hop = one_second_window(pe)
    ds = chunked(md, mirror_tree / "tracks" / "bass", window, hop)
    assert ds.chunk_offsets(window // 2) == [0]  # shorter than a window: one padded chunk
    assert ds.chunk_offsets(window) == [0]
    assert ds.chunk_offsets(3 * SR) == [0, hop, 2 * hop, 3 * hop]  # 3 s file, 50 % hop, tail < half a window
    ds.hop_size = window  # no overlap: a 1.9-window file keeps its 0.9 tail, a 1.4-window one does not
    assert ds.chunk_offsets(int(1.9 * window)) == [0, window]
    assert ds.chunk_offsets(int(1.4 * window)) == [0]


def test_hop_must_fit_the_window(md, mirror_tree):
    with pytest.raises(ValueError):
        chunked(md, mirror_tree / "tracks" / "bass", 4096, 8192)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def test_items_are_windows_with_chunk_metadata(pe, md, mirror_tree):
    window, hop = one_second_window(pe)
    ds = chunked(md, mirror_tree / "tracks" / "bass", window, hop)
    assert len(ds) == 8  # two 3 s files x 4 windows
    seen = {"A": 0, "B": 0}
    for i in range(8):
        audio, info = ds[i]
        track = Path(info["path"]).parent.name
        k = info["chunk_index"]
        seen[track] += 1
        assert audio.shape == (2, window)
        assert info["padding_mask"][0].sum() == window
        assert info["n_chunks"] == 4
        assert info["chunk_offset"] == k * hop and info["chunk_samples"] == window
        assert info["chunk_offset_seconds"] == pytest.approx(k * hop / SR)
        assert info["seconds_total"] == 3 and info["relpath"].endswith("bass.flac")
        if track == "A":
            assert info["prompt"] == "upright bass" and "__reject__" not in info
        else:
            # Track B has silent drums: the module rejects every window of it, and the
            # dataset hands the reject back flagged rather than swapping in another item.
            assert info["__reject__"] and "drums" in info["__reject_reason__"]
    assert seen == {"A": 4, "B": 4}


def test_short_file_is_one_padded_window(pe, md, tmp_path):
    window, hop = one_second_window(pe)
    _write(tmp_path / "short" / "x.flac", _tone(220, 0.5))
    ds = ChunkedSampleDataset(
        [LocalDatasetConfig(id="t", path=str(tmp_path / "short"))],
        sample_size=window, hop_size=hop, sample_rate=SR, resample_on_reject=False,
    )
    assert len(ds) == 1
    audio, info = ds[0]
    assert audio.shape == (2, window)
    assert info["padding_mask"][0].sum() == int(0.5 * SR)
    assert torch.all(audio[:, int(0.5 * SR):] == 0)


def test_drum_feature_of_a_chunk_matches_the_whole_file_at_that_offset(pe, md, mirror_tree):
    # Offsets are whole latent frames, so a chunk's per-frame RMS must be the same frames a
    # whole-file encode would have produced: this is what makes chunked and whole-file
    # controls interchangeable, and what proves the drums were read at the offset.
    window, hop = one_second_window(pe)
    ds = chunked(md, mirror_tree / "tracks" / "bass", window, hop)
    drums, _ = torchaudio.load(str(mirror_tree / "tracks" / "drums" / "A" / "drums.flac"))
    whole = rms_envelope_control(drums, valid_samples=drums.shape[-1])
    compared = 0
    for i in range(len(ds)):
        _, info = ds[i]
        if Path(info["path"]).parent.name != "A":
            continue  # track B is rejected (silent drums) and carries no feature
        f0 = info["chunk_offset"] // LATENT_HOP
        assert torch.equal(info["drums_rms"], whole[:, f0 : f0 + window // LATENT_HOP])
        compared += 1
    assert compared == 4


# ---------------------------------------------------------------------------
# Through the pre-encode script
# ---------------------------------------------------------------------------


def test_encode_dataset_writes_one_item_per_window(pe, md, mirror_tree, tmp_path):
    out = tmp_path / "latents"
    args = _preencode_args(features=["drums_rms"], chunk_seconds=1.0, chunk_hop_ratio=0.5, batch_size=2, pad=True)
    pe.encode_dataset(FakeAE(), str(mirror_tree / "tracks" / "bass"), str(out), md.get_custom_metadata, args)
    window, hop = one_second_window(pe)
    assert args.sample_size == window  # the window became the sample size
    written = sorted(p.name for p in out.glob("*.json") if not p.name.startswith("_"))
    # Track A's 4 windows survive, B's 4 are rejected (silent drums). Ids follow (batch, index)
    # at batch size 2 and the file order is the scanner's, so check the contents, not the names.
    assert len(written) == 4
    infos = [json.loads((out / name).read_text()) for name in written]
    assert sorted(i["chunk_index"] for i in infos) == [0, 1, 2, 3]
    for name, info in zip(written, infos):
        k = info["chunk_index"]
        assert Path(info["path"]).parent.name == "A"
        assert info["chunk_offset"] == k * hop and info["chunk_samples"] == window and info["n_chunks"] == 4
        assert info["prompt"] == "upright bass"
        assert len(info["padding_mask"]) == window // LATENT_HOP
        ctrl = np.load(out / name.replace(".json", "_controls.npy"))
        assert ctrl.shape == (1, window // LATENT_HOP) and info["control_keys"] == ["drums_rms"]
    assert (out / "_skipped.json").exists()


def test_add_features_reads_the_chunk_window_from_the_sidecar(pe, md, mirror_tree, tmp_path, monkeypatch):
    out = tmp_path / "latents"
    args = _preencode_args(features=["drums_rms"], chunk_seconds=1.0, chunk_hop_ratio=0.5, batch_size=1, pad=True)
    pe.encode_dataset(FakeAE(), str(mirror_tree / "tracks" / "bass"), str(out), md.get_custom_metadata, args)
    from stable_audio_3.data.features import TriaStats
    stats = tmp_path / "stats.json"
    TriaStats(split_hz=200.0, band_mean_db=[-40.0, -40.0], band_std_db=[10.0, 10.0]).save(stats)
    monkeypatch.setenv("WJD_CONTROL_MODE", "rms,tria_fixed")
    monkeypatch.setenv("WJD_TRIA_STATS", str(stats))
    tria_md = load(MD_MODULE, "custom_md_wjd_tria_chunked")
    add = load(ADD_FEATURES, "add_features_chunked")
    item = next(  # the third window of track A, whatever id the file order gave it
        p for p in out.glob("*.json") if not p.name.startswith("_") and json.loads(p.read_text())["chunk_index"] == 2
    )
    res = add.process_item(item, tria_md, None, ["drums_tria_fixed"])  # no --sample_size: from the sidecar
    assert res["keys"] == {"drums_rms": "verified", "drums_tria_fixed": "added"} and res["max_diff"]["drums_rms"] == 0.0
    info = json.loads(item.read_text())
    window, hop = one_second_window(pe)
    expected = tria_md.get_custom_metadata(
        {**_info(mirror_tree / "tracks" / "bass" / "A" / "bass.flac", window, window), "chunk_offset": 2 * hop}, None
    )["__features__"]["drums_tria_fixed"]
    fused = np.load(item.parent / f"{item.stem}_controls.npy")
    assert info["control_keys"] == ["drums_rms", "drums_tria_fixed"]
    assert np.array_equal(fused[1:3], expected.numpy())


# ---------------------------------------------------------------------------
# The Slakh accompaniment loader honours the offset too
# ---------------------------------------------------------------------------


def test_slakh_submix_is_read_at_the_offset(tmp_path):
    slakh = load(SLAKH_MODULE, "custom_md_slakh_streamgen_chunked")
    stem = torch.cat([torch.zeros(2, SR), _tone(440, 1.0), torch.zeros(2, SR)], dim=1)  # sound only in [1 s, 2 s)
    _write(tmp_path / "other" / "piano.flac", stem)
    paths = [tmp_path / "other" / "piano.flac"]
    mix, names = slakh.load_and_mix_stems(paths, SR, target_length=SR)  # window [0, 1 s): silent
    assert mix is None
    mix, names = slakh.load_and_mix_stems(paths, SR, target_length=SR, offset=SR)  # window [1 s, 2 s)
    assert mix is not None and mix.shape == (2, SR) and names == ["piano"]
