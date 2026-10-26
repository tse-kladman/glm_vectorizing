"""Extraction failures must be counted, written down, and turned into a non-zero exit."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from glmbench import config, models
from glmbench.representations import extract, transformers

EVO2_SCRIPT = Path(__file__).resolve().parent.parent / "extract_evo2.py"


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def unsqueeze(self, _):
        return FakeTensor(self.array[None])

    def to(self, _):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.array


class FakeModel:
    """Stands in for Evo2: raises for sequences that start with 'T'."""

    tokenizer = types.SimpleNamespace(tokenize=lambda seq: [ord(c) for c in seq])

    def __init__(self, model_id):
        pass

    def __call__(self, ids, return_embeddings, layer_names):
        if int(ids.array[0, 0]) == ord("T"):
            raise RuntimeError("out of memory (simulated)")
        return None, {name: FakeTensor(np.ones((1, 4, 3))) for name in layer_names}


@pytest.fixture
def evo2_script(monkeypatch):
    torch = types.ModuleType("torch")
    torch.tensor = lambda data, dtype=None: FakeTensor(data)
    torch.int = None
    torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None,
                                                  __exit__=lambda s, *a: None)

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    torch.no_grad = NoGrad
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    evo2 = types.ModuleType("evo2")
    evo2.Evo2 = FakeModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "evo2", evo2)

    spec = importlib.util.spec_from_file_location("extract_evo2_under_test", EVO2_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(module, monkeypatch, tmp_path, fasta_text):
    fasta = tmp_path / "in.fa"
    fasta.write_text(fasta_text)
    monkeypatch.setattr(sys, "argv", ["extract_evo2.py", "--fasta", str(fasta), "--name", "x",
                                      "--outdir", str(tmp_path / "out"), "--n_blocks", "2"])
    return module.main


def test_evo2_failures_are_listed_and_the_exit_is_non_zero(evo2_script, monkeypatch, tmp_path):
    main = run_script(evo2_script, monkeypatch, tmp_path,
                      ">good_0\nACGT\n>bad_1\nTTTT\n>good_2\nGGGG\n>empty_3\n\n")
    with pytest.raises(SystemExit) as stop:
        main()
    assert stop.value.code == 1

    failed = (tmp_path / "out" / "x.failed.txt").read_text().splitlines()
    assert [line.split("\t")[0] for line in failed] == ["bad_1", "empty_3"]
    assert "out of memory" in failed[0]
    # the failures did not poison the good fragments (the old `finally` hid a NameError)
    import pickle
    saved = pickle.load(open(tmp_path / "out" / "x.pickle", "rb"))
    assert len(saved[0]["good"]) == 2


def test_evo2_clean_run_exits_normally_and_clears_a_stale_failure_list(
        evo2_script, monkeypatch, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x.failed.txt").write_text("old\tstale\n")
    main = run_script(evo2_script, monkeypatch, tmp_path, ">a_0\nACGT\n>a_1\nGGGG\n")
    main()
    assert not (tmp_path / "out" / "x.failed.txt").exists()


def test_a_failed_file_stops_the_stage_after_the_others_were_tried(tmp_path, monkeypatch):
    evo2 = models.get("evo2")
    calls = []

    def fake(model, path, name, latent_dir=None):
        calls.append(name)
        return name != "sample_1"

    monkeypatch.setattr(transformers, "extract_evo2", fake)
    items = []
    for name in ("sample_0", "sample_1", "sample_2"):
        path = tmp_path / f"{name}.fa"
        path.write_text(">a_0\nACGT\n")
        items.append((path, name))

    with pytest.raises(SystemExit, match="1 of 3 files FAILED"):
        extract.extract_model(evo2, items, latent_dir=tmp_path / "latent")
    assert calls == ["sample_0", "sample_1", "sample_2"]         # it kept going
    assert (tmp_path / "latent" / "evo2" / "failed.txt").read_text().split() == ["sample_1"]


def test_a_missing_input_fasta_is_a_failure_not_a_skip(tmp_path):
    kmer = models.get("kmer")
    with pytest.raises(SystemExit, match="FAILED"):
        extract.extract_model(kmer, [(tmp_path / "nope.fa", "sample_0")],
                              latent_dir=tmp_path / "latent")


def test_kmer_extraction_records_how_the_vectors_were_made(tmp_path):
    path = tmp_path / "sample_0.fa"
    path.write_text(">cls_0\n" + "ACGT" * 100 + "\n")
    extract.extract_model(models.get("kmer"), [(path, "sample_0")],
                          latent_dir=tmp_path / "latent", seed=9)
    info = json.loads((tmp_path / "latent" / "kmer" / "extraction.json").read_text())
    assert info["k"] == 4 and info["seed"] == 9


# ── ViraLM ──────────────────────────────────────────────────────────────────

def test_viralm_fails_early_with_instructions_when_the_model_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CODE_ROOT", tmp_path)               # a checkout without ViraLM
    path = tmp_path / "sample_0.fa"
    path.write_text(">a_0\nACGT\n")
    with pytest.raises(SystemExit, match="viralm_hook.py"):
        extract.extract_model(models.get("viralm"), [(path, "sample_0")],
                              latent_dir=tmp_path / "latent")

    hook = tmp_path / transformers.VIRALM_HOOK_DIR / transformers.VIRALM_HOOK_SCRIPT
    hook.parent.mkdir(parents=True)
    hook.write_text("# stub\n")
    with pytest.raises(SystemExit) as stop:
        extract.extract_model(models.get("viralm"), [(path, "sample_0")],
                              latent_dir=tmp_path / "latent")
    message = str(stop.value)
    assert "ViraLM/model" in message
    assert "github.com/ChengPENG-wolf/ViraLM" in message
    assert "licence" in message
    assert not (tmp_path / "latent" / "viralm" / "sample_0_layer_1.pkl").exists()


def test_the_viralm_hook_is_run_with_the_documented_command_line(tmp_path, monkeypatch):
    hook = tmp_path / transformers.VIRALM_HOOK_DIR / transformers.VIRALM_HOOK_SCRIPT
    hook.parent.mkdir(parents=True)
    hook.write_text("# stub\n")
    (hook.parent / "model").mkdir()
    monkeypatch.setattr(config, "CODE_ROOT", tmp_path)

    seen = {}

    def fake_conda(env, arguments, cwd, extra_env=None):
        seen.update(env=env, arguments=arguments, cwd=cwd)
        return True

    monkeypatch.setattr(transformers, "_conda", fake_conda)
    transformers.extract_viralm(models.get("viralm"), tmp_path / "s.fa", "s", tmp_path / "l")
    arguments = seen["arguments"]
    assert seen["env"] == "viralm" and seen["cwd"] == hook.parent
    assert arguments[0] == "viralm_hook.py"
    for flag in ("--input_pth", "--output_pth", "--batch_size", "--layers_to_monitor"):
        assert flag in arguments
    assert arguments[arguments.index("--layers_to_monitor") + 1:] == [str(n) for n in range(1, 13)]
    assert arguments[arguments.index("--batch_size") + 1] == "1"


def test_the_viralm_hook_refuses_batches_it_would_average_together(tmp_path, monkeypatch):
    """The hook stores np.mean(states, axis=0) per batch under the batch's first name:
    with more than one window per batch, different fragments would be merged."""
    monkeypatch.setattr(transformers, "_conda",
                        lambda *a, **k: pytest.fail("the hook must not be started"))
    with pytest.raises(ValueError, match="batch_size must be 1"):
        transformers.extract_viralm(models.get("viralm"), tmp_path / "s.fa", "s",
                                    tmp_path / "l", batch_size=8)
