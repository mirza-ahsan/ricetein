"""pinned_pretrained: Kaggle loads the private weights copy only if it is the pinned commit."""

import pytest

from src import kaggle_bootstrap as kb

REPO, PIN = "facebook/dinov3-vitb16-pretrain-lvd1689m", "5931719e67bbdb9737e363e781fb0c67687896bc"


def fake_weights(root, revisions_line):
    model = root / REPO.replace("/", "--")
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (root / "REVISIONS.txt").write_text(revisions_line + "\n")
    return model


def test_local_machine_passes_the_revision_to_hugging_face(monkeypatch):
    monkeypatch.setattr(kb, "find_kaggle_input", lambda slug: None)
    assert kb.pinned_pretrained(REPO, PIN) == (REPO, {"revision": PIN})


def test_kaggle_copy_at_the_pinned_commit_is_used(monkeypatch, tmp_path):
    model = fake_weights(tmp_path, f"{REPO} {PIN}")
    monkeypatch.setattr(kb, "find_kaggle_input", lambda slug: tmp_path)
    assert kb.pinned_pretrained(REPO, PIN) == (str(model), {})


def test_kaggle_copy_at_another_commit_is_refused(monkeypatch, tmp_path):
    fake_weights(tmp_path, f"{REPO} 0000000000000000000000000000000000000000")
    monkeypatch.setattr(kb, "find_kaggle_input", lambda slug: tmp_path)
    with pytest.raises(RuntimeError, match="pinned revision"):
        kb.pinned_pretrained(REPO, PIN)
