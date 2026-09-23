"""Media metadata indexing tests (Phase 5, MEDIA-01, plan 05-01).

Fixtures are generated in-test with mutagen (tiny tagged MP3) — model-free
and dependency-optional: every test skips when mutagen is missing.
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["LOCAL_MEMORY_HOME"] = str(Path(__file__).parent / ".tmpdata-media")

from local_memory import config  # noqa: E402
from local_memory.extractors import text_extractor  # noqa: E402
from local_memory.store import database, vector_store  # noqa: E402
from local_memory.pipeline import scan_folder  # noqa: E402
from local_memory.search import query_engine  # noqa: E402

try:
    import mutagen  # noqa: F401
    HAVE_MUTAGEN = True
except ImportError:
    HAVE_MUTAGEN = False

_HOME = Path(__file__).parent / ".tmpdata-media"


def setup_module(_):
    shutil.rmtree(_HOME, ignore_errors=True)
    config.DATA_HOME = _HOME
    config.DB_PATH = _HOME / "index.db"
    config.THUMBS_DIR = _HOME / "thumbs"
    config.SETTINGS_PATH = _HOME / "settings.json"
    config.MODELS_DIR = _HOME / "no-models"  # model-free: hashing fallback
    database.close()
    vector_store.invalidate_cache()
    config.ensure_dirs()
    database.init_db()
    vector_store.init_db()


def teardown_module(_):
    database.close()
    vector_store.invalidate_cache()
    shutil.rmtree(_HOME, ignore_errors=True)


def _make_tagged_mp3(path: Path) -> None:
    """Minimal silent MP3 frame + ID3 tags via mutagen."""
    from mutagen.id3 import ID3, TALB, TIT2, TPE1, ID3NoHeaderError
    from mutagen.mp3 import MP3

    # MPEG-1 Layer III frames (128 kbps, 44.1 kHz, 417 B each), repeated so
    # mutagen's stream parser finds enough valid frames to sync.
    frame = b"\xff\xfb\x90\x44" + b"\x00" * 413
    path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + frame * 10)
    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()
    tags.add(TIT2(encoding=3, text="Neon River"))
    tags.add(TPE1(encoding=3, text="Awanish"))
    tags.add(TALB(encoding=3, text="Midnight Sessions"))
    tags.save(str(path))
    MP3(str(path))  # sanity: mutagen can parse it


@pytest.mark.skipif(not HAVE_MUTAGEN, reason="mutagen not installed")
def test_read_media_returns_tags():
    p = _HOME / "tagged.mp3"
    _make_tagged_mp3(p)
    text = text_extractor._read_media(p)
    assert text is not None
    assert "Neon River" in text
    assert "Awanish" in text
    assert "Midnight Sessions" in text
    assert "audio recording" in text


@pytest.mark.skipif(not HAVE_MUTAGEN, reason="mutagen not installed")
def test_metadata_chunks_media_two_chunk_contract():
    p = _HOME / "tagged.mp3"
    _make_tagged_mp3(p)
    kind, chunks = text_extractor.metadata_chunks(p)
    assert kind == "media"
    assert len(chunks) == 2, "media must keep the 2-chunk contract"
    assert "Neon River" in chunks[1] and "Awanish" in chunks[1]


def test_read_media_untagged_returns_none(tmp_path):
    p = tmp_path / "fake.mp3"
    p.write_bytes(b"garbage bytes not an mp3")
    if HAVE_MUTAGEN:
        assert text_extractor._read_media(p) is None
    else:
        assert text_extractor._read_media(p) is None


def test_media_kinds_and_indexable():
    assert config.binary_kind("x.mp3") == "media"
    assert config.binary_kind("x.mkv") == "media"
    assert ".mp4" in config.indexable_exts()
    assert ".flac" in config.indexable_exts()


@pytest.mark.skipif(not HAVE_MUTAGEN, reason="mutagen not installed")
def test_media_end_to_end_search():
    """Tagged MP3 found by song title; untagged .mkv found by filename tokens;
    both indexed with kind 'media' and 2 chunks each."""
    media = _HOME / "media_files"
    media.mkdir(exist_ok=True)
    song = media / "neon_river_track.mp3"
    _make_tagged_mp3(song)
    video = media / "kali-setup-demo.mkv"
    video.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 64)  # EBML magic, no tags
    assert scan_folder(str(media)) >= 2

    row_song = database.get_file(str(song))
    row_video = database.get_file(str(video))
    assert row_song["kind"] == "media"
    assert row_video["kind"] == "media"
    n_song_chunks = database.fts_rows(
        "SELECT COUNT(*) AS n FROM chunks WHERE file_id=?",
        (row_song["id"],))[0]["n"]
    n_video_chunks = database.fts_rows(
        "SELECT COUNT(*) AS n FROM chunks WHERE file_id=?",
        (row_video["id"],))[0]["n"]
    assert n_song_chunks == 2, n_song_chunks
    assert n_video_chunks == 2, n_video_chunks

    results = query_engine.search('"Neon River" Awanish')
    hit = next((r for r in results if r["name"] == "neon_river_track.mp3"), None)
    assert hit is not None, "song-title query missed the tagged MP3"
    assert hit["match_keyword"] > 0

    results = query_engine.search("kali setup demo")
    hit = next((r for r in results if r["name"] == "kali-setup-demo.mkv"), None)
    assert hit is not None, "untagged video not found by filename tokens"
