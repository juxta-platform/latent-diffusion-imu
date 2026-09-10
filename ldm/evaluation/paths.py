"""Input discovery and classification, output naming, GT-class inference.

Every script takes the same three input flags and this module turns them into a
list of :class:`Recording` items:

* ``--input PATH`` -- one ``.hdf5``/``.parquet``, or a pair directory
* ``--input_dir PATH`` -- a directory whose layout is auto-detected
* ``--dataset_dir PATH`` + ``--split`` -- preprocessed ``.pt`` windows

Directory layouts recognised by :func:`classify_dir`::

    pair        <dir>/real.hdf5 (+ synthetic.parquet, trajectory.txt)
    flat        <dir>/*.hdf5 | *.parquet
    pair_batch  <dir>/**/real.hdf5, discovered via manifest.csv when present
    labeled     <dir>/<placement>/*.hdf5, any placement folder name
    preprocessed <dir>/{train,val}/*.pt + stats.pt

The placement labels produced here are dataset vocabulary, not classifier
classes: a ``head/`` folder is discovered and evaluated even when no trained
classifier predicts head. Use :func:`scoreable_label` to drop labels a
particular classifier cannot be scored against.
"""

import csv
import functools
import os
import os.path as osp
import re
from dataclasses import dataclass, field

from ldm.evaluation.constants import (
    CARRYING_ALIASES,
    CARRYING_ALIASES_BY_LENGTH,
    HDF5_EXTS,
    MANIFEST_NAME,
    REAL_NAME,
    STATS_NAME,
    SYNTHETIC_NAME,
    TRAJECTORY_EXTS,
    TRAJECTORY_NAME,
)


@dataclass
class Recording:
    """One evaluable recording plus whatever paired artifacts exist next to it."""

    name: str
    path: str
    sim_path: str = None
    trajectory_path: str = None
    pair_dir: str = None
    label: str = None
    extras: dict = field(default_factory=dict)

    @property
    def is_parquet(self):
        return osp.splitext(self.path)[1].lower() == ".parquet"

    @property
    def dataset_type(self):
        """RoNIN sequence type implied by the recording's extension."""
        return "sim_parquet" if self.is_parquet else "hybrid"

    @property
    def stem(self):
        return osp.splitext(osp.basename(self.path.rstrip("/")))[0]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def is_recording_file(path):
    return osp.isfile(path) and osp.splitext(path)[1].lower() in TRAJECTORY_EXTS


def is_pair_dir(path):
    return osp.isdir(path) and osp.isfile(osp.join(path, REAL_NAME))


def is_preprocessed_dir(path):
    return osp.isdir(path) and osp.isfile(osp.join(path, STATS_NAME)) and any(
        osp.isdir(osp.join(path, split)) for split in ("train", "val")
    )


def list_label_dirs(path):
    """Immediate subdirectories that directly hold recordings, sorted.

    Pair directories are excluded so that a placement folder of a pair dataset
    (whose children are pair dirs holding real.hdf5) still resolves as
    pair_batch rather than as labeled.
    """
    if not osp.isdir(path):
        return []
    candidates = (osp.join(path, name) for name in sorted(os.listdir(path)))
    return [
        sub for sub in candidates
        if osp.isdir(sub) and not is_pair_dir(sub) and list_recording_files(sub)
    ]


def is_labeled_dir(path):
    """True when at least one subdirectory is named for a known placement.

    Requiring a *known* placement keeps containers-of-datasets
    (``data/hdf5_data/{dataset_full,...}/*.hdf5``) out of the labeled layout,
    where their dataset names would be mistaken for carrying types. Once a
    directory qualifies, :func:`resolve_input_dir` takes every recording
    subfolder it holds, so a new placement needs no registration as long as it
    sits alongside a known one.
    """
    return any(
        normalize_placement(osp.basename(sub)) in CARRYING_ALIASES
        for sub in list_label_dirs(path)
    )


def classify_dir(path):
    """Return one of pair / preprocessed / flat / labeled / pair_batch."""
    if not osp.isdir(path):
        raise NotADirectoryError(path)
    if is_pair_dir(path):
        return "pair"
    if is_preprocessed_dir(path):
        return "preprocessed"
    if list_recording_files(path):
        return "flat"
    if is_labeled_dir(path):
        return "labeled"
    if discover_pair_dirs(path):
        return "pair_batch"
    raise ValueError(
        f"{path} holds no .hdf5/.parquet files, no pair folders with {REAL_NAME}, "
        f"no class subfolders and no preprocessed {STATS_NAME}"
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_recording_files(input_dir):
    """Top-level ``.hdf5`` / ``.parquet`` files, sorted."""
    return [
        osp.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if is_recording_file(osp.join(input_dir, name))
    ]


def discover_pairs_from_manifest(root):
    """Pair ids listed as ``status=generated`` in ``manifest.csv``, or None."""
    manifest = osp.join(root, MANIFEST_NAME)
    if not osp.isfile(manifest):
        return None
    with open(manifest, newline="") as f:
        pairs = [
            row["pair_id"].strip()
            for row in csv.DictReader(f)
            if row.get("status", "").strip() == "generated" and row.get("pair_id")
        ]
    return sorted(pairs)


def discover_pair_dirs(root, require_synthetic=False):
    """Absolute pair directories under ``root``.

    ``manifest.csv`` is authoritative when present, so a dataset can exclude
    pairs whose generation failed; otherwise walk for ``real.hdf5``.
    """
    ids = discover_pairs_from_manifest(root)
    if ids is not None:
        dirs = [osp.join(root, pair_id) for pair_id in ids]
        return [d for d in dirs if osp.isdir(d) and
                (not require_synthetic or _sibling_sim(osp.join(d, REAL_NAME)))]
    found = []
    for current, _dirs, files in os.walk(root):
        if REAL_NAME not in files:
            continue
        if require_synthetic and SYNTHETIC_NAME not in files:
            continue
        found.append(current)
    return sorted(found)


def output_name(path, root):
    """Flattened, collision-free subdirectory name for one input under a root."""
    rel = osp.splitext(osp.relpath(path, root))[0]
    # A pair dir names its recording real.hdf5, so the directory carries identity.
    if osp.basename(rel) == "real":
        rel = osp.dirname(rel)
    return rel.replace(os.sep, "_") or osp.basename(osp.abspath(root))


# ---------------------------------------------------------------------------
# Recording construction
# ---------------------------------------------------------------------------

def _sibling_sim(path):
    """Synthetic parquet next to a recording, if the pair layout provides one."""
    directory = osp.dirname(osp.abspath(path))
    for candidate in (
        osp.join(directory, SYNTHETIC_NAME),
        osp.join(directory, ".run", "synthetic_0000.parquet"),
        osp.join(directory, ".run", SYNTHETIC_NAME),
    ):
        if osp.isfile(candidate):
            return candidate
    return None


def _sibling_trajectory(pair_dir):
    candidate = osp.join(pair_dir, TRAJECTORY_NAME)
    return candidate if osp.isfile(candidate) else None


def recording_from_pair_dir(pair_dir, name=None):
    """Build a :class:`Recording` for a ``real.hdf5`` pair directory."""
    pair_dir = pair_dir.rstrip("/")
    real = osp.join(pair_dir, REAL_NAME)
    if not osp.isfile(real):
        raise FileNotFoundError(f"{pair_dir} is a directory but has no {REAL_NAME}")
    return Recording(
        name=name or osp.basename(pair_dir),
        path=real,
        sim_path=_sibling_sim(real),
        trajectory_path=_sibling_trajectory(pair_dir),
        pair_dir=pair_dir,
        label=resolve_gt_class(real),
    )


def recording_from_file(path, name=None, sim_path=None):
    """Build a :class:`Recording` for a single ``.hdf5`` / ``.parquet`` file."""
    return Recording(
        name=name or osp.splitext(osp.basename(path))[0],
        path=path,
        sim_path=sim_path if sim_path is not None else _sibling_sim(path),
        trajectory_path=None,
        pair_dir=osp.dirname(path) if osp.basename(path) == REAL_NAME else None,
        label=resolve_gt_class(path),
    )


def resolve_input(path, sim_path=None):
    """One :class:`Recording` from ``--input`` (a file or a pair directory)."""
    if osp.isdir(path):
        recording = recording_from_pair_dir(path)
        if sim_path is not None:
            recording.sim_path = sim_path
        return recording
    if not osp.isfile(path):
        raise FileNotFoundError(f"--input not found: {path}")
    return recording_from_file(path, sim_path=sim_path)


def resolve_input_dir(input_dir, pairs_only=False):
    """Every :class:`Recording` under ``--input_dir``, plus the detected layout.

    Returns (layout, recordings). ``pairs_only`` restricts discovery to pair
    directories, which is what the real/sim comparisons need.
    """
    input_dir = input_dir.rstrip("/")
    layout = classify_dir(input_dir)

    if layout == "pair":
        return layout, [recording_from_pair_dir(input_dir)]

    if pairs_only or layout == "pair_batch":
        pair_dirs = discover_pair_dirs(input_dir)
        if not pair_dirs:
            raise ValueError(f"No pair directories with {REAL_NAME} under {input_dir}")
        return "pair_batch", [
            recording_from_pair_dir(
                pair_dir, name=output_name(pair_dir, input_dir),
            )
            for pair_dir in pair_dirs
        ]

    if layout == "flat":
        files = list_recording_files(input_dir)
        return layout, [
            recording_from_file(
                path, name=output_name(path, input_dir),
            )
            for path in files
        ]

    if layout == "labeled":
        recordings = []
        for cls_dir in list_label_dirs(input_dir):
            # The folder name is authoritative here, so an unregistered
            # placement (head, ...) keeps its own name instead of going unlabelled.
            label = normalize_placement(osp.basename(cls_dir))
            for name in sorted(os.listdir(cls_dir)):
                path = osp.join(cls_dir, name)
                if osp.splitext(name)[1].lower() in HDF5_EXTS and osp.isfile(path):
                    recordings.append(Recording(
                        name=output_name(path, input_dir), path=path, label=label,
                    ))
        return layout, recordings

    raise ValueError(
        f"{input_dir} looks like a preprocessed dataset; pass it as --dataset_dir"
    )


# ---------------------------------------------------------------------------
# Carrying placement from paths
#
# These return dataset placements, never filtered against a classifier's
# vocabulary. Filter with scoreable_label() at the point of scoring instead.
# ---------------------------------------------------------------------------

def normalize_placement(name):
    """Canonical placement for an explicit label, e.g. a labeled-layout folder.

    Known spellings collapse through :data:`CARRYING_ALIASES`; anything else is
    slugified and returned as-is, so a new placement works with no registration.
    """
    slug = "_".join(w for w in re.split(r"[^a-z0-9]+", str(name).lower()) if w)
    return CARRYING_ALIASES.get(slug, slug) or None


def infer_placement_from_name(path):
    """Best-effort placement from one filename or directory name, or None.

    Handles flat gen_world stems (``left_pocket_richa_1_slam.hdf5``) and the
    carrying-type directories of the pair datasets. A placement ending on a
    token boundary wins; a bare prefix is the fallback for older exports that
    ran placements together (``pocketrun3``). Unlike
    :func:`normalize_placement` this has to recognise the placement inside a
    longer name, so it can only match registered aliases.
    """
    stem = osp.splitext(osp.basename(str(path).rstrip("/\\")))[0].lower()
    for boundary in (True, False):
        for prefix in CARRYING_ALIASES_BY_LENGTH:
            hit = (stem == prefix or stem.startswith(prefix + "_")) if boundary \
                else stem.startswith(prefix)
            if hit:
                return CARRYING_ALIASES[prefix]
    return None


def infer_placement_in_name(name):
    """Placement named anywhere inside a name, not only at its start.

    Sequence names bury the placement in the middle
    (``john_left_pocket_corrected``), so prefix matching alone would miss it.
    Longest placement wins, and matches must cover whole words so
    ``richa_hip_corrected`` stays unlabelled.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", str(name).lower()) if w]
    for token in CARRYING_ALIASES_BY_LENGTH:
        parts = token.split("_")
        span = len(parts)
        if any(words[i:i + span] == parts for i in range(len(words) - span + 1)):
            return CARRYING_ALIASES[token]
    return None


def resolve_gt_class(path):
    """Carrying placement for a recording, from its filename or parent folders.

    Covers the flat gen_world layout (``chest_richa_2_slam.hdf5``) and the pair
    layout (``.../chest/richa_2_slam/real.hdf5``), whose filename carries no
    label so the carrying-type directory has to supply it.
    """
    for part in [path] + osp.abspath(str(path)).split(os.sep)[-2::-1]:
        cls = infer_placement_from_name(part)
        if cls is not None:
            return cls
    return None


def scoreable_label(label, class_names):
    """``label`` when a classifier can be scored against it, else None.

    Placements outside a classifier's vocabulary (a head recording against a
    four-class model) are still evaluated and plotted; they just cannot
    contribute to accuracy or a confusion matrix.
    """
    return label if label in set(class_names or ()) else None


@functools.lru_cache(maxsize=None)
def _native_holding_map(manifest):
    """``native_holding`` per pair_id, read once per manifest."""
    table = {}
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            native = (row.get("native_holding") or "").strip()
            if native and row.get("pair_id"):
                table[row["pair_id"]] = native
    return table


def find_manifest(path):
    """Nearest ``manifest.csv`` above a recording, or None."""
    directory = osp.dirname(osp.abspath(path))
    for _ in range(4):
        directory = osp.dirname(directory)
        candidate = osp.join(directory, MANIFEST_NAME)
        if osp.isfile(candidate):
            return candidate
    return None


def infer_native_gt_class(real_path):
    """Carrying type the real recording actually captured, or None if unknown.

    A pair folder is named for the placement the simulator re-rendered, which
    usually is not how the phone was really carried, so real IMU has to be
    scored against the manifest's ``native_holding`` (or a placement spelled out
    in the sequence name) instead.
    """
    abspath = osp.abspath(real_path)
    parts = abspath.split(os.sep)
    manifest = find_manifest(abspath) if len(parts) >= 3 else None
    if manifest is not None:
        native = _native_holding_map(manifest).get("/".join(parts[-3:-1]))
        if native:
            cls = normalize_placement(native)
            if cls:
                return cls
    return infer_placement_in_name(osp.basename(osp.dirname(abspath)))
