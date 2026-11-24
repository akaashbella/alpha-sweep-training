"""Stage 3 document schema over the existing exclusive publication primitives."""

from pathlib import Path

from alpha_sweep_training.serialization import read_json
from analysis.layer_sensitivity_stage2.storage import disjoint, identity, install, lease, plain, sha_file

CHECKOUT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


class Stage3Error(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Stage3Error(message)


def seal(kind, payload):
    require(not {"stage3_version", "kind", "document_sha256"} & payload.keys(), "Reserved document fields")
    document = {"stage3_version": 1, "kind": kind, **payload}
    return {**document, "document_sha256": identity(document)}


def read(path, kind):
    document = read_json(plain(path))
    require(type(document) is dict and type(document.get("stage3_version")) is int and
            document["stage3_version"] == 1 and document.get("kind") == kind,
            f"Unsupported Stage 3 {kind} document: {path}")
    require(document.get("document_sha256") == identity({k: v for k, v in document.items() if k != "document_sha256"}),
            f"Document checksum mismatch: {path}")
    return document


def finish(root, kind, names):
    files = {name: sha_file(root / name) for name in sorted(names)}
    install(root / "COMPLETE.json", seal("publication", {"publication_kind": kind, "files": files}))


def publication(root, kind):
    root = plain(root)
    receipt = read(root / "COMPLETE.json", "publication")
    primary = f"{kind}.json"
    require(receipt["publication_kind"] == kind and primary in receipt["files"], f"Incomplete {kind} publication")
    for name, digest in receipt["files"].items():
        path = plain(root / name)
        require(root in path.parents and path.relative_to(root).as_posix() == name and sha_file(path) == digest,
                f"Publication file mismatch: {name}")
    return read(root / primary, kind)


def publish(output, protected, kind, document):
    root = disjoint(output, protected)
    root.mkdir(parents=True)
    install(root / f"{kind}.json", document)
    finish(root, kind, [f"{kind}.json"])
    return root


def code_digest():
    """Include the read-only helpers invoked from both previous stages."""
    sets = (PACKAGE, PACKAGE.parent / "layer_sensitivity_stage2", PACKAGE.parent / "loss_geometry_stage1")
    return identity({p.relative_to(CHECKOUT).as_posix(): sha_file(p) for directory in sets for p in sorted(directory.glob("*.py"))})
