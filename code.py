import argparse
import bisect
import csv
import json
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_STANZA_VERSION = "1.13.0"
LANGUAGE = "ru"
PACKAGE = "syntagrus"
PROCESSORS = "tokenize,pos,lemma,depparse"
OUTPUT_SUFFIXES = (".sentences.jsonl", ".tokens.tsv", ".conllu")


@dataclass(frozen=True)
class ParagraphSpan:
    """Character span of a non-empty paragraph in the normalized document."""

    paragraph_id: int
    start_char: int
    end_char: int


def parse_preprocess_legacy_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate Russian UTF-8 policy texts with Stanza 1.13.0 and the "
            "Russian SynTagRus models (tokenize, POS/morphology, lemma, depparse)."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A UTF-8 .txt file or a directory containing .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("stanza_output"),
        help="Output directory (default: ./stanza_output).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories when input_path is a directory.",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download/update the Russian SynTagRus model before processing.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional Stanza model directory. The Stanza default is used otherwise.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device (default: auto).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Input encoding (default: utf-8).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    return parser.parse_preprocess_legacy_args()


def import_stanza() -> Any:
    try:
        import stanza  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Stanza is not installed. Run:\n"
            '  python3 -m pip install "stanza==1.13.0"'
        ) from exc

    installed = getattr(stanza, "__version__", "unknown")
    if installed != EXPECTED_STANZA_VERSION:
        print(
            f"WARNING: manuscript version is Stanza {EXPECTED_STANZA_VERSION}, "
            f"but installed version is {installed}.",
            file=sys.stderr,
        )
    return stanza


def resolve_use_gpu(device: str) -> bool:
    """Resolve auto/cpu/cuda without adding a direct PyTorch dependency."""

    if device == "cpu":
        return False

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if device == "cuda" and not cuda_available:
        raise SystemExit(
            "--device cuda was requested, but a CUDA-enabled PyTorch device is unavailable."
        )
    return cuda_available


def download_models(stanza: Any, model_dir: Path | None) -> None:
    kwargs: dict[str, Any] = {
        "lang": LANGUAGE,
        "package": PACKAGE,
        "processors": PROCESSORS,
        "verbose": True,
    }
    if model_dir is not None:
        kwargs["model_dir"] = str(model_dir)
    stanza.download(**kwargs)


def build_pipeline(stanza: Any, model_dir: Path | None, use_gpu: bool) -> Any:
    kwargs: dict[str, Any] = {
        "lang": LANGUAGE,
        "package": PACKAGE,
        "processors": PROCESSORS,
        "use_gpu": use_gpu,
        "verbose": False,
        # Models are downloaded explicitly with --download-model. Disabling
        # implicit updates makes reruns use the same local model files.
        "download_method": None,
    }
    if model_dir is not None:
        kwargs["model_dir"] = str(model_dir)

    try:
        return stanza.Pipeline(**kwargs)
    except Exception as exc:
        raise SystemExit(
            "Unable to load the Russian SynTagRus pipeline. On the first run, add "
            "--download-model. Original error:\n"
            f"  {type(exc).__name__}: {exc}"
        ) from exc


def discover_input_files(input_path: Path, recursive: bool) -> tuple[list[Path], Path]:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".txt":
            raise SystemExit(f"Input file must have a .txt extension: {input_path}")
        return [input_path], input_path.parent

    pattern = "**/*.txt" if recursive else "*.txt"
    files = sorted(p for p in input_path.glob(pattern) if p.is_file())
    if not files:
        scope = "recursively" if recursive else "in the top directory"
        raise SystemExit(f"No .txt files found {scope}: {input_path}")
    return files, input_path


def read_normalized_text(path: Path, encoding: str) -> str:
    text = path.read_text(encoding=encoding)
    text = text.lstrip("\ufeff")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def paragraph_spans(text: str) -> list[ParagraphSpan]:
    """Identify paragraphs separated by one or more blank lines."""

    spans: list[ParagraphSpan] = []
    cursor = 0
    paragraph_id = 1
    for block in re.split(r"\n[ \t]*\n+", text):
        if not block:
            cursor += 1
            continue

        found_at = text.find(block, cursor)
        if found_at < 0:
            found_at = cursor
        cursor = found_at + len(block)

        leading = len(block) - len(block.lstrip())
        trailing = len(block) - len(block.rstrip())
        start = found_at + leading
        end = found_at + len(block) - trailing
        if start < end:
            spans.append(ParagraphSpan(paragraph_id, start, end))
            paragraph_id += 1
    return spans


def paragraph_for_offset(
    offset: int | None,
    spans: Sequence[ParagraphSpan],
    starts: Sequence[int],
) -> int | None:
    if offset is None or not spans:
        return None

    index = bisect.bisect_right(starts, offset) - 1
    if index >= 0 and offset < spans[index].end_char:
        return spans[index].paragraph_id

    # This only occurs when a token offset falls in inter-paragraph whitespace.
    if index + 1 < len(spans):
        return spans[index + 1].paragraph_id
    return spans[-1].paragraph_id


def value_or_underscore(value: Any) -> str:
    if value is None or value == "":
        return "_"
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def sentence_offsets(sentence: Any) -> tuple[int | None, int | None]:
    starts = [getattr(token, "start_char", None) for token in sentence.tokens]
    ends = [getattr(token, "end_char", None) for token in sentence.tokens]
    starts = [value for value in starts if isinstance(value, int)]
    ends = [value for value in ends if isinstance(value, int)]
    return (min(starts) if starts else None, max(ends) if ends else None)


def word_offset_map(sentence: Any) -> dict[int, tuple[int | None, int | None]]:
    offsets: dict[int, tuple[int | None, int | None]] = {}
    for token in sentence.tokens:
        start = getattr(token, "start_char", None)
        end = getattr(token, "end_char", None)
        for word in token.words:
            word_id = getattr(word, "id", None)
            if isinstance(word_id, int):
                offsets[word_id] = (start, end)
    return offsets


def sentence_record(
    sentence: Any,
    document_id: str,
    source_file: str,
    sentence_id: int,
    spans: Sequence[ParagraphSpan],
    starts: Sequence[int],
) -> dict[str, Any]:
    sent_start, sent_end = sentence_offsets(sentence)
    paragraph_id = paragraph_for_offset(sent_start, spans, starts)
    offsets = word_offset_map(sentence)
    words: list[dict[str, Any]] = []

    for word in sentence.words:
        word_id = getattr(word, "id", None)
        start_char, end_char = offsets.get(word_id, (None, None))
        words.append(
            {
                "id": word_id,
                "text": getattr(word, "text", None),
                "lemma": getattr(word, "lemma", None),
                "upos": getattr(word, "upos", None),
                "xpos": getattr(word, "xpos", None),
                "feats": getattr(word, "feats", None),
                "head": getattr(word, "head", None),
                "deprel": getattr(word, "deprel", None),
                "deps": getattr(word, "deps", None),
                "misc": getattr(word, "misc", None),
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    sentence_text = getattr(sentence, "text", None)
    if sentence_text is None:
        sentence_text = " ".join(str(item["text"] or "") for item in words).strip()

    return {
        "schema_version": "1.0",
        "document_id": document_id,
        "source_file": source_file,
        "paragraph_id": paragraph_id,
        "sentence_id": sentence_id,
        "start_char": sent_start,
        "end_char": sent_end,
        "text": sentence_text,
        "words": words,
    }


def make_records(doc: Any, document_id: str, source_file: str, text: str) -> list[dict[str, Any]]:
    spans = paragraph_spans(text)
    starts = [span.start_char for span in spans]
    return [
        sentence_record(sentence, document_id, source_file, index, spans, starts)
        for index, sentence in enumerate(doc.sentences, start=1)
    ]


def output_paths(output_dir: Path, relative_input: Path) -> tuple[Path, Path, Path]:
    base = output_dir / relative_input.parent / relative_input.stem
    return tuple(Path(str(base) + suffix) for suffix in OUTPUT_SUFFIXES)  # type: ignore[return-value]


def preprocess_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def preprocess_write_tsv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    columns = (
        "document_id",
        "source_file",
        "paragraph_id",
        "sentence_id",
        "word_id",
        "text",
        "lemma",
        "upos",
        "xpos",
        "feats",
        "head",
        "deprel",
        "deps",
        "misc",
        "start_char",
        "end_char",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for record in records:
            common = {
                "document_id": record["document_id"],
                "source_file": record["source_file"],
                "paragraph_id": record["paragraph_id"],
                "sentence_id": record["sentence_id"],
            }
            for word in record["words"]:
                writer.writerow(
                    {
                        **common,
                        "word_id": word["id"],
                        "text": word["text"],
                        "lemma": word["lemma"],
                        "upos": word["upos"],
                        "xpos": word["xpos"],
                        "feats": word["feats"],
                        "head": word["head"],
                        "deprel": word["deprel"],
                        "deps": word["deps"],
                        "misc": word["misc"],
                        "start_char": word["start_char"],
                        "end_char": word["end_char"],
                    }
                )


def write_conllu(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            if index == 0:
                handle.write(f"# newdoc id = {record['document_id']}\n")
            handle.write(f"# sent_id = {record['document_id']}:{record['sentence_id']}\n")
            if record["paragraph_id"] is not None:
                handle.write(f"# paragraph_id = {record['paragraph_id']}\n")
            handle.write(f"# text = {value_or_underscore(record['text'])}\n")

            for word in record["words"]:
                fields = (
                    word["id"],
                    word["text"],
                    word["lemma"],
                    word["upos"],
                    word["xpos"],
                    word["feats"],
                    word["head"],
                    word["deprel"],
                    word["deps"],
                    word["misc"],
                )
                handle.write("\t".join(value_or_underscore(field) for field in fields) + "\n")
            handle.write("\n")


def process_file(
    nlp: Any,
    input_file: Path,
    input_root: Path,
    output_dir: Path,
    encoding: str,
    overwrite: bool,
) -> dict[str, Any]:
    relative_input = input_file.relative_to(input_root)
    jsonl_path, tsv_path, conllu_path = output_paths(output_dir, relative_input)
    products = (jsonl_path, tsv_path, conllu_path)

    if not overwrite and all(path.exists() for path in products):
        return {
            "source_file": relative_input.as_posix(),
            "status": "skipped",
            "reason": "all output files already exist",
        }

    for path in products:
        path.parent.mkdir(parents=True, exist_ok=True)

    text = read_normalized_text(input_file, encoding)
    if not text.strip():
        raise ValueError("input file contains no non-whitespace text")

    annotated = nlp(text)
    document_id = relative_input.with_suffix("").as_posix()
    records = make_records(annotated, document_id, relative_input.as_posix(), text)

    preprocess_write_jsonl(jsonl_path, records)
    preprocess_write_tsv(tsv_path, records)
    write_conllu(conllu_path, records)

    return {
        "source_file": relative_input.as_posix(),
        "status": "processed",
        "characters": len(text),
        "paragraphs": len(paragraph_spans(text)),
        "sentences": len(records),
        "words": sum(len(record["words"]) for record in records),
        "outputs": {
            "jsonl": str(jsonl_path.resolve()),
            "tsv": str(tsv_path.resolve()),
            "conllu": str(conllu_path.resolve()),
        },
    }


def preprocess_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def preprocess_legacy_main() -> int:
    args = parse_preprocess_legacy_args()
    stanza = import_stanza()
    use_gpu = resolve_use_gpu(args.device)

    if args.download_model:
        download_models(stanza, args.model_dir)

    nlp = build_pipeline(stanza, args.model_dir, use_gpu)
    files, input_root = discover_input_files(args.input_path, args.recursive)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "started_at_utc": preprocess_utc_now(),
        "stanza_version": getattr(stanza, "__version__", "unknown"),
        "language": LANGUAGE,
        "model_package": PACKAGE,
        "processors": PROCESSORS.split(","),
        "universal_dependencies": True,
        "device": "cuda" if use_gpu else "cpu",
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "documents": [],
    }
    error_lines: list[str] = []

    total = len(files)
    for index, input_file in enumerate(files, start=1):
        relative = input_file.relative_to(input_root).as_posix()
        print(f"[{index}/{total}] {relative}", flush=True)
        try:
            result = process_file(
                nlp=nlp,
                input_file=input_file,
                input_root=input_root,
                output_dir=output_dir,
                encoding=args.encoding,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            result = {
                "source_file": relative,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            error_lines.append(
                f"[{preprocess_utc_now()}] {relative}\n"
                f"{traceback.format_exc()}\n"
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        summary["documents"].append(result)

    summary["finished_at_utc"] = preprocess_utc_now()
    summary["counts"] = {
        "total": total,
        "processed": sum(d["status"] == "processed" for d in summary["documents"]),
        "skipped": sum(d["status"] == "skipped" for d in summary["documents"]),
        "errors": sum(d["status"] == "error" for d in summary["documents"]),
    }

    summary_path = output_dir / "processing_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    errors_path = output_dir / "errors.log"
    errors_path.write_text(
        "\n".join(error_lines) if error_lines else "No processing errors.\n",
        encoding="utf-8",
    )

    counts = summary["counts"]
    print(
        "Done: "
        f"processed={counts['processed']}, skipped={counts['skipped']}, "
        f"errors={counts['errors']}"
    )
    print(f"Summary: {summary_path}")
    return 1 if counts["errors"] else 0



# ===== KG EXTRACTION COMPONENT =====
import argparse
import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCRIPT_VERSION = "1.0.0"
DEFAULT_CONFIG = Path(__file__).with_name("ru_kg_rules.json")
OUTPUT_FILENAMES = (
    "triples.jsonl",
    "triples.tsv",
    "nodes.tsv",
    "edges_aggregated.tsv",
    "filtered_sentences.jsonl",
    "rejected_candidates.jsonl",
    "learned_abbreviations.tsv",
    "manual_validation_sample.tsv",
    "extraction_summary.json",
)

NOMINAL_POS = {"NOUN", "PROPN", "PRON", "ADJ", "NUM"}
SUBJECT_DEPS = {"nsubj", "csubj"}
CORE_OBJECT_DEPS = {"obj", "iobj"}
RUSSIAN_FUNCTION_WORDS = {
    "в",
    "во",
    "на",
    "по",
    "к",
    "ко",
    "из",
    "от",
    "до",
    "для",
    "и",
    "или",
    "а",
    "с",
    "со",
    "о",
    "об",
    "обо",
    "при",
}


@dataclass(frozen=True)
class Word:
    id: int
    text: str
    lemma: str
    upos: str
    xpos: str | None
    feats: str | None
    head: int
    deprel: str
    deps: str | None
    misc: str | None
    start_char: int | None
    end_char: int | None

    @property
    def dep_base(self) -> str:
        return self.deprel.split(":", 1)[0]

    @property
    def is_passive_subject(self) -> bool:
        return self.deprel.startswith("nsubj:pass") or self.deprel.startswith("csubj:pass")

    @property
    def feature_set(self) -> set[str]:
        return set((self.feats or "").split("|"))


class ParsedSentence:
    def __init__(self, record: Mapping[str, Any]):
        self.record = dict(record)
        self.document_id = str(record.get("document_id") or "")
        self.source_file = str(record.get("source_file") or "")
        self.paragraph_id = record.get("paragraph_id")
        self.sentence_id = record.get("sentence_id")
        self.text = str(record.get("text") or "")
        self.start_char = record.get("start_char")
        self.end_char = record.get("end_char")
        self.words: dict[int, Word] = {}
        self.children: dict[int, list[int]] = defaultdict(list)

        for raw in record.get("words") or []:
            try:
                word_id = int(raw["id"])
                head = int(raw.get("head") or 0)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid word id/head in sentence {self.sentence_id}: {raw}") from exc
            word = Word(
                id=word_id,
                text=str(raw.get("text") or ""),
                lemma=str(raw.get("lemma") or raw.get("text") or ""),
                upos=str(raw.get("upos") or ""),
                xpos=raw.get("xpos"),
                feats=raw.get("feats"),
                head=head,
                deprel=str(raw.get("deprel") or ""),
                deps=raw.get("deps"),
                misc=raw.get("misc"),
                start_char=raw.get("start_char"),
                end_char=raw.get("end_char"),
            )
            self.words[word_id] = word
            self.children[head].append(word_id)

        for child_ids in self.children.values():
            child_ids.sort()

    def child_words(self, head_id: int) -> list[Word]:
        return [self.words[word_id] for word_id in self.children.get(head_id, [])]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().replace("ё", "е")
    value = value.replace("–", "-").replace("—", "-").replace("‑", "-")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def normalize_for_regex(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip()


def normalize_abbreviation(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").upper().replace("Ё", "Е")
    return re.sub(r"[^0-9A-ZА-Я]", "", value)


def looks_like_abbreviation(value: str) -> bool:
    stripped = re.sub(r"[.\-–—\s]", "", value or "")
    letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", stripped)
    return bool(2 <= len(letters) <= 16 and letters == letters.upper())


def stable_hash_id(prefix: str, value: str, length: int = 14) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:length].upper()
    return f"{prefix}_{digest}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_year(*values: str) -> int | None:
    for value in values:
        match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", value or "")
        if match:
            return int(match.group(1))
    return None


def parse_kg_legacy_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Russian entities and extract auditable SPO triples from Stanza JSONL."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract and normalize SPO triples.")
    extract.add_argument("input_path", type=Path, help="A .sentences.jsonl file or directory.")
    extract.add_argument("--output-dir", "-o", type=Path, required=True)
    extract.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    extract.add_argument("--recursive", action="store_true")
    extract.add_argument(
        "--argument-scope",
        choices=("core", "extended"),
        default="core",
        help="core=obj/iobj; extended also permits one or more obl arguments when no core object exists.",
    )
    extract.add_argument(
        "--include-zero-cop",
        action="store_true",
        help="Infer an IS_A relation for nominal predicates without an overt copula.",
    )
    extract.add_argument("--validation-sample-size", type=int, default=200)
    extract.add_argument("--seed", type=int, default=202503)
    extract.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate-validation",
        help="Calculate sampled precision/error metrics from a manually completed TSV.",
    )
    evaluate.add_argument("validation_file", type=Path)
    evaluate.add_argument("--output", "-o", type=Path, required=True)
    return parser.parse_kg_legacy_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON configuration {path}: {exc}") from exc

    entity_ids = [item["id"] for item in config.get("entities", [])]
    if len(entity_ids) != len(set(entity_ids)):
        raise SystemExit("Configuration contains duplicate entity ids.")
    known_ids = set(entity_ids)
    for item in config.get("abbreviations", []):
        if item.get("entity_id") not in known_ids:
            raise SystemExit(f"Unknown abbreviation entity_id: {item.get('entity_id')}")
    for item in config.get("ambiguous_abbreviations", []):
        for candidate in item.get("candidates", []):
            if candidate.get("entity_id") not in known_ids:
                raise SystemExit(f"Unknown ambiguous abbreviation entity_id: {candidate.get('entity_id')}")

    for rule in config.get("boilerplate_rules", []) + config.get("entity_quality_rules", []):
        try:
            re.compile(rule["pattern"], re.IGNORECASE)
        except (KeyError, re.error) as exc:
            raise SystemExit(f"Invalid regex rule {rule.get('id')}: {exc}") from exc
    config["_resolved_path"] = str(path)
    config["_sha256"] = file_sha256(path)
    return config


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def discover_inputs(input_path: Path, recursive: bool, output_dir: Path) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if not input_path.name.endswith(".sentences.jsonl"):
            raise SystemExit("Input file must end with .sentences.jsonl")
        return [input_path]

    pattern = "**/*.sentences.jsonl" if recursive else "*.sentences.jsonl"
    files = sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file() and not is_within(path, output_dir)
    )
    if not files:
        raise SystemExit(f"No *.sentences.jsonl files found in: {input_path}")
    return files


def read_jsonl(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if "words" not in record:
                    raise ValueError(f"Missing words array at {path}:{line_number}")
                record["_input_jsonl"] = str(path)
                record["_input_line"] = line_number
                yield record


class FormulaFilter:
    def __init__(self, config: Mapping[str, Any]):
        self.domain_terms = [normalize_text(item) for item in config.get("domain_guard_terms", [])]
        self.sentence_rules = [
            (rule, re.compile(rule["pattern"], re.IGNORECASE))
            for rule in config.get("boilerplate_rules", [])
        ]
        self.entity_rules = [
            (rule, re.compile(rule["pattern"], re.IGNORECASE))
            for rule in config.get("entity_quality_rules", [])
        ]

    def sentence_decision(self, text: str) -> tuple[list[str], list[str]]:
        regex_text = normalize_for_regex(text)
        domain_text = normalize_text(text)
        has_domain_term = any(term and term in domain_text for term in self.domain_terms)
        excluded: list[str] = []
        flags: list[str] = []
        for rule, pattern in self.sentence_rules:
            if not pattern.search(regex_text):
                continue
            if rule.get("requires_absence_of_domain_terms") and has_domain_term:
                flags.append(f"{rule['id']}:DOMAIN_GUARD")
                continue
            if rule.get("action") == "exclude":
                excluded.append(rule["id"])
            else:
                flags.append(rule["id"])
        return excluded, flags

    def entity_decision(self, surface: str, lemma: str) -> tuple[list[str], list[str]]:
        candidates = {normalize_for_regex(surface), normalize_for_regex(lemma)}
        rejected: list[str] = []
        flags: list[str] = []
        for rule, pattern in self.entity_rules:
            if not any(pattern.fullmatch(candidate) for candidate in candidates if candidate):
                continue
            if rule.get("action") == "reject":
                rejected.append(rule["id"])
            else:
                flags.append(rule["id"])
        return rejected, flags


def abbreviation_initials(words: Sequence[str]) -> str:
    selected = [word for word in words if normalize_text(word) not in RUSSIAN_FUNCTION_WORDS]
    return normalize_abbreviation("".join(word[0] for word in selected if word))


def learn_parenthetical_abbreviations(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"([А-ЯЁA-Z][А-ЯЁа-яёA-Za-z-]*(?:\s+[А-ЯЁа-яёA-Za-z-]+){1,11})"
        r"\s*\(([А-ЯЁA-Z][А-ЯЁA-Z0-9.\-]{1,15})\)"
    )
    learned: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        long_block = match.group(1).strip()
        abbreviation = normalize_abbreviation(match.group(2))
        words = re.findall(r"[А-ЯЁа-яёA-Za-z-]+", long_block)
        best: str | None = None
        for width in range(2, min(len(words), 10) + 1):
            candidate_words = words[-width:]
            if abbreviation_initials(candidate_words) == abbreviation:
                best = " ".join(candidate_words)
        if best:
            learned.append((abbreviation, best))
    return learned


class EntityNormalizer:
    def __init__(self, config: Mapping[str, Any]):
        self.entities = {item["id"]: dict(item) for item in config.get("entities", [])}
        self.alias_index: dict[str, set[str]] = defaultdict(set)
        for entity_id, item in self.entities.items():
            for alias in (
                [item.get("canonical", "")]
                + list(item.get("surface_aliases", []))
                + list(item.get("lemma_aliases", []))
            ):
                normalized = normalize_text(alias)
                if normalized:
                    self.alias_index[normalized].add(entity_id)

        self.abbreviations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in config.get("abbreviations", []):
            self.abbreviations[normalize_abbreviation(item["abbr"])].append(dict(item))
        self.ambiguous_abbreviations = {
            normalize_abbreviation(item["abbr"]): dict(item)
            for item in config.get("ambiguous_abbreviations", [])
        }
        self.learned: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.learned_evidence: list[dict[str, Any]] = []

    def learn_from_record(self, record: Mapping[str, Any]) -> None:
        document_id = str(record.get("document_id") or record.get("source_file") or "")
        for abbreviation, long_form in learn_parenthetical_abbreviations(str(record.get("text") or "")):
            self.learned[document_id][abbreviation].add(long_form)
            self.learned_evidence.append(
                {
                    "document_id": document_id,
                    "source_file": record.get("source_file"),
                    "sentence_id": record.get("sentence_id"),
                    "abbreviation": abbreviation,
                    "long_form": long_form,
                    "sentence_text": record.get("text"),
                }
            )

    def _context_matches(self, rule: Mapping[str, Any], context_norm: str) -> bool:
        terms = [normalize_text(item) for item in rule.get("context_any", [])]
        return not terms or any(term in context_norm for term in terms)

    def _resolve_abbreviation(self, abbreviation: str, context: str) -> tuple[str | None, list[str]]:
        key = normalize_abbreviation(abbreviation)
        context_norm = normalize_text(context)
        matches = [
            item["entity_id"]
            for item in self.abbreviations.get(key, [])
            if self._context_matches(item, context_norm)
        ]
        if len(set(matches)) == 1:
            return matches[0], []
        if len(set(matches)) > 1:
            return None, ["AMBIGUOUS_STATIC_ABBREVIATION"]

        ambiguous = self.ambiguous_abbreviations.get(key)
        if ambiguous:
            candidates = [
                item["entity_id"]
                for item in ambiguous.get("candidates", [])
                if self._context_matches(item, context_norm)
            ]
            if len(set(candidates)) == 1:
                return candidates[0], []
            return None, ["AMBIGUOUS_CONTEXT_ABBREVIATION"]
        return None, []

    def _canonical_lemma(self, entity_id: str) -> str:
        entity = self.entities[entity_id]
        aliases = entity.get("lemma_aliases") or [entity.get("canonical", "")]
        return normalize_text(aliases[0])

    def _expand_abbreviations_in_phrase(self, phrase: str, context: str) -> str:
        expanded: list[str] = []
        for token in phrase.split():
            entity_id, flags = self._resolve_abbreviation(token, context)
            if entity_id and not flags:
                expanded.extend(self._canonical_lemma(entity_id).split())
            else:
                expanded.append(token)
        return " ".join(expanded)

    def _result_from_entity(self, entity_id: str, source: str, flags: Sequence[str] = ()) -> dict[str, Any]:
        entity = self.entities[entity_id]
        return {
            "entity_id": entity_id,
            "canonical": entity["canonical"],
            "entity_type": entity.get("entity_type", "unclassified"),
            "concept_group": entity.get("concept_group"),
            "mapping_source": source,
            "alignment_flags": list(flags),
        }

    def align(
        self,
        surface: str,
        lemma: str,
        sentence_text: str,
        document_id: str,
    ) -> dict[str, Any]:
        flags: list[str] = []
        if looks_like_abbreviation(surface):
            entity_id, abbreviation_flags = self._resolve_abbreviation(surface, sentence_text)
            flags.extend(abbreviation_flags)
            if entity_id:
                return self._result_from_entity(entity_id, "abbreviation_dictionary", flags)

            key = normalize_abbreviation(surface)
            learned_forms = self.learned.get(document_id, {}).get(key, set())
            if len(learned_forms) == 1:
                learned_form = next(iter(learned_forms))
                learned_norm = normalize_text(learned_form)
                entity_ids = self.alias_index.get(learned_norm, set())
                if len(entity_ids) == 1:
                    return self._result_from_entity(next(iter(entity_ids)), "document_abbreviation", flags)
                dynamic_id = stable_hash_id("ENT", learned_norm)
                return {
                    "entity_id": dynamic_id,
                    "canonical": learned_form,
                    "entity_type": "unclassified",
                    "concept_group": None,
                    "mapping_source": "document_abbreviation",
                    "alignment_flags": flags,
                }
            if len(learned_forms) > 1:
                flags.append("CONFLICTING_DOCUMENT_ABBREVIATION")

        surface_norm = normalize_text(surface)
        lemma_norm = normalize_text(lemma)
        candidates: set[str] = set()
        for normalized in (surface_norm, lemma_norm):
            candidates.update(self.alias_index.get(normalized, set()))

        if not candidates:
            for normalized in (surface_norm, lemma_norm):
                expanded = self._expand_abbreviations_in_phrase(normalized, sentence_text)
                candidates.update(self.alias_index.get(expanded, set()))

        if len(candidates) == 1:
            return self._result_from_entity(next(iter(candidates)), "entity_dictionary", flags)
        if len(candidates) > 1:
            flags.append("AMBIGUOUS_ENTITY_ALIAS")

        unknown_basis = lemma_norm or surface_norm or "empty"
        return {
            "entity_id": stable_hash_id("ENT", unknown_basis),
            "canonical": lemma.strip() or surface.strip(),
            "entity_type": "unclassified",
            "concept_group": None,
            "mapping_source": "generated_from_lemma",
            "alignment_flags": flags,
        }


class RelationNormalizer:
    def __init__(self, config: Mapping[str, Any]):
        self.by_lemma: dict[str, dict[str, Any]] = {}
        for item in config.get("relations", []):
            for lemma in item.get("lemmas", []):
                self.by_lemma[normalize_text(lemma)] = dict(item)

    def align(self, lemma: str, negated: bool) -> dict[str, Any]:
        lemma_norm = normalize_text(lemma)
        item = self.by_lemma.get(lemma_norm)
        if item:
            relation_id = item["id"]
            canonical = item["canonical"]
            source = "relation_dictionary"
        else:
            relation_id = stable_hash_id("REL", lemma_norm or "unknown")
            canonical = lemma_norm or lemma
            source = "predicate_lemma"
        if negated:
            relation_id = f"NOT_{relation_id}"
            canonical = f"не {canonical}"
        return {
            "relation_id": relation_id,
            "canonical": canonical,
            "mapping_source": source,
            "negated": negated,
        }


def expand_conjuncts(head_id: int, sentence: ParsedSentence) -> list[int]:
    result: list[int] = []
    queue = [head_id]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current in seen or current not in sentence.words:
            continue
        seen.add(current)
        result.append(current)
        for child in sentence.child_words(current):
            if child.dep_base == "conj" and child.upos in NOMINAL_POS:
                queue.append(child.id)
    return sorted(result)


def extract_entity_phrase(
    head_id: int,
    sentence: ParsedSentence,
    allowed_dependencies: set[str],
) -> dict[str, Any]:
    collected: set[int] = set()

    def visit(word_id: int) -> None:
        if word_id in collected or word_id not in sentence.words:
            return
        collected.add(word_id)
        for child in sentence.child_words(word_id):
            if child.dep_base in allowed_dependencies:
                visit(child.id)

    visit(head_id)
    token_ids = sorted(
        word_id
        for word_id in collected
        if sentence.words[word_id].upos != "PUNCT"
    )
    surface = " ".join(sentence.words[word_id].text for word_id in token_ids).strip()
    lemma = " ".join(sentence.words[word_id].lemma for word_id in token_ids).strip()
    return {
        "head_id": head_id,
        "token_ids": token_ids,
        "surface": surface,
        "lemma": lemma,
    }


def has_negation(predicate_id: int, sentence: ParsedSentence) -> bool:
    for child in sentence.child_words(predicate_id):
        if normalize_text(child.lemma) == "не" or child.deprel == "advmod:neg":
            return True
    return False


def direct_arguments(
    predicate_id: int,
    sentence: ParsedSentence,
    role: str,
    argument_scope: str,
) -> list[int]:
    children = sentence.child_words(predicate_id)
    if role == "subject":
        return [
            word.id
            for word in children
            if word.dep_base in SUBJECT_DEPS and not word.is_passive_subject and word.upos in NOMINAL_POS
        ]
    if role == "passive_subject":
        return [word.id for word in children if word.is_passive_subject and word.upos in NOMINAL_POS]
    if role == "object":
        core = [
            word.id
            for word in children
            if word.dep_base in CORE_OBJECT_DEPS and word.upos in NOMINAL_POS
        ]
        if core or argument_scope == "core":
            return core
        return [word.id for word in children if word.dep_base == "obl" and word.upos in NOMINAL_POS]
    if role == "agent":
        agents = [
            word.id
            for word in children
            if word.deprel.startswith("obl:agent") and word.upos in NOMINAL_POS
        ]
        if agents:
            return agents
        return [
            word.id
            for word in children
            if word.dep_base == "obl" and word.upos in NOMINAL_POS and "Case=Ins" in word.feature_set
        ]
    raise ValueError(f"unknown argument role: {role}")


def arguments_with_conj_inheritance(
    predicate_id: int,
    sentence: ParsedSentence,
    role: str,
    argument_scope: str,
) -> tuple[list[int], bool]:
    direct = direct_arguments(predicate_id, sentence, role, argument_scope)
    if direct:
        return direct, False
    predicate = sentence.words[predicate_id]
    if predicate.dep_base == "conj" and predicate.head in sentence.words:
        parent = sentence.words[predicate.head]
        if parent.upos == "VERB":
            inherited = direct_arguments(parent.id, sentence, role, argument_scope)
            if inherited:
                return inherited, True
    return [], False


def candidate_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["subject_head"],
        candidate["predicate_id"],
        candidate["object_head"],
        candidate["rule"],
    )


def extract_syntactic_candidates(
    sentence: ParsedSentence,
    argument_scope: str,
    include_zero_cop: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()

    for predicate in sentence.words.values():
        if predicate.upos != "VERB":
            continue
        diagnostics["verbal_predicates"] += 1
        passive_subjects, passive_inherited = arguments_with_conj_inheritance(
            predicate.id, sentence, "passive_subject", argument_scope
        )
        if passive_subjects:
            agents, agent_inherited = arguments_with_conj_inheritance(
                predicate.id, sentence, "agent", argument_scope
            )
            if not agents:
                diagnostics["passive_without_explicit_agent"] += 1
                continue
            for agent_head in agents:
                for expanded_agent in expand_conjuncts(agent_head, sentence):
                    for passive_head in passive_subjects:
                        for expanded_object in expand_conjuncts(passive_head, sentence):
                            candidates.append(
                                {
                                    "subject_head": expanded_agent,
                                    "predicate_id": predicate.id,
                                    "object_head": expanded_object,
                                    "rule": "PASSIVE_EXPLICIT_AGENT",
                                    "argument_kind": "core",
                                    "inherited_argument": passive_inherited or agent_inherited,
                                }
                            )
            continue

        subjects, subject_inherited = arguments_with_conj_inheritance(
            predicate.id, sentence, "subject", argument_scope
        )
        objects, object_inherited = arguments_with_conj_inheritance(
            predicate.id, sentence, "object", argument_scope
        )
        if not subjects or not objects:
            diagnostics["verbal_predicates_without_complete_spo"] += 1
            continue

        object_kind = "core"
        if not direct_arguments(predicate.id, sentence, "object", "core") and argument_scope == "extended":
            object_kind = "oblique"
        rule = "ACTIVE_CORE" if object_kind == "core" else "ACTIVE_OBLIQUE"
        if subject_inherited or object_inherited:
            rule += "_CONJ_INHERITED"

        for subject_head in subjects:
            for expanded_subject in expand_conjuncts(subject_head, sentence):
                for object_head in objects:
                    for expanded_object in expand_conjuncts(object_head, sentence):
                        candidates.append(
                            {
                                "subject_head": expanded_subject,
                                "predicate_id": predicate.id,
                                "object_head": expanded_object,
                                "rule": rule,
                                "argument_kind": object_kind,
                                "inherited_argument": subject_inherited or object_inherited,
                            }
                        )

    for predicate in sentence.words.values():
        if predicate.upos not in {"NOUN", "PROPN", "ADJ"}:
            continue
        copulas = [child for child in sentence.child_words(predicate.id) if child.dep_base == "cop"]
        subjects = [
            child.id
            for child in sentence.child_words(predicate.id)
            if child.dep_base in SUBJECT_DEPS and child.upos in NOMINAL_POS
        ]
        if not subjects:
            continue
        if not copulas and not include_zero_cop:
            continue
        relation_word = copulas[0] if copulas else None
        rule = "COPULAR_OVERT" if relation_word else "COPULAR_ZERO_INFERRED"
        for subject_head in subjects:
            for expanded_subject in expand_conjuncts(subject_head, sentence):
                candidates.append(
                    {
                        "subject_head": expanded_subject,
                        "predicate_id": relation_word.id if relation_word else predicate.id,
                        "object_head": predicate.id,
                        "rule": rule,
                        "argument_kind": "copular",
                        "inherited_argument": False,
                        "inferred_relation_lemma": None if relation_word else "являться",
                    }
                )

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        key = candidate_key(candidate)
        if key not in seen:
            seen.add(key)
            deduplicated.append(candidate)
        else:
            diagnostics["duplicate_syntactic_candidate"] += 1
    diagnostics["syntactic_candidates"] += len(deduplicated)
    return deduplicated, diagnostics


def build_triple(
    sentence: ParsedSentence,
    candidate: Mapping[str, Any],
    phrase_dependencies: set[str],
    entity_normalizer: EntityNormalizer,
    relation_normalizer: RelationNormalizer,
    formula_filter: FormulaFilter,
    sentence_flags: Sequence[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    subject_phrase = extract_entity_phrase(candidate["subject_head"], sentence, phrase_dependencies)
    object_phrase = extract_entity_phrase(candidate["object_head"], sentence, phrase_dependencies)
    predicate = sentence.words[candidate["predicate_id"]]
    relation_lemma = str(candidate.get("inferred_relation_lemma") or predicate.lemma)
    relation_surface = (
        "∅→являться"
        if candidate.get("inferred_relation_lemma")
        else predicate.text
    )

    subject_reject, subject_flags = formula_filter.entity_decision(
        subject_phrase["surface"], subject_phrase["lemma"]
    )
    object_reject, object_flags = formula_filter.entity_decision(
        object_phrase["surface"], object_phrase["lemma"]
    )
    rejection_reasons = [f"SUBJECT:{item}" for item in subject_reject] + [
        f"OBJECT:{item}" for item in object_reject
    ]
    if not subject_phrase["surface"] or not object_phrase["surface"]:
        rejection_reasons.append("EMPTY_ARGUMENT_PHRASE")

    evidence_key = "|".join(
        str(item)
        for item in (
            sentence.document_id,
            sentence.source_file,
            sentence.paragraph_id,
            sentence.sentence_id,
            candidate["subject_head"],
            candidate["predicate_id"],
            candidate["object_head"],
            candidate["rule"],
        )
    )
    triple_id = stable_hash_id("TRP", evidence_key, length=16)

    if rejection_reasons:
        return None, {
            "triple_id": triple_id,
            "document_id": sentence.document_id,
            "source_file": sentence.source_file,
            "paragraph_id": sentence.paragraph_id,
            "sentence_id": sentence.sentence_id,
            "sentence_text": sentence.text,
            "extraction_rule": candidate["rule"],
            "subject_surface": subject_phrase["surface"],
            "relation_surface": relation_surface,
            "object_surface": object_phrase["surface"],
            "rejection_reasons": rejection_reasons,
        }

    subject_alignment = entity_normalizer.align(
        subject_phrase["surface"],
        subject_phrase["lemma"],
        sentence.text,
        sentence.document_id,
    )
    object_alignment = entity_normalizer.align(
        object_phrase["surface"],
        object_phrase["lemma"],
        sentence.text,
        sentence.document_id,
    )
    negated = False if candidate.get("inferred_relation_lemma") else has_negation(predicate.id, sentence)
    relation_alignment = relation_normalizer.align(relation_lemma, negated)

    subject = {**subject_phrase, **subject_alignment, "quality_flags": subject_flags}
    obj = {**object_phrase, **object_alignment, "quality_flags": object_flags}
    relation = {
        "word_id": predicate.id,
        "surface": relation_surface,
        "lemma": relation_lemma,
        **relation_alignment,
    }
    combined_flags = sorted(
        set(sentence_flags)
        | set(subject_flags)
        | set(object_flags)
        | set(subject_alignment["alignment_flags"])
        | set(object_alignment["alignment_flags"])
    )
    if subject["entity_id"] == obj["entity_id"]:
        combined_flags.append("SELF_LOOP_AFTER_ALIGNMENT")

    triple = {
        "triple_id": triple_id,
        "document_id": sentence.document_id,
        "source_file": sentence.source_file,
        "year": infer_year(sentence.source_file, sentence.document_id),
        "paragraph_id": sentence.paragraph_id,
        "sentence_id": sentence.sentence_id,
        "sentence_start_char": sentence.start_char,
        "sentence_end_char": sentence.end_char,
        "sentence_text": sentence.text,
        "extraction_rule": candidate["rule"],
        "argument_kind": candidate["argument_kind"],
        "inherited_argument": bool(candidate.get("inherited_argument")),
        "subject": subject,
        "relation": relation,
        "object": obj,
        "quality_flags": combined_flags,
        "review_status": "pending",
    }
    return triple, None


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_triple(triple: Mapping[str, Any]) -> dict[str, Any]:
    subject = triple["subject"]
    relation = triple["relation"]
    obj = triple["object"]
    return {
        "triple_id": triple["triple_id"],
        "document_id": triple["document_id"],
        "source_file": triple["source_file"],
        "year": triple["year"],
        "paragraph_id": triple["paragraph_id"],
        "sentence_id": triple["sentence_id"],
        "sentence_text": triple["sentence_text"],
        "extraction_rule": triple["extraction_rule"],
        "argument_kind": triple["argument_kind"],
        "inherited_argument": triple["inherited_argument"],
        "subject_surface": subject["surface"],
        "subject_lemma": subject["lemma"],
        "subject_entity_id": subject["entity_id"],
        "subject_canonical": subject["canonical"],
        "subject_type": subject["entity_type"],
        "subject_concept_group": subject["concept_group"],
        "subject_mapping_source": subject["mapping_source"],
        "relation_surface": relation["surface"],
        "relation_lemma": relation["lemma"],
        "relation_id": relation["relation_id"],
        "relation_canonical": relation["canonical"],
        "relation_mapping_source": relation["mapping_source"],
        "negated": relation["negated"],
        "object_surface": obj["surface"],
        "object_lemma": obj["lemma"],
        "object_entity_id": obj["entity_id"],
        "object_canonical": obj["canonical"],
        "object_type": obj["entity_type"],
        "object_concept_group": obj["concept_group"],
        "object_mapping_source": obj["mapping_source"],
        "quality_flags": "|".join(triple["quality_flags"]),
        "review_status": triple["review_status"],
    }


TRIPLE_COLUMNS = (
    "triple_id",
    "document_id",
    "source_file",
    "year",
    "paragraph_id",
    "sentence_id",
    "sentence_text",
    "extraction_rule",
    "argument_kind",
    "inherited_argument",
    "subject_surface",
    "subject_lemma",
    "subject_entity_id",
    "subject_canonical",
    "subject_type",
    "subject_concept_group",
    "subject_mapping_source",
    "relation_surface",
    "relation_lemma",
    "relation_id",
    "relation_canonical",
    "relation_mapping_source",
    "negated",
    "object_surface",
    "object_lemma",
    "object_entity_id",
    "object_canonical",
    "object_type",
    "object_concept_group",
    "object_mapping_source",
    "quality_flags",
    "review_status",
)


def aggregate_nodes(triples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for triple in triples:
        for role in ("subject", "object"):
            entity = triple[role]
            node = nodes.setdefault(
                entity["entity_id"],
                {
                    "entity_id": entity["entity_id"],
                    "canonical": entity["canonical"],
                    "entity_type": entity["entity_type"],
                    "concept_group": entity["concept_group"],
                    "occurrence_count": 0,
                    "surface_forms": set(),
                    "lemma_forms": set(),
                    "mapping_sources": set(),
                },
            )
            node["occurrence_count"] += 1
            node["surface_forms"].add(entity["surface"])
            node["lemma_forms"].add(entity["lemma"])
            node["mapping_sources"].add(entity["mapping_source"])

    rows: list[dict[str, Any]] = []
    for node in nodes.values():
        rows.append(
            {
                **node,
                "surface_forms": " | ".join(sorted(node["surface_forms"])),
                "lemma_forms": " | ".join(sorted(node["lemma_forms"])),
                "mapping_sources": " | ".join(sorted(node["mapping_sources"])),
            }
        )
    return sorted(rows, key=lambda row: (-row["occurrence_count"], row["entity_id"]))


def aggregate_edges(triples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for triple in triples:
        subject = triple["subject"]
        relation = triple["relation"]
        obj = triple["object"]
        key = (subject["entity_id"], relation["relation_id"], obj["entity_id"])
        edge = edges.setdefault(
            key,
            {
                "edge_id": stable_hash_id("EDGE", "|".join(key), length=16),
                "subject_entity_id": subject["entity_id"],
                "subject_canonical": subject["canonical"],
                "relation_id": relation["relation_id"],
                "relation_canonical": relation["canonical"],
                "object_entity_id": obj["entity_id"],
                "object_canonical": obj["canonical"],
                "weight": 0,
                "document_ids": set(),
                "source_files": set(),
                "years": set(),
                "triple_ids": [],
            },
        )
        edge["weight"] += 1
        edge["document_ids"].add(triple["document_id"])
        edge["source_files"].add(triple["source_file"])
        if triple["year"] is not None:
            edge["years"].add(str(triple["year"]))
        edge["triple_ids"].append(triple["triple_id"])

    rows: list[dict[str, Any]] = []
    for edge in edges.values():
        rows.append(
            {
                **edge,
                "document_count": len(edge["document_ids"]),
                "document_ids": " | ".join(sorted(edge["document_ids"])),
                "source_files": " | ".join(sorted(edge["source_files"])),
                "years": " | ".join(sorted(edge["years"])),
                "triple_ids": " | ".join(edge["triple_ids"]),
            }
        )
    return sorted(rows, key=lambda row: (-row["weight"], row["edge_id"]))


def stratified_validation_sample(
    triples: Sequence[Mapping[str, Any]],
    sample_size: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    if sample_size <= 0 or not triples:
        return []
    if sample_size >= len(triples):
        return list(triples)
    rng = random.Random(seed)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for triple in triples:
        groups[str(triple["extraction_rule"])].append(triple)
    for items in groups.values():
        rng.shuffle(items)

    selected: list[Mapping[str, Any]] = []
    ordered_groups = sorted(groups)
    while len(selected) < sample_size and ordered_groups:
        remaining: list[str] = []
        for group_name in ordered_groups:
            if groups[group_name] and len(selected) < sample_size:
                selected.append(groups[group_name].pop())
            if groups[group_name]:
                remaining.append(group_name)
        ordered_groups = remaining
    rng.shuffle(selected)
    return selected


def validation_rows(sample: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for triple in sample:
        flat = flatten_triple(triple)
        rows.append(
            {
                "triple_id": flat["triple_id"],
                "source_file": flat["source_file"],
                "paragraph_id": flat["paragraph_id"],
                "sentence_id": flat["sentence_id"],
                "sentence_text": flat["sentence_text"],
                "extraction_rule": flat["extraction_rule"],
                "subject_surface": flat["subject_surface"],
                "subject_canonical": flat["subject_canonical"],
                "relation_surface": flat["relation_surface"],
                "relation_canonical": flat["relation_canonical"],
                "object_surface": flat["object_surface"],
                "object_canonical": flat["object_canonical"],
                "subject_correct": "",
                "relation_correct": "",
                "object_correct": "",
                "triple_correct": "",
                "include_in_graph": "",
                "reviewer_note": "",
            }
        )
    return rows


VALIDATION_COLUMNS = (
    "triple_id",
    "source_file",
    "paragraph_id",
    "sentence_id",
    "sentence_text",
    "extraction_rule",
    "subject_surface",
    "subject_canonical",
    "relation_surface",
    "relation_canonical",
    "object_surface",
    "object_canonical",
    "subject_correct",
    "relation_correct",
    "object_correct",
    "triple_correct",
    "include_in_graph",
    "reviewer_note",
)


def run_extraction(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"Output files already exist ({names}). Add --overwrite to replace them.")

    config = load_config(args.config)
    input_files = discover_inputs(args.input_path, args.recursive, output_dir)
    records = list(read_jsonl(input_files))
    entity_normalizer = EntityNormalizer(config)
    relation_normalizer = RelationNormalizer(config)
    formula_filter = FormulaFilter(config)
    for record in records:
        entity_normalizer.learn_from_record(record)

    phrase_dependencies = set(config.get("entity_phrase_dependencies", []))
    triples: list[dict[str, Any]] = []
    filtered_sentences: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()
    extraction_rule_counts: Counter[str] = Counter()
    mapping_source_counts: Counter[str] = Counter()

    for record in records:
        sentence = ParsedSentence(record)
        diagnostics["sentences_total"] += 1
        excluded_rules, sentence_flags = formula_filter.sentence_decision(sentence.text)
        if excluded_rules:
            diagnostics["sentences_excluded_as_formula"] += 1
            filtered_sentences.append(
                {
                    "document_id": sentence.document_id,
                    "source_file": sentence.source_file,
                    "paragraph_id": sentence.paragraph_id,
                    "sentence_id": sentence.sentence_id,
                    "sentence_text": sentence.text,
                    "matched_rules": excluded_rules,
                    "action": "excluded",
                }
            )
            continue

        syntactic_candidates, sentence_diagnostics = extract_syntactic_candidates(
            sentence,
            argument_scope=args.argument_scope,
            include_zero_cop=args.include_zero_cop,
        )
        diagnostics.update(sentence_diagnostics)
        for candidate in syntactic_candidates:
            triple, rejected = build_triple(
                sentence,
                candidate,
                phrase_dependencies,
                entity_normalizer,
                relation_normalizer,
                formula_filter,
                sentence_flags,
            )
            if rejected:
                diagnostics["candidates_rejected"] += 1
                rejected_candidates.append(rejected)
                continue
            assert triple is not None
            triples.append(triple)
            extraction_rule_counts[triple["extraction_rule"]] += 1
            mapping_source_counts[f"subject:{triple['subject']['mapping_source']}"] += 1
            mapping_source_counts[f"object:{triple['object']['mapping_source']}"] += 1
            mapping_source_counts[f"relation:{triple['relation']['mapping_source']}"] += 1

    diagnostics["triples_accepted"] = len(triples)
    nodes = aggregate_nodes(triples)
    edges = aggregate_edges(triples)
    sample = stratified_validation_sample(triples, args.validation_sample_size, args.seed)

    write_jsonl(output_dir / "triples.jsonl", triples)
    write_tsv(output_dir / "triples.tsv", (flatten_triple(item) for item in triples), TRIPLE_COLUMNS)
    write_tsv(
        output_dir / "nodes.tsv",
        nodes,
        (
            "entity_id",
            "canonical",
            "entity_type",
            "concept_group",
            "occurrence_count",
            "surface_forms",
            "lemma_forms",
            "mapping_sources",
        ),
    )
    write_tsv(
        output_dir / "edges_aggregated.tsv",
        edges,
        (
            "edge_id",
            "subject_entity_id",
            "subject_canonical",
            "relation_id",
            "relation_canonical",
            "object_entity_id",
            "object_canonical",
            "weight",
            "document_count",
            "document_ids",
            "source_files",
            "years",
            "triple_ids",
        ),
    )
    write_jsonl(output_dir / "filtered_sentences.jsonl", filtered_sentences)
    write_jsonl(output_dir / "rejected_candidates.jsonl", rejected_candidates)
    write_tsv(
        output_dir / "learned_abbreviations.tsv",
        entity_normalizer.learned_evidence,
        ("document_id", "source_file", "sentence_id", "abbreviation", "long_form", "sentence_text"),
    )
    write_tsv(
        output_dir / "manual_validation_sample.tsv",
        validation_rows(sample),
        VALIDATION_COLUMNS,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "input_files": [str(path) for path in input_files],
        "config_path": config["_resolved_path"],
        "config_sha256": config["_sha256"],
        "settings": {
            "argument_scope": args.argument_scope,
            "include_zero_cop": args.include_zero_cop,
            "validation_sample_size_requested": args.validation_sample_size,
            "validation_sample_size_written": len(sample),
            "random_seed": args.seed,
        },
        "counts": dict(sorted(diagnostics.items())),
        "extraction_rule_counts": dict(sorted(extraction_rule_counts.items())),
        "mapping_source_counts": dict(sorted(mapping_source_counts.items())),
        "graph": {"nodes": len(nodes), "unique_edges": len(edges), "edge_occurrences": len(triples)},
        "notes": [
            "Heuristic extraction rule names are audit labels, not probability scores.",
            "The validation sample estimates precision/error among extracted triples after manual coding.",
            "Recall requires a separately annotated sentence-level gold sample and is not inferred here."
        ],
    }
    (output_dir / "extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Done: sentences={diagnostics['sentences_total']}, triples={len(triples)}, "
        f"nodes={len(nodes)}, unique_edges={len(edges)}, filtered={len(filtered_sentences)}"
    )
    print(f"Output: {output_dir}")
    return 0


TRUE_VALUES = {"1", "true", "yes", "y", "是", "正确", "对"}
FALSE_VALUES = {"0", "false", "no", "n", "否", "错误", "错"}


def parse_boolean(value: str) -> bool | None:
    normalized = unicodedata.normalize("NFKC", value or "").casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return None
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + (z * z / total)
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total))) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def metric(values: Sequence[bool]) -> dict[str, Any]:
    total = len(values)
    correct = sum(values)
    accuracy = correct / total if total else None
    return {
        "reviewed": total,
        "correct": correct,
        "accuracy": accuracy,
        "error_rate": (1 - accuracy) if accuracy is not None else None,
        "wilson_95_ci": wilson_interval(correct, total),
    }


def run_validation_evaluation(args: argparse.Namespace) -> int:
    path = args.validation_file.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Validation file not found: {path}")

    component_values: dict[str, list[bool]] = {
        "subject": [],
        "relation": [],
        "object": [],
        "triple": [],
        "include_in_graph": [],
    }
    total_rows = 0
    reviewed_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"subject_correct", "relation_correct", "object_correct", "triple_correct"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Validation file is missing columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            try:
                subject = parse_boolean(row.get("subject_correct", ""))
                relation = parse_boolean(row.get("relation_correct", ""))
                obj = parse_boolean(row.get("object_correct", ""))
                triple = parse_boolean(row.get("triple_correct", ""))
                include = parse_boolean(row.get("include_in_graph", ""))
            except ValueError as exc:
                raise SystemExit(f"{path}:{row_number}: {exc}") from exc

            if all(value is None for value in (subject, relation, obj, triple, include)):
                continue
            reviewed_rows += 1
            if subject is not None:
                component_values["subject"].append(subject)
            if relation is not None:
                component_values["relation"].append(relation)
            if obj is not None:
                component_values["object"].append(obj)
            if triple is None and None not in (subject, relation, obj):
                triple = bool(subject and relation and obj)
            if triple is not None:
                component_values["triple"].append(triple)
            if include is not None:
                component_values["include_in_graph"].append(include)

    report = {
        "created_at_utc": utc_now(),
        "validation_file": str(path),
        "total_sample_rows": total_rows,
        "reviewed_rows": reviewed_rows,
        "metrics": {name: metric(values) for name, values in component_values.items()},
        "interpretation": {
            "triple_accuracy": "sampled exact-triple precision among extracted candidates",
            "triple_error_rate": "1 minus sampled exact-triple precision",
            "recall": "not estimated; requires a sentence-level gold standard including missed triples"
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Report: {output}")
    return 0


def kg_legacy_main() -> int:
    args = parse_kg_legacy_args()
    if args.command == "extract":
        return run_extraction(args)
    if args.command == "evaluate-validation":
        return run_validation_evaluation(args)
    raise SystemExit(f"Unknown command: {args.command}")


def build_unified_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Russian Stanza preprocessing and auditable dependency-based "
            "knowledge-graph extraction from one command-line program."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_preprocess_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "input_path", type=Path,
            help="A UTF-8 .txt file or a directory containing .txt files."
        )
        command_parser.add_argument("--recursive", action="store_true")
        command_parser.add_argument("--download-model", action="store_true")
        command_parser.add_argument("--model-dir", type=Path, default=None)
        command_parser.add_argument(
            "--device", choices=("auto", "cpu", "cuda"), default="auto"
        )
        command_parser.add_argument("--encoding", default="utf-8")
        command_parser.add_argument("--overwrite", action="store_true")

    preprocess = subparsers.add_parser(
        "preprocess", help="Run Stanza linguistic preprocessing only."
    )
    add_preprocess_options(preprocess)
    preprocess.add_argument(
        "--output-dir", "-o", type=Path, default=Path("stanza_output")
    )

    pipeline = subparsers.add_parser(
        "pipeline", help="Run preprocessing and SPO/graph extraction in sequence."
    )
    add_preprocess_options(pipeline)
    pipeline.add_argument(
        "--output-dir", "-o", type=Path, default=Path("pipeline_output"),
        help="Root output directory; creates stanza/ and kg/ subdirectories."
    )
    pipeline.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pipeline.add_argument(
        "--argument-scope", choices=("core", "extended"), default="core"
    )
    pipeline.add_argument("--include-zero-cop", action="store_true")
    pipeline.add_argument("--validation-sample-size", type=int, default=200)
    pipeline.add_argument("--seed", type=int, default=202503)

    extract = subparsers.add_parser(
        "extract", help="Extract normalized SPO triples from Stanza JSONL."
    )
    extract.add_argument(
        "input_path", type=Path,
        help="A .sentences.jsonl file or directory containing such files."
    )
    extract.add_argument("--output-dir", "-o", type=Path, required=True)
    extract.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    extract.add_argument("--recursive", action="store_true")
    extract.add_argument(
        "--argument-scope", choices=("core", "extended"), default="core"
    )
    extract.add_argument("--include-zero-cop", action="store_true")
    extract.add_argument("--validation-sample-size", type=int, default=200)
    extract.add_argument("--seed", type=int, default=202503)
    extract.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate-validation",
        help="Calculate sampled precision/error metrics from a completed TSV."
    )
    evaluate.add_argument("validation_file", type=Path)
    evaluate.add_argument("--output", "-o", type=Path, required=True)
    return parser


def run_preprocessing_stage(args: argparse.Namespace, output_dir: Path) -> int:
    stanza = import_stanza()
    use_gpu = resolve_use_gpu(args.device)
    if args.download_model:
        download_models(stanza, args.model_dir)

    nlp = build_pipeline(stanza, args.model_dir, use_gpu)
    input_files, input_root = discover_input_files(args.input_path, args.recursive)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "started_at_utc": preprocess_utc_now(),
        "stanza_version": getattr(stanza, "__version__", "unknown"),
        "language": LANGUAGE,
        "model_package": PACKAGE,
        "processors": PROCESSORS.split(","),
        "universal_dependencies": True,
        "device": "cuda" if use_gpu else "cpu",
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "documents": [],
    }
    error_lines: list[str] = []

    total = len(input_files)
    for index, input_file in enumerate(input_files, start=1):
        relative = input_file.relative_to(input_root).as_posix()
        print(f"[preprocess {index}/{total}] {relative}", flush=True)
        try:
            result = process_file(
                nlp=nlp,
                input_file=input_file,
                input_root=input_root,
                output_dir=output_dir,
                encoding=args.encoding,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            result = {
                "source_file": relative,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            error_lines.append(
                f"[{preprocess_utc_now()}] {relative}\n{traceback.format_exc()}\n"
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        summary["documents"].append(result)

    summary["finished_at_utc"] = preprocess_utc_now()
    summary["counts"] = {
        "total": total,
        "processed": sum(d["status"] == "processed" for d in summary["documents"]),
        "skipped": sum(d["status"] == "skipped" for d in summary["documents"]),
        "errors": sum(d["status"] == "error" for d in summary["documents"]),
    }
    (output_dir / "processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "errors.log").write_text(
        "\n".join(error_lines) if error_lines else "No processing errors.\n",
        encoding="utf-8",
    )
    counts = summary["counts"]
    print(
        "Preprocessing done: "
        f"processed={counts['processed']}, skipped={counts['skipped']}, "
        f"errors={counts['errors']}"
    )
    return 1 if counts["errors"] else 0


def run_full_pipeline(args: argparse.Namespace) -> int:
    root = args.output_dir.expanduser().resolve()
    stanza_dir = root / "stanza"
    kg_dir = root / "kg"
    root.mkdir(parents=True, exist_ok=True)

    preprocess_status = run_preprocessing_stage(args, stanza_dir)
    if preprocess_status != 0:
        print(
            "Extraction was not started because one or more preprocessing "
            "documents failed. See stanza/errors.log.",
            file=sys.stderr,
        )
        return preprocess_status

    extraction_args = argparse.Namespace(
        input_path=stanza_dir,
        output_dir=kg_dir,
        config=args.config,
        recursive=True,
        argument_scope=args.argument_scope,
        include_zero_cop=args.include_zero_cop,
        validation_sample_size=args.validation_sample_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print("Starting entity normalization and SPO extraction...", flush=True)
    return run_extraction(extraction_args)


def main() -> int:
    args = build_unified_parser().parse_args()
    if args.command == "preprocess":
        return run_preprocessing_stage(args, args.output_dir)
    if args.command == "pipeline":
        return run_full_pipeline(args)
    if args.command == "extract":
        return run_extraction(args)
    if args.command == "evaluate-validation":
        return run_validation_evaluation(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
