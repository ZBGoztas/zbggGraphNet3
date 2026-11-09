"""
export_ai_vocabulary.py — 3D Mesh yamalarindan AI Vocabulary olusturma.

Bu script iki farkli patch JSON formatini destekler:
1) Ham format: patch icinde `wl_hash`, `local_nodes`, `local_elements`
2) main.py cikti formati: patch icinde `topology_hash`,
   `local_nodes` (dict) ve `local_elements` (dict -> {type, nodes})

Cikti:
- 3 adet special token (0,1,2)
- Frekansa gore siralanmis patch tokenlari (3'ten baslar)
- ai_vocabulary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TypeAlias, Union

LocalNodes: TypeAlias = List[List[float]]
LocalElements: TypeAlias = List[List[int]]
WLHash: TypeAlias = str


@dataclass
class PatchTemplate:
    local_nodes: LocalNodes
    local_elements: LocalElements


@dataclass
class TopologyInfo:
    wl_hash: WLHash
    frequency: int = 0
    template: Optional[PatchTemplate] = None


@dataclass
class VocabularyToken:
    token_id: int
    type: str
    name: Optional[str] = None
    description: Optional[str] = None
    wl_hash: Optional[WLHash] = None
    rank: Optional[int] = None
    frequency: Optional[int] = None
    local_nodes: Optional[LocalNodes] = None
    local_elements: Optional[LocalElements] = None


def _configure_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = _configure_logger(__name__)


class AIVocabularyExporter:
    SPECIAL_TOKENS: Dict[int, Dict[str, str]] = {
        0: {"type": "SPECIAL", "name": "<PAD>", "description": "Padding"},
        1: {"type": "SPECIAL", "name": "<START>", "description": "Start of Sequence"},
        2: {"type": "SPECIAL", "name": "<END>", "description": "End of Sequence"},
    }

    SPECIAL_TOKENS_COUNT = len(SPECIAL_TOKENS)
    FIRST_PATCH_TOKEN_ID = SPECIAL_TOKENS_COUNT

    def __init__(
        self,
        input_json: Union[str, Path] = "extracted_patches.json",
        output_json: Union[str, Path] = "ai_vocabulary.json",
    ) -> None:
        self.input_json = Path(input_json)
        self.output_json = Path(output_json)
        self.topologies: Dict[WLHash, TopologyInfo] = {}
        self.vocabulary: Dict[int, VocabularyToken] = {}

        logger.info("AIVocabularyExporter baslatildi")
        logger.info(f"  -> Giris: {self.input_json}")
        logger.info(f"  -> Cikis: {self.output_json}")

    @staticmethod
    def _extract_wl_hash(patch: Dict[str, Any]) -> Optional[str]:
        wl_hash = patch.get("wl_hash")
        if isinstance(wl_hash, str) and wl_hash:
            return wl_hash

        topology_hash = patch.get("topology_hash")
        if isinstance(topology_hash, str) and topology_hash:
            return topology_hash

        return None

    @staticmethod
    def _coerce_local_nodes(raw_nodes: Any) -> Optional[LocalNodes]:
        if isinstance(raw_nodes, list):
            nodes: LocalNodes = []
            for row in raw_nodes:
                if not isinstance(row, list):
                    return None
                nodes.append([float(v) for v in row])
            return nodes

        if isinstance(raw_nodes, dict):
            ordered_nodes: LocalNodes = []
            for _, coords in sorted(raw_nodes.items(), key=lambda kv: int(kv[0])):
                if not isinstance(coords, list):
                    return None
                ordered_nodes.append([float(v) for v in coords])
            return ordered_nodes

        return None

    @staticmethod
    def _coerce_local_elements(raw_elements: Any) -> Optional[LocalElements]:
        if isinstance(raw_elements, list):
            elements: LocalElements = []
            for conn in raw_elements:
                if not isinstance(conn, list):
                    return None
                elements.append([int(v) for v in conn])
            return elements

        if isinstance(raw_elements, dict):
            # main.py formatinda her eleman: {"type": "tria|quad", "nodes": [...]}
            ordered_elements: LocalElements = []
            for _, edata in sorted(raw_elements.items(), key=lambda kv: int(kv[0])):
                if isinstance(edata, dict) and isinstance(edata.get("nodes"), list):
                    ordered_elements.append([int(v) for v in edata["nodes"]])
                    continue
                return None
            return ordered_elements

        return None

    @staticmethod
    def _iter_patches_from_loaded_json(data: Any) -> Iterator[Dict[str, Any]]:
        # Format A: root list
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return

        # Format B/C: root dict with "patches"
        if isinstance(data, dict) and "patches" in data:
            patches = data["patches"]
            if isinstance(patches, list):
                for item in patches:
                    if isinstance(item, dict):
                        yield item
                return

            if isinstance(patches, dict):
                for item in patches.values():
                    if isinstance(item, dict):
                        yield item
                return

        raise ValueError(
            "Desteklenmeyen JSON yapisi. Beklenenler: root list, "
            "veya root['patches'] altinda list/dict."
        )

    def _process_patch(self, patch: Dict[str, Any], patch_index: int, source_name: str) -> None:
        wl_hash = self._extract_wl_hash(patch)
        if wl_hash is None:
            logger.warning(f"{source_name} patch#{patch_index}: hash yok, atlaniyor")
            return

        if wl_hash not in self.topologies:
            template = None
            local_nodes = self._coerce_local_nodes(patch.get("local_nodes"))
            local_elements = self._coerce_local_elements(patch.get("local_elements"))

            if local_nodes is not None and local_elements is not None:
                template = PatchTemplate(local_nodes=local_nodes, local_elements=local_elements)

            self.topologies[wl_hash] = TopologyInfo(wl_hash=wl_hash, template=template)

        self.topologies[wl_hash].frequency += 1

    def _read_one_json_file(self, file_path: Path, start_count: int = 0) -> int:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        patch_count = start_count
        for patch in self._iter_patches_from_loaded_json(data):
            patch_count += 1
            self._process_patch(patch, patch_count, file_path.name)

            if patch_count % 100000 == 0:
                logger.info(
                    f"  {patch_count:,} patch islendi... ({len(self.topologies)} benzersiz topoloji)"
                )

        return patch_count

    def _stream_read_patches(self) -> None:
        if not self.input_json.exists():
            raise FileNotFoundError(f"Dosya/dizin bulunamadi: {self.input_json}")

        logger.info(f"Patch verisi okunuyor: {self.input_json}")
        patch_count = 0

        if self.input_json.is_file():
            patch_count = self._read_one_json_file(self.input_json)
        else:
            files = sorted(self.input_json.glob("*_patches.json"))
            if not files:
                raise FileNotFoundError(
                    f"{self.input_json} altinda *_patches.json bulunamadi"
                )

            logger.info(f"  {len(files)} adet *_patches.json bulundu")
            for json_file in files:
                patch_count = self._read_one_json_file(json_file, start_count=patch_count)

        logger.info(
            f"Toplam {patch_count:,} patch icinde {len(self.topologies)} benzersiz topoloji bulundu"
        )

    def _sort_by_frequency(self) -> List[TopologyInfo]:
        sorted_topos = sorted(self.topologies.values(), key=lambda t: t.frequency, reverse=True)
        logger.info(f"Topolojiler frekansa gore siralandi (Rank 1 -> {len(sorted_topos)})")
        return sorted_topos

    def _build_vocabulary(self, sorted_topos: List[TopologyInfo]) -> None:
        for token_id, token_info in self.SPECIAL_TOKENS.items():
            self.vocabulary[token_id] = VocabularyToken(
                token_id=token_id,
                type=token_info["type"],
                name=token_info["name"],
                description=token_info["description"],
            )

        for rank, topology in enumerate(sorted_topos, start=1):
            token_id = self.FIRST_PATCH_TOKEN_ID + rank - 1
            self.vocabulary[token_id] = VocabularyToken(
                token_id=token_id,
                type="PATCH",
                wl_hash=topology.wl_hash,
                rank=rank,
                frequency=topology.frequency,
                local_nodes=topology.template.local_nodes if topology.template else None,
                local_elements=topology.template.local_elements if topology.template else None,
            )

        logger.info(
            f"Vocabulary olusturuldu: {len(self.SPECIAL_TOKENS)} ozel + "
            f"{len(sorted_topos)} patch = {len(self.vocabulary)} token"
        )

    def build(self) -> Dict[int, VocabularyToken]:
        self._stream_read_patches()
        sorted_topos = self._sort_by_frequency()
        self._build_vocabulary(sorted_topos)
        return self.vocabulary

    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        if not self.vocabulary:
            raise ValueError("Vocabulary bos. Once build() cagrilmali.")

        output_path = Path(output_path or self.output_json)

        # Kullanici klasor yolu verdiyse varsayilan dosya adini ekle.
        # Ornek: --output C:/data/vocabulary_data  -> C:/data/vocabulary_data/ai_vocabulary.json
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / "ai_vocabulary.json"
        elif output_path.suffix.lower() != ".json":
            output_path = output_path / "ai_vocabulary.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_dict: Dict[str, Any] = {"vocab_size": len(self.vocabulary), "tokens": {}}

        for token_id, token in self.vocabulary.items():
            token_data: Dict[str, Any] = {"type": token.type}
            if token.type == "SPECIAL":
                token_data["name"] = token.name
                token_data["description"] = token.description
            else:
                token_data["wl_hash"] = token.wl_hash
                token_data["rank"] = token.rank
                token_data["frequency"] = token.frequency
                token_data["local_nodes"] = token.local_nodes
                token_data["local_elements"] = token.local_elements

            vocab_dict["tokens"][str(token_id)] = token_data

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Vocabulary kaydedildi: {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        if not self.vocabulary:
            return {}

        patch_tokens = [t for t in self.vocabulary.values() if t.type == "PATCH"]
        frequencies = [int(t.frequency or 0) for t in patch_tokens]

        return {
            "total_vocab_size": len(self.vocabulary),
            "special_tokens": len(self.SPECIAL_TOKENS),
            "patch_tokens": len(patch_tokens),
            "unique_topologies": len(self.topologies),
            "min_frequency": min(frequencies) if frequencies else 0,
            "max_frequency": max(frequencies) if frequencies else 0,
            "avg_frequency": (sum(frequencies) / len(frequencies)) if frequencies else 0.0,
            "total_patches": sum(frequencies) if frequencies else 0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="main.py ciktilarindan AI vocabulary olustur")
    parser.add_argument(
        "--input",
        default="extracted_patches.json",
        help="Giris JSON dosyasi veya *_patches.json dosyalarini iceren dizin",
    )
    parser.add_argument(
        "--output",
        default="ai_vocabulary.json",
        help="Cikis vocabulary JSON yolu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        exporter = AIVocabularyExporter(input_json=args.input, output_json=args.output)
        exporter.build()
        exporter.save()

        stats = exporter.get_statistics()
        logger.info("Istatistikler:")
        logger.info(f"  Toplam Vocabulary Boyutu: {stats['total_vocab_size']}")
        logger.info(f"  Ozel Tokenler: {stats['special_tokens']}")
        logger.info(f"  Patch Tokenler: {stats['patch_tokens']}")
        logger.info(f"  Benzersiz Topolojiler: {stats['unique_topologies']}")
        logger.info(f"  Min Frekans: {stats['min_frequency']:,}")
        logger.info(f"  Max Frekans: {stats['max_frequency']:,}")
        logger.info(f"  Ort. Frekans: {stats['avg_frequency']:,.0f}")
        logger.info(f"  Toplam Patch Sayisi: {stats['total_patches']:,}")

    except Exception as e:
        logger.error(f"Hata: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
