#!/usr/bin/env python3
"""
Inspect a PokemonKnowledgeBase: lists every species it has recorded,
how many battles it's been seen in, its known typing, and its
most-common move/ability/item so far.

Without this, the only way to see what's in a knowledge base was to
read the JSON file by hand.

Usage:
    python scripts/inspect_knowledge.py --knowledge-path runs/pokemon_knowledge.json
    python scripts/inspect_knowledge.py --knowledge-path runs/pokemon_knowledge.json --sort battles_seen
    python scripts/inspect_knowledge.py --knowledge-path runs/pokemon_knowledge.json --species garchomp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge.pokemon_knowledge import PokemonKnowledgeBase

_SORT_KEYS = {
    "battles_seen": lambda s: -s["battles_seen"],
    "name": lambda s: s["species"],
}


def _print_species_detail(summary: dict) -> None:
    print(f"{summary['species']}")
    print(f"  types: {'/'.join(summary['types']) or 'unknown'}")
    print(f"  battles_seen: {summary['battles_seen']}")
    print(f"  most_common_ability: {summary['most_common_ability'] or '-'}")
    print(f"  most_common_item: {summary['most_common_item'] or '-'}")
    top_moves = summary["known_moves_by_frequency"][:5]
    print(f"  top_moves: {', '.join(top_moves) if top_moves else '-'}")
    if summary["weaknesses"]:
        weak = ", ".join(f"{t}x{m:g}" for t, m in sorted(summary["weaknesses"].items(), key=lambda kv: -kv[1]))
        print(f"  weaknesses: {weak}")
    if summary["immunities"]:
        print(f"  immunities: {', '.join(summary['immunities'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-path", type=str, required=True)
    parser.add_argument(
        "--sort",
        type=str,
        default="battles_seen",
        choices=list(_SORT_KEYS),
        help="battles_seen: most-observed species first (default). name: alphabetical.",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Show full detail for a single species instead of listing every species.",
    )
    args = parser.parse_args()

    kb = PokemonKnowledgeBase(path=args.knowledge_path)

    if len(kb) == 0:
        print(f"Knowledge base at {args.knowledge_path} is empty.")
        return

    if args.species:
        summary = kb.species_summary(args.species)
        if summary is None:
            print(f"No knowledge recorded for {args.species!r}.")
            return
        _print_species_detail(summary)
        return

    summaries = [kb.species_summary(species) for species in kb.known_species()]
    summaries.sort(key=_SORT_KEYS[args.sort])

    name_width = max(len(s["species"]) for s in summaries)
    header = f"{'species':<{name_width}}  {'battles':>7}  {'types':<15}  top_move"
    print(header)
    print("-" * len(header))
    for s in summaries:
        types = "/".join(s["types"]) or "-"
        top_move = s["known_moves_by_frequency"][0] if s["known_moves_by_frequency"] else "-"
        print(f"{s['species']:<{name_width}}  {s['battles_seen']:>7}  {types:<15}  {top_move}")

    print()
    print(f"{len(summaries)} species known at {args.knowledge_path}")


if __name__ == "__main__":
    main()