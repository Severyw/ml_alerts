"""Land-border adjacency between the oblast-level units we model.

Provenance: first drafted by the AI assistant from general geographic
knowledge, then checked pair by pair against a map by the author, who
confirmed the list as it stands (including the five pairs that had been
tagged uncertain). `python -m src.adjacency` prints it as a plain list.

Scope notes:
* Only borders between the 24 units in config.MODELLED_OBLASTS are listed.
  Luhanska oblast (excluded from the target) and Crimea/Sevastopol (absent
  from both CSVs) are omitted, so e.g. Kharkiv-Luhansk and Donetsk-Luhansk
  are NOT in this list even though those borders exist.
* "Kyiv City" is its own unit in the data. It borders only Kyivska oblast.
* Borders with other countries are irrelevant here and not listed.
"""
from __future__ import annotations

from src.config import MODELLED_OBLASTS

# (a, b, note). note is free text, currently empty for every pair; the author
# checked all pairs on a map. Order within a pair is irrelevant.
_RAW_EDGES: list[tuple[str, str, str]] = [
    # --- north / centre ---
    ("Kyiv City", "Kyivska oblast", ""),
    ("Kyivska oblast", "Chernihivska oblast", ""),
    ("Kyivska oblast", "Poltavska oblast", ""),
    ("Kyivska oblast", "Cherkaska oblast", ""),
    ("Kyivska oblast", "Zhytomyrska oblast", ""),
    ("Kyivska oblast", "Vinnytska oblast", ""),
    ("Chernihivska oblast", "Poltavska oblast", ""),
    ("Chernihivska oblast", "Sumska oblast", ""),
    ("Sumska oblast", "Poltavska oblast", ""),
    ("Sumska oblast", "Kharkivska oblast", ""),
    ("Poltavska oblast", "Kharkivska oblast", ""),
    ("Poltavska oblast", "Cherkaska oblast", ""),
    ("Poltavska oblast", "Kirovohradska oblast", ""),
    ("Poltavska oblast", "Dnipropetrovska oblast", ""),
    ("Cherkaska oblast", "Kirovohradska oblast", ""),
    ("Cherkaska oblast", "Vinnytska oblast", ""),
    # --- east / south-east ---
    ("Kharkivska oblast", "Dnipropetrovska oblast", ""),
    ("Kharkivska oblast", "Donetska oblast", ""),
    ("Donetska oblast", "Dnipropetrovska oblast", ""),
    ("Donetska oblast", "Zaporizka oblast", ""),
    ("Zaporizka oblast", "Dnipropetrovska oblast", ""),
    ("Zaporizka oblast", "Khersonska oblast", ""),
    ("Dnipropetrovska oblast", "Kirovohradska oblast", ""),
    ("Dnipropetrovska oblast", "Khersonska oblast", ""),
    ("Dnipropetrovska oblast", "Mykolaivska oblast", ""),
    # --- south ---
    ("Khersonska oblast", "Mykolaivska oblast", ""),
    ("Mykolaivska oblast", "Kirovohradska oblast", ""),
    ("Mykolaivska oblast", "Odeska oblast", ""),
    ("Odeska oblast", "Vinnytska oblast", ""),
    ("Odeska oblast", "Kirovohradska oblast", ""),
    ("Kirovohradska oblast", "Vinnytska oblast", ""),
    # --- west ---
    ("Zhytomyrska oblast", "Rivnenska oblast", ""),
    ("Zhytomyrska oblast", "Khmelnytska oblast", ""),
    ("Zhytomyrska oblast", "Vinnytska oblast", ""),
    ("Vinnytska oblast", "Khmelnytska oblast", ""),
    ("Khmelnytska oblast", "Rivnenska oblast", ""),
    ("Khmelnytska oblast", "Ternopilska oblast", ""),
    ("Khmelnytska oblast", "Chernivetska oblast", ""),
    ("Chernivetska oblast", "Ternopilska oblast", ""),
    ("Chernivetska oblast", "Ivano-Frankivska oblast", ""),
    ("Ternopilska oblast", "Rivnenska oblast", ""),
    ("Ternopilska oblast", "Lvivska oblast", ""),
    ("Ternopilska oblast", "Ivano-Frankivska oblast", ""),
    ("Ivano-Frankivska oblast", "Lvivska oblast", ""),
    ("Ivano-Frankivska oblast", "Zakarpatska oblast", ""),
    ("Zakarpatska oblast", "Lvivska oblast", ""),
    ("Lvivska oblast", "Volynska oblast", ""),
    ("Lvivska oblast", "Rivnenska oblast", ""),
    ("Volynska oblast", "Rivnenska oblast", ""),
]

# Pairs a reader might expect to be adjacent but that are NOT (checked by the
# author: Cherkasy-Dnipropetrovsk and Vinnytsia-Chernivtsi do not border).
_BELIEVED_NOT_ADJACENT: list[tuple[str, str]] = [
    ("Vinnytska oblast", "Chernivetska oblast"),
    ("Cherkaska oblast", "Dnipropetrovska oblast"),
    ("Kyiv City", "Zhytomyrska oblast"),
    ("Kyiv City", "Cherkaska oblast"),
]


def edges() -> list[tuple[str, str]]:
    """Undirected edges as sorted (a, b) tuples, sorted list."""
    return sorted({tuple(sorted((a, b))) for a, b, _ in _RAW_EDGES})


def adjacency() -> dict[str, list[str]]:
    """oblast -> sorted list of neighbouring oblasts, for every modelled oblast
    (isolated ones would get an empty list)."""
    result: dict[str, list[str]] = {o: [] for o in MODELLED_OBLASTS}
    for a, b in edges():
        result[a].append(b)
        result[b].append(a)
    return {o: sorted(ns) for o, ns in result.items()}


def main() -> None:
    es = edges()
    print("Land-border adjacency (checked by the author against a map).")
    print(f"{len(es)} border pairs among {len(MODELLED_OBLASTS)} units "
          f"(Luhansk and Crimea/Sevastopol omitted; Kyiv City borders only Kyivska oblast).\n")
    for a, b in es:
        print(f"{a} - {b}")
    print("\nPairs that are NOT adjacent:")
    for a, b in _BELIEVED_NOT_ADJACENT:
        print(f"{a} - {b}")
    print("\nPer-oblast neighbour counts:")
    for o, ns in adjacency().items():
        print(f"  {o}: {len(ns)}")


if __name__ == "__main__":
    main()
