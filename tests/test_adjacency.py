from src.adjacency import adjacency, edges
from src.config import EXCLUDED_OBLASTS, MODELLED_OBLASTS


def test_all_edge_endpoints_are_modelled_oblasts():
    for a, b in edges():
        assert a in MODELLED_OBLASTS and b in MODELLED_OBLASTS
        assert a != b


def test_no_edges_to_excluded_oblasts():
    flat = {o for e in edges() for o in e}
    assert not flat & set(EXCLUDED_OBLASTS)


def test_adjacency_is_symmetric_and_has_no_duplicates():
    adj = adjacency()
    for a, nbrs in adj.items():
        assert len(nbrs) == len(set(nbrs))
        for b in nbrs:
            assert a in adj[b]


def test_kyiv_city_borders_only_kyivska_oblast():
    assert adjacency()["Kyiv City"] == ["Kyivska oblast"]


def test_every_modelled_oblast_has_at_least_one_neighbour():
    assert all(len(v) >= 1 for v in adjacency().values())
