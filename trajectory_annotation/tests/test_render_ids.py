import numpy as np
from realmirror_annot.render import clean_ids, color_to_id, id_to_color


def test_id_color_roundtrip():
    for i in [0, 1, 24, 255, 256, 4097, 60001, 60002, 60003]:
        ids, ok = color_to_id(np.array([[id_to_color(i)]], dtype=np.uint8))
        assert ids[0, 0] == i and ok[0, 0]


def test_blended_edge_colours_are_rejected_mostly():
    rng = np.random.default_rng(0)
    ids = np.concatenate([np.arange(1, 120), [60001, 60002, 60003]])
    cols = np.array([id_to_color(int(i)) for i in ids], dtype=float)
    bad = tot = 0
    for _ in range(3000):
        a, b = rng.choice(len(ids), 2, replace=False)
        for w in (0.25, 0.5, 0.75):
            c = np.round(cols[a] * w + cols[b] * (1 - w)).astype(np.int32)[None, None]
            i, v = color_to_id(c)
            tot += 1
            bad += bool(v[0, 0] and i[0, 0] in ids and i[0, 0] not in (ids[a], ids[b]))
    assert bad / tot < 0.01


def test_clean_ids_uses_neighbour_majority():
    img = np.array([[id_to_color(5)] * 3, [id_to_color(5), [9, 0, 0], id_to_color(5)], [id_to_color(7)] * 3], dtype=np.uint8)
    ids, ok = color_to_id(img)
    out = clean_ids(ids, np.array([0, 5, 7]), ok)
    assert out[1, 1] == 5
