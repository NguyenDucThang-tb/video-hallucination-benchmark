import itertools


def test_season_paper_tcd_grid_is_complete_and_unique():
    frame_counts = (2, 4)
    contrast = ((1.0, 0.1), (0.5, 0.5))
    grid = set(itertools.product(frame_counts, contrast))
    assert grid == {
        (2, (1.0, 0.1)),
        (2, (0.5, 0.5)),
        (4, (1.0, 0.1)),
        (4, (0.5, 0.5)),
    }
