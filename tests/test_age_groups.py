from src.config import load_config
from src.data.age_groups import assign_age_group, build_age_groups, group_midpoint


def test_assign_age_group_boundaries():
    groups = build_age_groups(load_config())
    assert assign_age_group(0, groups).name == "child"
    assert assign_age_group(12, groups).name == "child"
    assert assign_age_group(13, groups).name == "teen"
    assert assign_age_group(56, groups).name == "senior"


def test_group_midpoint():
    groups = build_age_groups(load_config())
    assert group_midpoint("teen", groups) == 16.0

