from baselines.IPPO.plot_eval_statistics import ROLE_LAYOUTS, parse_log_name


def test_role_scenario_log_names_are_parsed_by_family():
    metadata = parse_log_name("recipe_switch_0_cross_seed0_seed1.log")

    assert metadata == {
        "layout": "recipe_switch_0",
        "layout_group": "recipe_switch",
        "group_order": 2,
        "index": 0,
        "pairing": "cross_seed0_seed1",
    }


def test_statistics_catalog_matches_all_role_scenarios():
    assert ROLE_LAYOUTS == (
        "split_0",
        "outage_0",
        "recipe_switch_0",
        "distance_switch_0",
    )
