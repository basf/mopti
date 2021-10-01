from opti import read_modde


def test_read_modde():
    problem = read_modde("examples/bread.mip")
    assert problem.data is not None
    assert problem.inputs.names == [
        "kneading",
        "yeast_type",
        # "yeast_age", uncontrolled parameters not yet supported
        "flour",
        "water",
        "salt",
        "yeast",
        "kneading_time",
        "resting_time",
        "baking_time",
        "baking_temperature",
    ]
    assert problem.outputs.names == [
        "crust",
        "elasticity",
        "softness_24h",
        "taste",
        "appearance",
    ]
