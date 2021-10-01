import io
import re

import numpy as np
import pandas as pd

import opti


class MipFile:
    """File reader for MODDE investigation files (.mip)"""

    def __init__(self, filename):
        """Read in a MODDE file.

        Args:
            filename (str or path): path to MODDE file
        """
        s = b"".join(open(filename, "rb").readlines())

        # Split into blocks of 1024 bytes
        num_blocks = len(s) // 1024
        assert len(s) % 1024 == 0
        blocks = [s[i * 1024 : (i + 1) * 1024] for i in range(num_blocks)]

        # Split blocks into header (first 21 bytes) and body
        headers = [b[0:21] for b in blocks]
        data = [b[21:] for b in blocks]

        # Parse the block headers to create a mapping {block_index: next_block_index}
        block_order = {}
        for i, header in enumerate(headers):
            i_next = np.where([h.startswith(header[15:19]) for h in headers])[0]
            block_order[i] = i_next[0] if len(i_next) == 1 else None

        # Join all blocks that belong together and decode to UTF-8
        self.parts = []
        while len(block_order) > 0:
            i = next(iter(block_order))  # get first key in ordered dict
            s = b""
            while i is not None:
                s += data[i]
                i = block_order.pop(i)
            s = re.sub(b"\r|\x00", b"", s).decode("utf-8", errors="ignore")
            self.parts.append(s)

        self.settings = self._get_design_settings()
        self.factors = {k: self.settings[k] for k in self.settings["Factors"]}
        self.responses = {k: self.settings[k] for k in self.settings["Responses"]}
        self.data = self._get_experimental_data()

    def _get_experimental_data(self):
        """Parse the experimental data.

        Returns:
            pd.DataFrame: dataframe of experimental data
        """
        part = [p for p in self.parts if p.startswith("ExpNo")][0]
        return pd.read_csv(io.StringIO(part), delimiter="\t", index_col="ExpNo")

    def _get_design_settings(self):
        """Parse the design settings.

        Returns:
            dict of dicts: dictionary with 'Factors', 'Responses', 'Options' as well as the individual variables.
        """
        part = [p for p in self.parts if p.startswith("[Status]")][0]
        settings = {}
        for line in part.split("\n"):
            if len(line) == 0:
                continue
            if line.startswith("["):
                thing = line.strip("[]")
                settings[thing] = {}
            else:
                key, value = line.split("=")
                settings[thing][key] = value
        return settings


def read_modde(filepath):
    """Read a problem specification from a MODDE file.

    Args:
        filepath (path-like): path to MODDE .mip file

    Returns:
        opti.Problem: problem specification
    """
    mip = MipFile(filepath)

    inputs = []
    outputs = []
    constraints = []
    formulation_parameters = []

    # build input space
    for name, props in mip.factors.items():
        if props["Use"] == "Uncontrolled":
            print(f"Uncontrolled factors not supported. Skipping {name}")
            continue
        if props["Type"] == "Formulation":
            domain = [float(s) for s in props["Settings"].split(",")]
            inputs.append(opti.Continuous(name=name, domain=domain))
            formulation_parameters.append(name)
        elif props["Type"] == "Quantitative":
            domain = [float(s) for s in props["Settings"].split(",")]
            inputs.append(opti.Continuous(name=name, domain=domain))
        elif props["Type"] == "Multilevel":
            domain = [float(s) for s in props["Settings"].split(",")]
            inputs.append(opti.Discrete(name=name, domain=domain))
        elif props["Type"] == "Qualitative":
            domain = props["Settings"].split(",")
            inputs.append(opti.Categorical(name=name, domain=domain))
    inputs = opti.Parameters(inputs)

    # build formulation constraint
    constraints.append(
        opti.constraint.LinearEquality(
            names=formulation_parameters,
            lhs=np.ones(len(formulation_parameters)),
            rhs=1,
        )
    )

    # build output space
    for name, props in mip.responses.items():
        # check if data available that allows to infer the domain
        vmin = mip.data[name].min()
        vmax = mip.data[name].max()
        if np.isfinite(vmin) or np.isfinite(vmax) and vmin < vmax:
            domain = [vmin, vmax]
        else:
            domain = [0, 1]
        dim = opti.Continuous(name=name, domain=domain)
        outputs.append(dim)
    outputs = opti.Parameters(outputs)

    # data
    data = mip.data.drop(columns=["ExpName", "InOut"])

    return opti.Problem(
        inputs=inputs, outputs=outputs, constraints=constraints, data=data
    )
