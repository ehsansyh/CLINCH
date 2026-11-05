# -*- coding: utf-8 -*-
"""Target sequences used by the released checkpoints."""

# Wild-type HIV-1 protease, the obligate homodimer, two copies of the 99 residue chain.
# This is the exact string the predictor and the generator were trained against.
WT_HIV1_PROTEASE = (
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHK"
    "AIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHK"
    "AIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
)

MONOMER_LENGTH = 99
