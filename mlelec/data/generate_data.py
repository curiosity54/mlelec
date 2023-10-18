# script to generate data

import sys
import argparse
from pyscf_calculator import calculator

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_file",
)
parser.add_argument(
    "basis",
)
parser.add_argument(
    "target",
)
parser.add_argument(
    "save_path",
)


args = parser.parse_args()

pyscf_calc = calculator(
    structures=args.data_file, basis_set=args.basis, target=args.target
)
# Perform calculations in pyscf

# ...

# save
