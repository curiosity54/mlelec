# script to generate data
import argparse

from pyscf_calculator import calculator

from mlelec.data.dataset import precomputed_molecules

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol", type=str, default="H2O")
    parser.add_argument("--data_file", type=str, default="../examples/data/")
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--target", type=str, default="fock")
    parser.add_argument("--save_path", type=str, default="../examples/data/")
    parser.add_argument("--dft", type=bool, default=False)
    parser.add_argument("--frame_slice", type=str, default=":")
    parser.add_argument("--use_precomputed", type=bool, default=True)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--symmetry", type=bool, default=False)
    parser.add_argument("--kpts", type=list, default=None)
    ## TODO: add more arguments for possible use cases
    args = parser.parse_args()
    if args.mol in precomputed_molecules and args.use_precomputed:
        data_file = precomputed_molecules[args.mol.lower()].value
        # perform no new calculation if data is precomputed

    pyscf_calc = calculator(
        structures=args.data_file, basis_set=args.basis, target=args.target
    )
    # Perform calculations in pyscf
    pyscf_calc.calculate(
        basis_set=args.basis,
        dft=args.dft,
        spin=args.spin,
        charge=args.charge,
        symmetry=args.symmetry,
        kpts=args.kpts,
    )
    # Save data
    pyscf_calc.save(args.save_path)
    # save
