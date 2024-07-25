from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import torch
import hickle as hkl
import sys
import io
from ase.io import read
from contextlib import redirect_stderr
import warnings
from mlelec.utils.pbc_utils import inverse_fourier_transform
from mlelec.data.pyscf_calculator import get_scell_phase, _instantiate_pyscf_mol

warnings.simplefilter('always', DeprecationWarning)

class QMDataset:
    '''
    Class containing information about the quantum chemistry calculation and its results.
    '''

    def __init__(
        self,
        frames: List,
        kmesh: Optional[Union[List[int], List[List[int]]]] = None,
        fock_kspace: Optional[Union[List, torch.Tensor, np.ndarray]] = None,
        fock_realspace: Optional[Union[Dict, torch.Tensor, np.ndarray]] = None,
        overlap_kspace: Optional[Union[List, torch.Tensor, np.ndarray]] = None,
        overlap_realspace: Optional[Union[Dict, torch.Tensor, np.ndarray]] = None,
        device: str = "cpu",
        orbs_name: str = "sto-3g",
        orbs: List = None,
        dimension: int = 3,
        fix_p_orbital_order: bool = False,
        apply_condon_shortley: bool = False,
    ):
        if fix_p_orbital_order or apply_condon_shortley:
            warnings.warn(
                "The `fix_p_orbital_order` and `apply_condon_shortley` options have been moved to MLDataset.",
                DeprecationWarning
            )

        self._device = device
        self._basis = orbs
        self._basis_name = orbs_name
        self._dimension = dimension
        self._structures = self._wrap_frames(frames)

        self._kmesh = self._set_kmesh(kmesh)
        self._nao = self._set_nao()
        self._ncore = self._set_ncore()

        self._initialize_pyscf_objects()

        self._set_matrices(
            fock_realspace=fock_realspace,
            fock_kspace=fock_kspace,
            overlap_realspace=overlap_realspace,
            overlap_kspace=overlap_kspace,
        )

    def __repr__(self):
        return (f"QMDataset(\n"
                f"  device: {self.device},\n"
                f"  basis_name: {self.basis_name},\n"
                f"  dimension: {self.dimension},\n"
                f"  nstructs: {self.nstructs},\n"
                f"  kmesh: {self.kmesh},\n"
                f"  nao: {self.nao},\n"
                f"  ncore: {self.ncore},\n"
                f"  is_molecule: {self.is_molecule},\n"
                f"  fock_realspace: {self.fock_realspace is not None},\n"
                f"  fock_kspace: {self.fock_kspace is not None},\n"
                f"  overlap_realspace: {self.overlap_realspace is not None},\n"
                f"  overlap_kspace: {self.overlap_kspace is not None}\n"
                f")")

    @classmethod
    def from_file(cls, frames_path: str, fock_realspace_path: Optional[str] = None, fock_kspace_path: Optional[str] = None, 
                  overlap_realspace_path: Optional[str] = None, overlap_kspace_path: Optional[str] = None, device: str = "cpu", 
                  orbs_name: str = "sto-3g", orbs: List = None, dimension: int = 3, frame_slice: Optional[Union[slice, str]] = None) -> 'QMDataset':
        """
        Create a QMDataset instance by loading frames and matrices from files.

        Args:
            frames_path (str): Path to the file containing the frames.
            fock_realspace_path (Optional[str]): Path to the file containing the Fock realspace matrices.
            fock_kspace_path (Optional[str]): Path to the file containing the Fock kspace matrices.
            overlap_realspace_path (Optional[str]): Path to the file containing the overlap realspace matrices.
            overlap_kspace_path (Optional[str]): Path to the file containing the overlap kspace matrices.
            device (str): Device to use for the dataset.
            orbs_name (str): Basis set name.
            orbs (List): Basis set orbitals.
            dimension (int): Dimension of the system.
            frame_slice (Optional[Union[slice, str]]): Slice object or string to select a subset of frames and matrices.

        Returns:
            QMDataset: An instance of QMDataset with loaded frames and matrices.
        """
        if isinstance(frame_slice, str):
            frame_slice = parse_slice(frame_slice)
        
        frames = cls.load_frames(frames_path)
        frames = frames[frame_slice]

        fock_realspace = cls.load_matrix(fock_realspace_path, device) if fock_realspace_path else None
        if fock_realspace is not None:
            fock_realspace = fock_realspace[frame_slice]
        
        fock_kspace = cls.load_matrix(fock_kspace_path, device) if fock_kspace_path else None
        if fock_kspace is not None:
            fock_kspace = fock_kspace[frame_slice]

        overlap_realspace = cls.load_matrix(overlap_realspace_path, device) if overlap_realspace_path else None
        if overlap_realspace is not None:
            overlap_realspace = overlap_realspace[frame_slice]
        
        overlap_kspace = cls.load_matrix(overlap_kspace_path, device) if overlap_kspace_path else None
        if overlap_kspace is not None:
            overlap_kspace = overlap_kspace[frame_slice]

        return cls(frames=frames, fock_realspace=fock_realspace, fock_kspace=fock_kspace, 
                   overlap_realspace=overlap_realspace, overlap_kspace=overlap_kspace, 
                   device=device, orbs_name=orbs_name, orbs=orbs, dimension=dimension)


    @staticmethod
    def load_frames(file_path: str) -> List:
        """
        Load frames from a file.

        Args:
            file_path (str): Path to the file containing the frames.

        Returns:
            List: Loaded frames.
        """
        # Implement the logic to load frames from the file
        frames = read(file_path, index=":")  # Assuming ASE-readable file format
        return frames

    @staticmethod
    def load_matrix(file_path: str, device: str) -> Union[Dict, List, torch.Tensor]:
        """
        Load a matrix from a file.

        Args:
            file_path (str): Path to the file containing the matrix.
            device (str): Device to use for the matrix.

        Returns:
            Union[Dict, List, torch.Tensor]: Loaded matrix.
        """
        if file_path.endswith('.pt'):
            matrix = torch.load(file_path, map_location=device)
        elif file_path.endswith('.npy'):
            matrix = np.load(file_path, allow_pickle=True)
            if isinstance(matrix, np.ndarray) and matrix.dtype == object:
                try:
                    # Dictionary
                    matrix = matrix.item()
                except ValueError: # can only convert an array of size 1 to a Python scalar
                    try:
                        # List of numpy arrays
                        matrix = matrix.astype(np.float64)
                    except ValueError: # can only convert an array of size 1 to a Python scalar
                        matrix = [np.array(m) for m in list(matrix)]
                except:
                    raise ValueError(f"Unsupported file type: {file_path}")

        elif file_path.endswith('.hkl'):
            matrix = hkl.load(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        if isinstance(matrix, dict):
            return {k: torch.tensor(v, device=device) if isinstance(v, np.ndarray) else v.to(device) for k, v in matrix.items()}
        elif isinstance(matrix, list) and isinstance(matrix[0], dict):
            return [{k: torch.tensor(v, device=device) if isinstance(v, np.ndarray) else v.to(device) for k, v in sub_matrix.items()} for sub_matrix in matrix]
        elif isinstance(matrix, list):
            return [torch.tensor(m, device=device) if isinstance(m, np.ndarray) else m.to(device) for m in matrix]
        elif isinstance(matrix, np.ndarray):
            return torch.from_numpy(matrix).to(device)
        else:
            return matrix.to(device)

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str):
        if value not in ['cpu', 'cuda']:
            raise ValueError("Device must be either 'cpu' or 'cuda'")
        self._device = value

    @property
    def basis(self) -> List:
        return self._basis

    @property
    def basis_name(self) -> str:
        return self._basis_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_molecule(self) -> bool:
        return self._dimension == 0

    @property
    def structures(self) -> List:
        return self._structures

    def _wrap_frames(self, frames: List) -> List:
        for f in frames:
            if self.dimension == 2:
                f.pbc = [True, True, False]
                f.wrap(center=(0, 0, 0), eps=1e-60)
                f.pbc = True
            elif self.dimension == 3:
                f.wrap(center=(0, 0, 0), eps=1e-60)
                f.pbc = True
            elif self.dimension == 0:
                f.pbc = False
            else:
                raise NotImplementedError('Dimension must be 0, 2, or 3')
        return frames

    @property
    def nstructs(self) -> int:
        return len(self.structures)

    @property
    def kmesh(self) -> Optional[List[Union[int, List[int]]]]:
        return self._kmesh

    def _set_kmesh(self, kmesh: Optional[Union[List[int], List[List[int]]]]) -> Optional[List[Union[int, List[int]]]]:
        if self.is_molecule:
            return None
        if kmesh is None:
            kmesh = [1, 1, 1]
        if isinstance(kmesh[0], list):
            if len(kmesh) != self.nstructs:
                raise ValueError("If kmesh is a list of lists, it must have the same length as the number of structures")
            return kmesh
        else:
            return [kmesh for _ in range(self.nstructs)]

    @property
    def nao(self) -> List[int]:
        return self._nao

    def _set_nao(self) -> List[int]:
        return [sum(len(self._basis[s]) for s in frame.numbers) for frame in self._structures]

    @property
    def ncore(self) -> Dict[int, int]:
        return self._ncore

    def _set_ncore(self) -> Dict[int, int]:
        ncore = {}
        for s in self._basis:
            basis = np.array(self._basis[s])
            nmin = np.min(basis[:, 0])
            ncore[s] = 0
            for n in np.arange(nmin):
                for l in range(n):
                    ncore[s] += 2 * (2 * l + 1)
            llist = set(basis[np.argwhere(basis[:, 0] == nmin)][:, 0, 1])
            llist_nmin = set(range(max(llist) + 1))
            l_diff = llist_nmin - llist
            for l in l_diff:
                ncore[s] += 2 * (2 * l + 1)
        return ncore

    def _initialize_pyscf_objects(self):
        if self.is_molecule:
            self._mols = self._initialize_pyscf_mol()
            self._cells = None
        else:
            self._mols = None
            self._cells = self._initialize_pyscf_cell()
            self._set_kpts()

    def _initialize_pyscf_cell(self) -> List:
        cells = []
        _stderr_capture = io.StringIO()
        with redirect_stderr(_stderr_capture):
            for ifr, structure in enumerate(self._structures):
                cell, _, _ = get_scell_phase(structure, self._kmesh[ifr], basis=self._basis_name)
                cells.append(cell)
        stderr_output = _stderr_capture.getvalue()
        if stderr_output:
            sys.stderr.write(stderr_output)
        return cells

    def _initialize_pyscf_mol(self) -> List:
        mols = []
        _stderr_capture = io.StringIO()
        with redirect_stderr(_stderr_capture):
            for structure in self._structures:
                mols.append(_instantiate_pyscf_mol(structure, basis=self._basis_name))
        stderr_output = _stderr_capture.getvalue()
        if stderr_output:
            sys.stderr.write(stderr_output)
        return mols

    @property
    def cells(self) -> List:
        if self.is_molecule:
            raise AttributeError('This system is not periodic')
        return self._cells

    @property
    def mols(self) -> List:
        if not self.is_molecule:
            raise AttributeError('This system is not a molecule')
        return self._mols

    @property
    def kpts_rel(self) -> List:
        return self._kpts_rel

    @property
    def kpts_abs(self) -> List:
        return self._kpts_abs

    def _set_kpts(self):
        self._kpts_rel = [c.get_scaled_kpts(c.make_kpts(k)) for c, k in zip(self.cells, self.kmesh)]
        self._kpts_abs = [c.get_abs_kpts(kpts) for c, kpts in zip(self.cells, self.kpts_rel)]

    @property
    def fock_realspace(self) -> Optional[Union[Dict, List[Dict]]]:
        return self._fock_realspace

    @property
    def fock_kspace(self) -> Optional[Union[np.ndarray, torch.Tensor, List[torch.Tensor]]]:
        return self._fock_kspace

    @property
    def overlap_realspace(self) -> Optional[Union[Dict, List[Dict]]]:
        return self._overlap_realspace

    @property
    def overlap_kspace(self) -> Optional[Union[np.ndarray, torch.Tensor, List[torch.Tensor]]]:
        return self._overlap_kspace

    def _set_matrices(
        self,
        fock_realspace: Optional[Union[Dict, List]] = None,
        fock_kspace: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
        overlap_realspace: Optional[Union[Dict, List]] = None,
        overlap_kspace: Optional[Union[np.ndarray, torch.Tensor, List]] = None
    ):
        self._fock_realspace, self._fock_kspace = self._assign_or_compute_matrices(
            fock_realspace, fock_kspace, self._set_fock_kspace)
        self._overlap_realspace, self._overlap_kspace = self._assign_or_compute_matrices(
            overlap_realspace, overlap_kspace, self._set_overlap_kspace)

    def _assign_or_compute_matrices(
        self,
        realspace: Optional[Union[Dict, List]],
        kspace: Optional[Union[np.ndarray, torch.Tensor, List]],
        kspace_setter: Any
    ) -> Tuple[Optional[Union[Dict, List]], Optional[Union[np.ndarray, torch.Tensor, List]]]:
        if kspace is not None and realspace is None:
            kspace_setter(kspace)
            realspace = None
        elif kspace is None and realspace is not None:
            realspace = self._set_matrices_realspace(realspace)
            if not self.is_molecule:
                kspace = self.bloch_sum(realspace, is_tensor=True)
        elif kspace is None and realspace is None:
            warnings.warn("Matrices not provided.")
            realspace = None
            kspace = None
        elif kspace is not None and realspace is not None:
            raise NotImplementedError("Check consistency between realspace and kspace matrices.")
        else:
            raise NotImplementedError("Unhandled condition.")
        return realspace, kspace

    def _set_matrices_realspace(self, matrices_realspace: Union[Dict, List[Dict]]) -> List[Dict]:
        if not isinstance(matrices_realspace[0], dict):
            assert self.is_molecule, "Matrices_realspace should be a dictionary unless it's a molecule"
            return matrices_realspace
        return [self._convert_matrix(m) for m in matrices_realspace]

    def _convert_matrix(self, matrix: Dict) -> Dict:
        return {k: self._to_tensor(v) for k, v in matrix.items()}

    def _to_tensor(self, data: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=self.device)
        elif isinstance(data, list):
            return torch.tensor(data, device=self.device)
        else:
            raise ValueError("Matrix elements should be torch.Tensor, numpy.ndarray, or list")

    def _set_matrices_kspace(self, matrices_kspace: Union[List, np.ndarray, torch.Tensor]) -> List[torch.Tensor]:
        if isinstance(matrices_kspace, list):
            return [self._to_tensor(m) for m in matrices_kspace]
        elif isinstance(matrices_kspace, (np.ndarray, torch.Tensor)):
            assert matrices_kspace.shape[0] == len(self.structures), "Provide matrices_kspace for each structure"
            return [self._to_tensor(matrices_kspace[i]) for i in range(matrices_kspace.shape[0])]
        else:
            raise TypeError("Matrices_kspace should be a list, np.ndarray, or torch.Tensor")

    def _set_fock_kspace(self, fock_kspace: Union[List, np.ndarray, torch.Tensor]):
        self._fock_kspace = self._set_matrices_kspace(fock_kspace)

    def _set_overlap_kspace(self, overlap_kspace: Union[List, np.ndarray, torch.Tensor]):
        self._overlap_kspace = self._set_matrices_kspace(overlap_kspace)

    def compute_matrices_realspace(self, matrices_kspace: Any):
        raise NotImplementedError("This must happen when the targets are computed!")

    def bloch_sum(
        self,
        matrices_realspace: List[Dict],
        is_tensor: bool = True,
        structure_ids: Optional[List[int]] = None
    ) -> List[Optional[torch.Tensor]]:
        matrices_kspace = []
        structure_ids = structure_ids or range(len(matrices_realspace))
        for ifr, H in zip(structure_ids, matrices_realspace):
            if H:
                H_T = self._stack_tensors(H, is_tensor)
                T_list = self._convert_keys_to_tensor(H, is_tensor)
                k = torch.from_numpy(self.kpts_rel[ifr]).to(device=self.device)
                matrices_kspace.append(inverse_fourier_transform(H_T, T_list=T_list, k=k, norm=1))
            else:
                matrices_kspace.append(None)
        return matrices_kspace

    def _stack_tensors(self, H: Dict, is_tensor: bool) -> torch.Tensor:
        if is_tensor:
            return torch.stack(list(H.values())).to(device=self.device)
        else:
            return torch.from_numpy(np.array(list(H.values()))).to(device=self.device)

    def _convert_keys_to_tensor(self, H: Dict, is_tensor: bool) -> torch.Tensor:
        if is_tensor:
            return torch.tensor(list(H.keys()), dtype=torch.float64, device=self.device)
        else:
            return torch.from_numpy(np.array(list(H.keys()), dtype=np.float64)).to(device=self.device)

    def __len__(self) -> int:
        return self.nstructs


def parse_slice(slice_str: str) -> slice:
    """
    Parse a slice string and return a slice object.
    
    Args:
        slice_str (str): A string representing a slice (e.g., '0:4', '::10').

    Returns:
        slice: A slice object.
    """
    return slice(*map(lambda x: int(x) if x else None, slice_str.split(':')))
