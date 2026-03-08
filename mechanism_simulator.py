"""
ChemFactory - Mechanism Simulation Module
=========================================
Dynamically simulates molecular structure changes based on user-drawn electron movement arrows.
"""

import logging
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger("mechanism_simulator")

# --- Pydantic Models for Simulation ---

class Arrow(BaseModel):
    from_id: str = Field(..., alias="from")
    to_id: str = Field(..., alias="to")
    arrow_type: str = Field(default="double", alias="type")  # "double" or "fishhook"

    class Config:
        populate_by_name = True

class SimulateRequest(BaseModel):
    reactant_smiles: str
    target_smiles: str
    arrows: List[Arrow]

class AtomInfo(BaseModel):
    idx: int
    symbol: str
    x: float
    y: float
    is_reactive: bool = False

class BondInfo(BaseModel):
    begin_idx: int
    end_idx: int
    order: float

class PuzzleGraph(BaseModel):
    atoms: List[AtomInfo] = []
    bonds: List[BondInfo] = []

class SimulateResponse(BaseModel):
    success: bool
    is_correct: bool
    result_smiles: Optional[str] = None
    error_message: Optional[str] = None
    lone_pairs: Dict[str, int] = {}
    puzzle_graph: Optional[PuzzleGraph] = None

# --- Helper Functions ---

def parse_element_id(id_str: str) -> Tuple[Optional[str], List[int]]:
    """Parse 'atom_N' or 'bond_N_M' into (type, indices)."""
    if id_str.startswith("atom_"):
        try:
            return "atom", [int(id_str.split("_")[1])]
        except:
            return None, []
    elif id_str.startswith("bond_"):
        try:
            parts = id_str.split("_")
            return "bond", [int(parts[1]), int(parts[2])]
        except:
            return None, []
    return None, []

VALENCE_ELECTRONS = {
    "H": 1, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
    "P": 5, "S": 6, "Cl": 7, "Br": 7, "I": 7, "Mg": 2
}

def calculate_lone_pairs(mol: Chem.Mol) -> Dict[str, int]:
    """Calculate lone pairs for each atom based on valence, shared electrons, and charge."""
    lp_dict = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        
        # Calculate shared electrons (sum of bond orders)
        shared_electron_count = 0
        for bond in atom.GetBonds():
            # For lone pair calculation, we need the number of electrons SHARED.
            shared_electron_count += int(bond.GetBondTypeAsDouble())
            
        valence = VALENCE_ELECTRONS.get(symbol, 0)
        if valence == 0:
            lp_dict[f"atom_{atom.GetIdx()}"] = 0
            continue
            
        # Lone Pairs = (Valence - Shared - FormalCharge) / 2
        lps = (valence - shared_electron_count - charge) // 2
        lp_dict[f"atom_{atom.GetIdx()}"] = max(0, int(lps))
    return lp_dict

# --- Simulation Logic ---

class ElectronPusher:
    def __init__(self, smiles: str):
        self.mol = Chem.MolFromSmiles(smiles)
        if not self.mol:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Consistent with generate_puzzle_graph
        self.mol = Chem.AddHs(self.mol)
        self.rw_mol = Chem.RWMol(self.mol)
        # We work on explicit bonds. In mechanism, we usually assume Kekule structures.
        try:
            Chem.Kekulize(self.rw_mol, clearAromaticFlags=True)
        except:
            pass # Not aromatic or couldn't kekulize (not fatal for simple ones)

    def _get_bond_type(self, order: float) -> Chem.BondType:
        if order <= 1.0: return Chem.BondType.SINGLE
        if order <= 2.0: return Chem.BondType.DOUBLE
        if order <= 3.0: return Chem.BondType.TRIPLE
        return Chem.BondType.SINGLE

    def push_arrow(self, arrow: Arrow) -> Tuple[bool, str]:
        if arrow.arrow_type != "double":
            return False, f"Arrow type '{arrow.arrow_type}' not supported yet (Double arrows only)."
            
        s_type, s_indices = parse_element_id(arrow.from_id)
        e_type, e_indices = parse_element_id(arrow.to_id)
        
        if not s_type or not e_type:
            return False, f"Invalid element IDs: {arrow.from_id} -> {arrow.to_id}"

        # Current atom status
        donor_atom_idx = -1
        acceptor_atom_idx = -1
        
        # 1. Process Source (Donor)
        if s_type == "atom":
            donor_atom_idx = s_indices[0]
            # No bond to remove
        elif s_type == "bond":
            u, v = s_indices
            bond = self.rw_mol.GetBondBetweenAtoms(u, v)
            if not bond:
                return False, f"Source bond {arrow.from_id} not found."
            
            # Decrement bond order
            cur_order = bond.GetBondTypeAsDouble()
            new_order = cur_order - 1.0
            if new_order <= 0.1:
                self.rw_mol.RemoveBond(u, v)
            else:
                bond.SetBondType(self._get_bond_type(new_order))
            
            # Identify which atom remains connected or receives Electrons
            if e_type == "atom":
                target = e_indices[0]
                if target == u:
                    donor_atom_idx = v
                elif target == v:
                    donor_atom_idx = u
                else:
                    # Guess donor based on adjacency to target
                    neigh_u = [n.GetIdx() for n in self.rw_mol.GetAtomWithIdx(u).GetNeighbors()]
                    if target in neigh_u: donor_atom_idx = v
                    else: donor_atom_idx = u
            elif e_type == "bond":
                k, l = e_indices
                if u == k or u == l: donor_atom_idx = v
                elif v == k or v == l: donor_atom_idx = u
                else: donor_atom_idx = u
                
        # 2. Process Destination (Acceptor)
        if e_type == "atom":
            acceptor_atom_idx = e_indices[0]
            # Avoid adding a bond if it's a Bond-to-Atom move within the same bond (LP formation)
            # or if it's an Atom-to-Atom move to itself.
            is_lp_formation = (s_type == "bond" and acceptor_atom_idx in s_indices)
            if acceptor_atom_idx != donor_atom_idx and donor_atom_idx != -1 and not is_lp_formation:
                # Add/Increment bond between donor and acceptor
                existing = self.rw_mol.GetBondBetweenAtoms(donor_atom_idx, acceptor_atom_idx)
                if existing:
                    existing.SetBondType(self._get_bond_type(existing.GetBondTypeAsDouble() + 1.0))
                else:
                    self.rw_mol.AddBond(donor_atom_idx, acceptor_atom_idx, Chem.BondType.SINGLE)
        elif e_type == "bond":
            k, l = e_indices
            existing = self.rw_mol.GetBondBetweenAtoms(k, l)
            if existing:
                existing.SetBondType(self._get_bond_type(existing.GetBondTypeAsDouble() + 1.0))
            else:
                self.rw_mol.AddBond(k, l, Chem.BondType.SINGLE)
            
            # Charge destination: The atom of the bond that is NOT the anchor/shared atom.
            if donor_atom_idx != -1:
                shared_atoms = set(s_indices) & set(e_indices)
                if shared_atoms:
                    shared = list(shared_atoms)[0]
                    acceptor_atom_idx = l if k == shared else k
                else:
                    # Fallback if no shared atom
                    if donor_atom_idx == k: acceptor_atom_idx = l
                    elif donor_atom_idx == l: acceptor_atom_idx = k
                    else: acceptor_atom_idx = k
        
        # 3. Update Formal Charges
        if donor_atom_idx != -1:
            at = self.rw_mol.GetAtomWithIdx(donor_atom_idx)
            at.SetFormalCharge(at.GetFormalCharge() + 1)
        if acceptor_atom_idx != -1:
            at = self.rw_mol.GetAtomWithIdx(acceptor_atom_idx)
            at.SetFormalCharge(at.GetFormalCharge() - 1)
            
        return True, ""

    def simulate(self, arrows: List[Arrow]) -> Tuple[bool, str, str]:
        for arrow in arrows:
            success, msg = self.push_arrow(arrow)
            if not success:
                return False, msg, ""
        
        # Sanitize and Validate
        try:
            self.rw_mol.UpdatePropertyCache(strict=False)
            res = Chem.SanitizeMol(self.rw_mol, catchErrors=True)
            if res != Chem.SanitizeFlags.SANITIZE_NONE:
                for atom in self.rw_mol.GetAtoms():
                    try: atom.UpdatePropertyCache()
                    except Chem.AtomValenceException as ve:
                        return False, f"化学的に不可能な構造です: {atom.GetSymbol()} の結合数が多すぎます ({ve})", ""
                return False, f"サニタイズエラー: {res}", ""
        except Exception as e:
            return False, f"シミュレーションエラー: {str(e)}", ""

        # Remove Explicit Hs for comparison
        try:
            final_mol = Chem.RemoveHs(self.rw_mol)
            result_smiles = Chem.MolToSmiles(final_mol)
            return True, "", result_smiles
        except:
            return False, "SMILES出力に失敗しました", ""

# --- API Router ---

from fastapi import APIRouter

from fastapi import APIRouter
simulate_router = APIRouter()

# --- New endpoint helper ---
def local_generate_puzzle_graph(smiles: str) -> Optional[PuzzleGraph]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atoms = [AtomInfo(idx=a.GetIdx(), symbol=a.GetSymbol(), x=conf.GetAtomPosition(a.GetIdx()).x, y=conf.GetAtomPosition(a.GetIdx()).y, is_reactive=True) for a in mol.GetAtoms()]
        bonds = [BondInfo(begin_idx=b.GetBeginAtomIdx(), end_idx=b.GetEndAtomIdx(), order=float(b.GetBondTypeAsDouble())) for b in mol.GetBonds()]
        return PuzzleGraph(atoms=atoms, bonds=bonds)
    except:
        return None

@simulate_router.post("/simulate_mechanism", response_model=SimulateResponse)
async def simulate_mechanism(req: SimulateRequest):
    try:
        # Pre-simulation lone pairs
        start_mol = Chem.MolFromSmiles(req.reactant_smiles)
        if not start_mol:
            return SimulateResponse(success=False, is_correct=False, error_message="Invalid reactant SMILES")
        
        start_mol = Chem.AddHs(start_mol)
        lps = calculate_lone_pairs(start_mol)

        # Simulation
        pusher = ElectronPusher(req.reactant_smiles)
        success, err, result_smi = pusher.simulate(req.arrows)
        
        if not success:
            return SimulateResponse(success=False, is_correct=False, error_message=err, lone_pairs=lps)
        
        # Correctness check
        canonical_result = Chem.CanonSmiles(result_smi)
        canonical_target = Chem.CanonSmiles(req.target_smiles)
        is_correct = (canonical_result == canonical_target)
        
        # Generate new puzzle graph for next step
        pg = local_generate_puzzle_graph(result_smi)

        return SimulateResponse(
            success=True,
            is_correct=is_correct,
            result_smiles=result_smi,
            lone_pairs=lps,
            puzzle_graph=pg
        )
    except Exception as e:
        logger.exception("Mechanism simulation error")
        return SimulateResponse(success=False, is_correct=False, error_message=str(e))
