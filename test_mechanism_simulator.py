import pytest
from fastapi.testclient import TestClient
from main_rdkit import app

client = TestClient(app)

def test_sn2_success():
    """SN2: ClC + OH- -> CO + Cl-"""
    payload = {
        "reactant_smiles": "ClC.[OH-]",
        "target_smiles": "CO.[Cl-]",
        "arrows": [
            {"from": "atom_2", "to": "atom_1", "type": "double"},   # O(2) attacks C(1)
            {"from": "bond_0_1", "to": "atom_0", "type": "double"} # Bond C(1)-Cl(0) breaks to Cl(0)
        ]
    }
    response = client.post("/simulate_mechanism", json=payload)
    data = response.json()
    print(f"\nSN2 Response: {data}")
    assert response.status_code == 200
    assert data["success"] is True, data.get("error_message")
    assert data["is_correct"] is True

def test_valence_error():
    """Methane (C) + OH- attack -> Valence Error (5 bonds on C)"""
    payload = {
        "reactant_smiles": "C.[OH-]",
        "target_smiles": "CO",
        "arrows": [
            {"from": "atom_1", "to": "atom_0", "type": "double"}
        ]
    }
    response = client.post("/simulate_mechanism", json=payload)
    data = response.json()
    print(f"\nValence Response: {data}")
    assert response.status_code == 200
    assert data["success"] is False
    assert "結合数が多すぎます" in data["error_message"]
