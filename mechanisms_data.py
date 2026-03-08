"""
ChemFactory - Reaction Mechanisms and Open World Recipe Data
Phase 3: Final Phase - Advanced Synthesis, Tier 3-5, and Protection Groups.
"""
from typing import List, Dict, Any, Optional

def get_reaction_key(r1: str, r2: str, catalyst: str = None) -> tuple[str, str, str]:
    sorted_r = sorted([r1, r2])
    cat = catalyst if catalyst else "None"
    return (sorted_r[0], sorted_r[1], cat)

# ---------------------------------------------------------------------------
# Indigo Synthesis Route
# ---------------------------------------------------------------------------
INDIGO_REACTIONS = {
    "nitration": {
        "product": "O=[N+]([O-])c1ccccc1", "product_name": "ニトロベンゼン",
        "message": "⚡ ニトロ化成功！\n💡ヒント：次はスズ(Sn)と塩酸で『還元』してアニリンを作ろう！",
        "byproducts": ["H2O"], "reaction_type": "nitration",
        "mechanism_steps": [
            {
                "step_name": "求電子攻撃",
                "reactants_smiles": ["[c:1]1[c:2][c:3][c:4][c:5][c:6]1", "O=[N+:10]=O"],
                "arrows": [{"from": "bond_1_2", "to": "atom_10", "type": "double"}],
                "intermediates_smiles": ["[c:1]1([H:11])[c:2]([N+:10](=O)[O-])[c:3][c:4][c:5][c:6]1"]
            },
            {
                "step_name": "脱プロトン化",
                "reactants_smiles": ["[c:1]1([H:11])[c:2]([N:10](=O)=O)[c:3][c:4][c:5][c:6]1", "[O:20]S(=O)(=O)O-"],
                "arrows": [
                    {"from": "atom_20", "to": "atom_11", "type": "double"},
                    {"from": "bond_1_11", "to": "bond_1_2", "type": "double"}
                ],
                "intermediates_smiles": ["O=[N+:10]([O-])c1ccccc1", "[O:20]S(=O)(=O)O"]
            }
        ]
    },
    "reduction": {
        "product": "Nc1ccccc1", "product_name": "アニリン",
        "message": "🔄 還元成功！\n💡ヒント：次は『Vilsmeier-Haack 反応』でホルミル化しよう！",
        "byproducts": ["H2O"], "reaction_type": "reduction",
        "mechanism_steps": [
            {
                "step_name": "ニトロ基の還元 (簡略化)",
                "reactants_smiles": ["[O:1]=[N+:2]([O:3])c1ccccc1", "[H:10][H:11]"],
                "arrows": [
                    {"from": "bond_10_11", "to": "atom_2", "type": "double"},
                    {"from": "bond_2_1", "to": "atom_1", "type": "double"}
                ],
                "intermediates_smiles": ["[O-:1][N:2](=O)c1ccccc1", "[H:10]+"]
            }
        ]
    },
    "vilsmeier_haack": {
        "product": "Nc1c(C=O)cccc1", "product_name": "2-アミノベンズアルデヒド",
        "message": "✨ Vilsmeier-Haack 成功！\n💡ヒント：次は『ジアゾ化』経由でニトロ基に戻そう！",
        "byproducts": ["POCl3_waste"], "reaction_type": "vilsmeier_haack",
        "mechanism_steps": [
            {
                "step_name": "Vilsmeier試薬の攻撃",
                "reactants_smiles": ["[c:1]1(NH2)[c:2][c:3][c:4][c:5][c:6]1", "[C:10](Cl)=[N+](C)C"],
                "arrows": [
                    {"from": "bond_1_2", "to": "atom_10", "type": "double"},
                    {"from": "bond_10_Cl", "to": "atom_Cl", "type": "double"}
                ],
                "intermediates_smiles": ["[c:1]1(NH2)[c:2]([C:10]H=[N+](C)C)cccc1", "Cl-"]
            }
        ]
    },
    "diazotization": {
        "product": "O=[N+]([O-])c1c(C=O)cccc1", "product_name": "2-ニトロベンズアルデヒド",
        "message": "🌀 ジアゾ化→ニトロ化成功！\n💡ヒント：次は『アセトン』と『NaOH』でインドール環を作ろう！",
        "byproducts": ["N2", "H2O"], "reaction_type": "diazotization",
        "mechanism_steps": [
            {
                "step_name": "ニトロソ化",
                "reactants_smiles": ["[N:1]H2c1c(C=O)cccc1", "[N+:10]=O"],
                "arrows": [{"from": "atom_1", "to": "atom_10", "type": "double"}],
                "intermediates_smiles": ["[N:1]H2([N:10]=O)+c1c(C=O)cccc1"]
            }
        ]
    },
    "baeyer_drewson": {
        "product": "Oc1c[nH]c2ccccc12", "product_name": "インドキシル",
        "message": "🔮 Baeyer-Drewson 成功！\n💡ヒント：最後は空気（酸素）で酸化だ！",
        "byproducts": ["H2O"], "reaction_type": "baeyer_drewson",
        "mechanism_steps": [
            {
                "step_name": "アルドール縮合",
                "reactants_smiles": ["[O:1]=[C:2]H[c:3]1c([N+:5](=O)O-)cccc1", "[C:10](=O)[C:11]H2-"],
                "arrows": [
                    {"from": "atom_11", "to": "atom_2", "type": "double"},
                    {"from": "bond_1_2", "to": "atom_1", "type": "double"}
                ],
                "intermediates_smiles": ["[O-:1][C:2]H([C:11]H2C(=O)C)[c:3]1c(NO2)cccc1"]
            }
        ]
    },
    "oxidative_dimerization": {
        "product": "O=C1C(=C2C(=O)Nc3ccccc32)Nc2ccccc12", "product_name": "🔵 インディゴ",
        "message": "🎉🔵 インディゴ合成達成！！！",
        "byproducts": ["H2O"], "reaction_type": "oxidative_dimerization",
        "mechanism_steps": [
            {
                "step_name": "酸化二量化",
                "reactants_smiles": ["[c:1]1([O:10]H)cnc2ccccc12", "[O:20]=[O:21]"],
                "arrows": [
                    {"from": "atom_10", "to": "atom_20", "type": "double"},
                    {"from": "bond_20_21", "to": "atom_21", "type": "double"}
                ],
                "intermediates_smiles": ["[c:1]1(=[O:10])cnc2ccccc12", "O-O-"]
            }
        ]
    },
}

# ---------------------------------------------------------------------------
# Open World Reaction Registry (Full Spectrum)
# ---------------------------------------------------------------------------
OPEN_WORLD_REACTIONS = {
    # --- Benzene Synthesis ---
    get_reaction_key("CCCCCC", "None", "Pt"): {
        "product": "c1ccccc1", "product_name": "ベンゼン",
        "message": "✨ 接触改質成功！",
        "byproducts": ["H2", "H2", "H2", "H2"], "reaction_type": "catalytic_reforming"
    },
    get_reaction_key("C=CC=C", "C#C", "None"): {
        "product": "C1=CCC=CC1", "product_name": "1,4-シクロヘキサジエン",
        "message": "✨ Diels-Alder成功！",
        "byproducts": [], "reaction_type": "diels_alder",
        "mechanism_steps": [
            {
                "step_name": "[4+2] 環化付加",
                "reactants_smiles": ["[C:1]=[C:2][C:3]=[C:4]", "[C:5]#[C:6]"],
                "arrows": [
                    {"from": "bond_1_2", "to": "bond_1_6", "type": "double"},
                    {"from": "bond_5_6", "to": "bond_4_5", "type": "double"},
                    {"from": "bond_3_4", "to": "bond_2_3", "type": "double"}
                ],
                "intermediates_smiles": ["C1:1=C:2[C:3][C:4]=C:5[C:6]1"]
            }
        ]
    },
    get_reaction_key("C1=CCC=CC1", "O=O", "Pd/C"): {
        "product": "c1ccccc1", "product_name": "ベンゼン",
        "message": "✨ 芳香族化成功！",
        "byproducts": ["H2O"], "reaction_type": "aromatization"
    },

    # --- Tier 1 & 2 (Already predefined in Phase 2) ---
    get_reaction_key("Oc1ccccc1", "O=C=O", "NaOH"): {
        "product": "Oc1ccccc1C(=O)O", "product_name": "サリチル酸",
        "message": "✨ コルベ・シュミット成功！",
        "byproducts": [], "reaction_type": "kolbe_schmitt",
        "mechanism_steps": [
            {
                "step_name": "求核攻撃",
                "reactants_smiles": ["[O-:1][c:2]1ccccc1", "[O:10]=[C:11]=[O:12]"],
                "arrows": [
                    {"from": "bond_1_2", "to": "atom_11", "type": "double"}, # Ortho attack simplified
                    {"from": "bond_11_12", "to": "atom_12", "type": "double"}
                ],
                "intermediates_smiles": ["O=c1ccccc1[C:11](=O)[O-:12]"]
            }
        ]
    },

    # --- Tier 3: Ibuprofen & Saccharin ---
    get_reaction_key("CC(C)Cc1ccccc1", "CC(=O)Cl", "AlCl3"): {
        "product": "CC(C)Cc1ccc(C(C)=O)cc1", "product_name": "アシル化中間体",
        "message": "✨ Friedel-Crafts アシル化成功！",
        "byproducts": ["HCl"], "reaction_type": "eas_acylation",
        "mechanism_steps": [
            {
                "step_name": "アシル化",
                "reactants_smiles": ["[c:1]1ccccc1CC(C)C", "[C:10](=O)(Cl)+"],
                "arrows": [{"from": "bond_1_2", "to": "atom_10", "type": "double"}],
                "intermediates_smiles": ["[c:1]1(H)cccc(CC(C)C)[c:2]1([C:10](=O)C)+"]
            }
        ]
    },
    get_reaction_key("CC(C)Cc1ccc(C(C)=O)cc1", "[C-]#[O+]", "Pd"): {
        "product": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "product_name": "💊 イブプロフェン",
        "message": "✨ BHCプロセス成功！カルボニル化によりイブプロフェンが完成した。",
        "byproducts": [], "reaction_type": "carbonylation"
    },
    get_reaction_key("Cc1ccccc1", "O=S(=O)(Cl)O", "None"): {
        "product": "Cc1ccccc1S(=O)(=O)Cl", "product_name": "o-トルエンスルホニルクロリド",
        "message": "✨ クロロスルホン化成功！",
        "byproducts": ["H2O"], "reaction_type": "chlorosulfonation",
        "mechanism_steps": [
            {
                "step_name": "求電子攻撃",
                "reactants_smiles": ["[c:1]1ccccc1C", "[S:10](=O)(=O)(Cl)+"],
                "arrows": [{"from": "bond_1_2", "to": "atom_10", "type": "double"}],
                "intermediates_smiles": ["Cc1ccccc1[S:10](=O)(=O)Cl"]
            }
        ]
    },
    get_reaction_key("Cc1ccccc1S(=O)(=O)Cl", "N", "None"): {
        "product": "Cc1ccccc1S(=O)(=O)N", "product_name": "o-トルエンスルホンアミド",
        "message": "✨ アミド化成功！",
        "byproducts": ["HCl"], "reaction_type": "amidation"
    },
    get_reaction_key("Cc1ccccc1S(=O)(=O)N", "O=[Mn](=O)(=O)[O-]", "Acid"): {
        "product": "O=C1NS(=O)(=O)c2ccccc12", "product_name": "💊 サッカリン",
        "message": "✨ 酸化・閉環成功！サッカリンが完成した。",
        "byproducts": ["H2O"], "reaction_type": "oxidation_cyclization"
    },

    # --- Tier 4: Nylon & Cubane ---
    get_reaction_key("O=C(Cl)CCCCC(=O)Cl", "NCCCCCCN", "None"): {
        "product": "O=C(CCCCC(=O)NCCCCCCN)NCCCCCCN", "product_name": "🧵 ナイロン6,6",
        "message": "✨ 界面重縮合成功！強靭な繊維、ナイロン6,6が誕生した。",
        "byproducts": ["HCl"], "reaction_type": "polycondensation",
        "mechanism_steps": [
            {
                "step_name": "求核攻撃と脱離",
                "reactants_smiles": ["[C:1](=O)Cl", "[N:10]H2"],
                "arrows": [
                    {"from": "atom_10", "to": "atom_1", "type": "double"},
                    {"from": "bond_1_Cl", "to": "atom_Cl", "type": "double"}
                ],
                "intermediates_smiles": ["[C:1](=O)[N:10]H", "HCl"]
            }
        ]
    },
    get_reaction_key("O=C1C=CC=C1", "O=C1C=CC=C1", "UV Lamp"): {
        "product": "O=C1C2C3C4C1C5C2C3C45", "product_name": "ケージドジオン",
        "message": "✨ 光化学的 [2+2] 成功！",
        "byproducts": [], "reaction_type": "photochemical_addition",
        "mechanism_steps": [
            {
                "step_name": "[2+2] 付加環化",
                "reactants_smiles": ["[C:1]=[C:2]", "[C:3]=[C:4]"],
                "arrows": [
                    {"from": "bond_1_2", "to": "atom_3", "type": "fishhook"},
                    {"from": "bond_1_2", "to": "atom_1", "type": "fishhook"},
                    {"from": "bond_3_4", "to": "atom_2", "type": "fishhook"},
                    {"from": "bond_3_4", "to": "atom_4", "type": "fishhook"}
                ],
                "intermediates_smiles": ["[C:1]1[C:2][C:4][C:3]1"]
            }
        ]
    },
    get_reaction_key("O=C1C2C3C4C1C5C2C3C45", "NaOH", "Heat"): {
        "product": "C12C3C4C1C5C2C3C45", "product_name": "🧊 キュバン",
        "message": "✨ Favorskii 転位成功！ひずみに満ちた立方体分子、キュバンの完成だ！",
        "byproducts": ["O=C=O"], "reaction_type": "favorskii_rearrangement",
        "mechanism_steps": [
            {
                "step_name": "求核攻撃と環縮小",
                "reactants_smiles": ["[C:1]1(=[O:2])[C:3][C:4]", "[O:10]H-"],
                "arrows": [
                    {"from": "atom_10", "to": "atom_1", "type": "double"},
                    {"from": "bond_1_3", "to": "atom_3", "type": "double"},
                    {"from": "atom_3", "to": "atom_4", "type": "double"}
                ],
                "intermediates_smiles": ["[O:2]=[C:1]([O:10]H)[C:3]..."]
            }
        ]
    },

    # --- Tier 5: Nanoputian ---
    get_reaction_key("c1ccccc1I", "C#Cc1ccc(C#C)cc1", "Pd/Cu"): {
        "product": "c1ccccc1C#Cc2ccc(C#C)cc2", "product_name": "上半身",
        "message": "✨ Sonogashira カップリング(1)成功！",
        "byproducts": ["HI"], "reaction_type": "sonogashira",
        "mechanism_steps": [
            {
                "step_name": "還元的脱離",
                "reactants_smiles": ["[Pd:10]([c:1]R)([C:20]#[C:21]R)", ""],
                "arrows": [
                    {"from": "bond_10_1", "to": "atom_20", "type": "double"}
                ],
                "intermediates_smiles": ["[c:1]R[C:20]#[C:21]R", "[Pd:10]"]
            }
        ]
    },

    # --- Protection & Deprotection ---

    # Amine: Acetylation (Protect) -> Hydrolysis (Deprotect)
    get_reaction_key("Nc1cc([N+](=O)[O-])ccc1", "CC(=O)OC(C)=O", "Acid"): {
        "product": "CC(=O)Nc1ccc([N+](=O)[O-])cc1", "product_name": "保護されたアニリン",
        "message": "🛡️ 保護成功！アミノ基をアセチル基で保護した。",
        "byproducts": ["CC(=O)O"], "reaction_type": "protection"
    },
    get_reaction_key("CC(=O)Nc1ccc([N+](=O)[O-])cc1", "O", "Acid"): {
        "product": "Nc1ccc([N+](=O)[O-])cc1", "product_name": "p-ニトロアニリン",
        "message": "🔓 脱保護成功！アミノ基が再生した。",
        "byproducts": ["CC(=O)O"], "reaction_type": "deprotection"
    },

    # Ketone: Acetalization
    get_reaction_key("CC(=O)c1ccc(C(=O)O)cc1", "OCCO", "Acid"): {
        "product": "CC1(O C C O 1)c2ccc(C(=O)O)cc2", "product_name": "保護されたケトン",
        "message": "🛡️ 保護成功！ケトンをアセタールとして保護した。",
        "byproducts": ["H2O"], "reaction_type": "protection_acetal",
        "mechanism_steps": [
            {
                "step_name": "求核攻撃",
                "reactants_smiles": ["[C:1](=O)R", "[O:10]H[C:11]"],
                "arrows": [{"from": "atom_10", "to": "atom_1", "type": "double"}],
                "intermediates_smiles": ["[C:1]([O:10][C:11])([O-]R)"]
            }
        ]
    },

    # Alcohol: TMS
    get_reaction_key("Oc1ccc(Br)cc1", "C[Si](C)(C)Cl", "Base"): {
        "product": "C[Si](C)(C)Oc1ccc(Br)cc1", "product_name": "保護されたフェノール",
        "message": "🛡️ 保護成功！TMS基でアルコールを保護した。",
        "byproducts": ["HCl"], "reaction_type": "protection_tms"
    },
    get_reaction_key("CC(C)(O)c1ccc(O[Si](C)(C)C)cc1", "TBAF", "None"): {
        "product": "CC(C)(O)c1ccc(O)cc1", "product_name": "4-(2-ヒドロキシプロパン-2-イル)フェノール",
        "message": "🔓 脱保護成功！F⁻イオンによりTMS基が外された。",
        "byproducts": ["C[Si](C)(C)F"], "reaction_type": "deprotection_tms"
    },

    # Alkyne: TMS
    get_reaction_key("O=Cc1ccc(C#C[Si](C)(C)C)cc1", "O=C([O-])[O-]", "None"): {
        "product": "O=Cc1ccc(C#C)cc1", "product_name": "4-エチニルベンズアルデヒド",
        "message": "🔓 脱保護成功！末端アルキンが再生した。",
        "byproducts": [], "reaction_type": "deprotection_alkyne"
    },
}

# ---------------------------------------------------------------------------
# Merge Indigo into Open World
# ---------------------------------------------------------------------------
_INDIGO_MAP = {
    "nitration": ("c1ccccc1", "HNO3", "Acid"),
    "reduction": ("O=[N+]([O-])c1ccccc1", "HCl", "Sn"),
    "vilsmeier_haack": ("Nc1ccccc1", "CN(C)C=O", "POCl3"),
    "diazotization": ("Nc1c(C=O)cccc1", "NaNO2", "HNO3"),
    "baeyer_drewson": ("O=[N+]([O-])c1c(C=O)cccc1", "CC(=O)C", "NaOH"),
    "oxidative_dimerization": ("Oc1c[nH]c2ccccc12", "O=O", "None"),
}

for token, (r1, r2, cat) in _INDIGO_MAP.items():
    key = get_reaction_key(r1, r2, cat)
    if token in INDIGO_REACTIONS:
        OPEN_WORLD_REACTIONS[key] = INDIGO_REACTIONS[token]
