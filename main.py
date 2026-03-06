"""
ChemFactory – Full Synthesis API (Tier 1 + Tier 2 Hybrid)
==============================================================
Tier 1: ローカルルールベース処理 (Pure Python, RDKit-free)
Tier 2: 外部 AI 推論フォールバック (Hugging Face Inference API)

Phase 0: アルカンのラジカルハロゲン化  R-H + X₂ --hν→ R-X + HX
Phase 1: ハロアルカンの置換/脱離反応
         - SN2 (低温 + 非プロトン性溶媒) R-X + Nu⁻ → R-Nu + X⁻
         - E2  (高温)                     R-X + Base → Alkene + HX + Base-H
Phase 2: グリニャール試薬
         - 調製:  R-X + Mg → R-Mg-X
         - 酸化:  R-Mg-X + O₂ → R-OH
         - 求核付加: R-Mg-X + R'CHO → R-CH(OH)-R'
Phase 3: 芳香族求電子置換 (EAS)
         - Friedel-Crafts アシル化: ArH + RCOCl + AlCl₃ → ArCOR + HCl
Phase 4: ナノプシャン組立 (NANO_HEAD + NANO_BODY → NANOPUTIAN)
Tier 2:  未知の組み合わせ → タール化
"""

from __future__ import annotations

import copy
import logging
import os
import re
import time
from enum import Enum
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("chemfactory")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ChemFactory – Tier 1+2 Hybrid API",
    description=(
        "Tier 1: ルールベース反応処理 (Phase 0 ラジカルハロゲン化 / Phase 1 SN2・E2) | "
        "Tier 2: AI 推論フォールバック (Hugging Face Inference API)"
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
#  SECTION 1:  共通 – 軽量 SMILES ケミストリーエンジン (RDKit-free)
# =====================================================================

# ハロゲン分子 SMILES → ハロゲン原子シンボル
HALOGEN_MAP: dict[str, str] = {
    "ClCl": "Cl",
    "BrBr": "Br",
    "FF": "F",
    "II": "I",
}

# ハロゲン原子シンボル (単体)
HALOGEN_ATOMS: set[str] = {"F", "Cl", "Br", "I"}

# ゲームオーバー扱いのハロゲン
FORBIDDEN_HALOGENS: dict[str, str] = {
    "FF": "💥 フッ素 (F₂) は反応性が高すぎて爆発しました！ゲームオーバー！",
    "II": "🧊 ヨウ素 (I₂) は反応性が低すぎて何も起きませんでした…",
}

# 求核剤/塩基の定義: SMILES → (名前, 求核性, 塩基性, 置換後の SMILES 断片)
NUCLEOPHILE_BASE_MAP: dict[str, dict] = {
    "O": {
        "name": "水酸化物イオン (OH⁻)",
        "nucleophilicity": "strong",
        "basicity": "strong",
        "fragment": "O",          # C-OH
        "product_suffix": "オール",
        "leaving_group": None,
    },
    "[O-]": {
        "name": "水酸化物イオン (OH⁻)",
        "nucleophilicity": "strong",
        "basicity": "strong",
        "fragment": "O",
        "product_suffix": "オール",
        "leaving_group": None,
    },
    "[NH3]": {
        "name": "アンモニア (NH₃)",
        "nucleophilicity": "moderate",
        "basicity": "weak",
        "fragment": "N",
        "product_suffix": "アミン",
        "leaving_group": None,
    },
    "N": {
        "name": "アンモニア (NH₃)",
        "nucleophilicity": "moderate",
        "basicity": "weak",
        "fragment": "N",
        "product_suffix": "アミン",
        "leaving_group": None,
    },
    "OCC": {
        "name": "エトキシド (EtO⁻)",
        "nucleophilicity": "moderate",
        "basicity": "very_strong",
        "fragment": "OCC",
        "product_suffix": "エチルエーテル",
        "leaving_group": None,
    },
    "OC(C)(C)C": {
        "name": "t-ブトキシド (t-BuO⁻)",
        "nucleophilicity": "weak",     # 嵩高い → 求核性低
        "basicity": "very_strong",     # 強塩基
        "fragment": "OC(C)(C)C",
        "product_suffix": "t-ブチルエーテル",
        "leaving_group": None,
    },
}

# IUPAC 風の炭素数プレフィックス
CARBON_PREFIX: dict[int, str] = {
    1: "メタン", 2: "エタン", 3: "プロパン", 4: "ブタン", 5: "ペンタン",
    6: "ヘキサン", 7: "ヘプタン", 8: "オクタン", 9: "ノナン", 10: "デカン",
}

ALKENE_PREFIX: dict[int, str] = {
    2: "エチレン", 3: "プロペン", 4: "ブテン", 5: "ペンテン",
    6: "ヘキセン", 7: "ヘプテン", 8: "オクテン",
}

HALOGEN_JP: dict[str, str] = {
    "Cl": "クロロ", "Br": "ブロモ", "F": "フルオロ", "I": "ヨード",
}


# ---------------------------------------------------------------------------
# Molecule: 汎用的な分子内部表現 (Phase 0 + 1 共用)
# ---------------------------------------------------------------------------

class Molecule:
    """
    炭素骨格ベースの分子内部表現。
    各炭素は隣接炭素、水素数、ハロゲン、置換基、二重結合情報を保持。
    """

    def __init__(self, carbons: list[dict]):
        """
        carbons: list of dicts, each with:
            - "neighbors": list[int]     (隣接する炭素のインデックス)
            - "h_count": int             (結合している水素の数)
            - "halogens": list[str]      (結合しているハロゲン記号, e.g. ["Cl"])
            - "substituents": list[str]  (ハロゲン以外の置換基, e.g. ["O", "N"])
            - "double_bond_to": list[int](二重結合で結合している炭素のインデックス)
        """
        self.carbons = carbons
        # 旧形式との互換性: 不足キーを補完
        for c in self.carbons:
            c.setdefault("substituents", [])
            c.setdefault("double_bond_to", [])

    @property
    def carbon_count(self) -> int:
        return len(self.carbons)

    def classify_carbon(self, idx: int) -> str:
        """炭素の級数を返す: primary / secondary / tertiary / quaternary"""
        neighbor_c_count = len(self.carbons[idx]["neighbors"])
        if neighbor_c_count <= 1:
            return "primary"
        elif neighbor_c_count == 2:
            return "secondary"
        elif neighbor_c_count == 3:
            return "tertiary"
        else:
            return "quaternary"

    def to_smiles(self) -> str:
        """分子を正規化 (canonical) SMILES 文字列に変換する。
        全ての候補開始点から SMILES を生成し、辞書順最小のものを選ぶ。
        これにより同じ分子は常に同じ SMILES を返す。
        """
        if self.carbon_count == 0:
            return ""
        # 末端炭素 (隣接炭素が1以下) を優先候補とする
        terminals = [i for i in range(self.carbon_count)
                     if len(self.carbons[i]["neighbors"]) <= 1]
        # 末端がなければ全ノードを候補に
        candidates = terminals if terminals else list(range(self.carbon_count))
        best = None
        for start in candidates:
            visited = [False] * self.carbon_count
            s = self._build_smiles(start, visited)
            if best is None or s < best:
                best = s
        return best or ""

    def _build_smiles(self, idx: int, visited: list[bool]) -> str:
        visited[idx] = True
        c = self.carbons[idx]

        # 隣接する未訪問の炭素
        unvisited_neighbors = [n for n in c["neighbors"] if not visited[n]]

        # ハロゲン置換基 — 分岐があるときだけ括弧で囲む
        hal_parts = []
        for h in c["halogens"]:
            hal_parts.append(h)

        # その他の置換基 (OH, NH2, etc.)
        sub_parts = []
        for s in c.get("substituents", []):
            sub_parts.append(s)

        # 終端炭素 (先に進む炭素がない)
        if not unvisited_neighbors:
            suffix = ""
            for h in hal_parts:
                suffix += h
            for s in sub_parts:
                suffix += s
            return f"C{suffix}"

        main = unvisited_neighbors[0]
        branches = unvisited_neighbors[1:]

        # 二重結合判定
        bond_char = "=" if main in c.get("double_bond_to", []) else ""

        # 分岐がある場合: ハロゲン/置換基はブランチとして括弧に入れる
        result = "C"
        # ハロゲンをブランチとして追加
        for h in hal_parts:
            result += f"({h})"
        # 置換基をブランチとして追加
        for s in sub_parts:
            result += f"({s})"
        # 炭素ブランチ
        for branch in branches:
            br_bond = "=" if branch in c.get("double_bond_to", []) else ""
            result += f"({br_bond}{self._build_smiles(branch, visited)})"
        result += f"{bond_char}{self._build_smiles(main, visited)}"

        return result


# ---------------------------------------------------------------------------
# SMILES パーサー: アルカン用 (Phase 0)
# ---------------------------------------------------------------------------

def parse_alkane_smiles(smiles: str) -> Optional[Molecule]:
    """
    アルカンの SMILES をパースして Molecule を返す。
    アルカンでない場合は None。

    対応: "C", "CC", "CCC", "CC(C)C", "CC(C)(C)C" 等
    """
    if not smiles or not isinstance(smiles, str):
        return None

    cleaned = smiles.strip()
    if not re.match(r'^[C()]+$', cleaned):
        return None

    carbons: list[dict] = []
    stack: list[int] = []
    current_parent: int = -1

    i = 0
    while i < len(cleaned):
        ch = cleaned[i]
        if ch == 'C':
            idx = len(carbons)
            new_c = {"neighbors": [], "h_count": 0, "halogens": [],
                     "substituents": [], "double_bond_to": []}
            carbons.append(new_c)
            if current_parent >= 0:
                carbons[current_parent]["neighbors"].append(idx)
                new_c["neighbors"].append(current_parent)
            current_parent = idx
            i += 1
        elif ch == '(':
            stack.append(current_parent)
            i += 1
        elif ch == ')':
            if not stack:
                return None
            current_parent = stack.pop()
            i += 1
        else:
            return None

    if stack:
        return None

    for c in carbons:
        bonds = len(c["neighbors"]) + len(c["halogens"])
        c["h_count"] = 4 - bonds

    if any(c["h_count"] < 0 for c in carbons):
        return None

    return Molecule(carbons)


# ---------------------------------------------------------------------------
# SMILES パーサー: ハロアルカン用 (Phase 1)
# ---------------------------------------------------------------------------

def parse_haloalkane_smiles(smiles: str) -> Optional[Molecule]:
    """
    ハロアルカンの SMILES をパースして Molecule を返す。
    C, Cl, Br, F, I, (, ) を含む文字列を処理する。

    対応例:
    - "CCl"        → クロロメタン
    - "CCCl"       → クロロエタン
    - "CC(Cl)C"    → 2-クロロプロパン
    - "CC(C)(Cl)C" → 2-クロロ-2-メチルプロパン (t-BuCl)
    - "CCBr"       → ブロモエタン
    """
    if not smiles or not isinstance(smiles, str):
        return None

    cleaned = smiles.strip()

    # 許可文字: C, l, B, r, F, I, (, )
    if not re.match(r'^[ClBrFI()]+$', cleaned):
        return None

    carbons: list[dict] = []
    stack: list[int] = []
    current_parent: int = -1

    i = 0
    while i < len(cleaned):
        ch = cleaned[i]

        if ch == 'C' and i + 1 < len(cleaned) and cleaned[i + 1] == 'l':
            # "Cl" → ハロゲンとして直前の炭素に付加
            if current_parent < 0:
                return None
            carbons[current_parent]["halogens"].append("Cl")
            i += 2

        elif ch == 'B' and i + 1 < len(cleaned) and cleaned[i + 1] == 'r':
            # "Br" → ハロゲン
            if current_parent < 0:
                return None
            carbons[current_parent]["halogens"].append("Br")
            i += 2

        elif ch == 'F':
            if current_parent < 0:
                return None
            carbons[current_parent]["halogens"].append("F")
            i += 1

        elif ch == 'I':
            if current_parent < 0:
                return None
            carbons[current_parent]["halogens"].append("I")
            i += 1

        elif ch == 'C':
            # 炭素原子
            idx = len(carbons)
            new_c = {"neighbors": [], "h_count": 0, "halogens": [],
                     "substituents": [], "double_bond_to": []}
            carbons.append(new_c)
            if current_parent >= 0:
                carbons[current_parent]["neighbors"].append(idx)
                new_c["neighbors"].append(current_parent)
            current_parent = idx
            i += 1

        elif ch == '(':
            stack.append(current_parent)
            i += 1

        elif ch == ')':
            if not stack:
                return None
            current_parent = stack.pop()
            i += 1

        else:
            return None

    if stack:
        return None

    # ハロゲンが 1 つもなければハロアルカンではない
    total_halogens = sum(len(c["halogens"]) for c in carbons)
    if total_halogens == 0:
        return None

    # 水素数を計算
    for c in carbons:
        bonds = len(c["neighbors"]) + len(c["halogens"]) + len(c["substituents"])
        c["h_count"] = 4 - bonds

    if any(c["h_count"] < 0 for c in carbons):
        return None

    return Molecule(carbons)


# =====================================================================
#  SECTION 2:  Phase 0 – ラジカルハロゲン化
# =====================================================================

def radical_halogenation(mol: Molecule, halogen: str) -> list[dict]:
    """
    ラジカルハロゲン化を実行し、全ての位置異性体を返す。

    SMARTS 相当ルール: [C:1]([H]) → [C:1](X)
    各 C-H を 1 つ C-X に置換した生成物を列挙。
    """
    results: list[dict] = []
    seen: set[str] = set()

    for idx, c in enumerate(mol.carbons):
        if c["h_count"] <= 0:
            continue

        new_carbons = copy.deepcopy(mol.carbons)
        new_carbons[idx]["halogens"].append(halogen)
        new_carbons[idx]["h_count"] -= 1

        new_mol = Molecule(new_carbons)
        canonical = _canonical_key(new_mol)
        if canonical in seen:
            continue
        seen.add(canonical)

        carbon_class = mol.classify_carbon(idx)
        name = _name_haloalkane(new_mol, halogen, carbon_class)

        results.append({
            "smiles": new_mol.to_smiles(),
            "carbon_class": carbon_class,
            "position": idx,
            "name": name,
        })

    return results


# =====================================================================
#  SECTION 3:  Phase 1 – SN2 置換反応 & E2 脱離反応
# =====================================================================

# --- SN2 ---
# SMARTS 相当: [C:1](X) + Nu⁻ → [C:1](Nu) + X⁻
# 条件: 温度 < 60, 非プロトン性溶媒, 第一級 or 第二級炭素

def sn2_substitution(
    mol: Molecule,
    nucleophile_info: dict,
) -> list[dict]:
    """
    SN2 置換反応を実行する。
    ハロゲンが結合している炭素が 第一級 or メチル の場合のみ進行。
    (第二級は E2 と競合するが、非プロトン性溶媒で SN2 優先とする)

    Returns: list of product dicts
    """
    results: list[dict] = []
    seen: set[str] = set()

    for idx, c in enumerate(mol.carbons):
        if not c["halogens"]:
            continue

        carbon_class = mol.classify_carbon(idx)

        # 第三級炭素では SN2 は立体障害で進行しない
        if carbon_class in ("tertiary", "quaternary"):
            continue

        # 各ハロゲンについて置換を実行
        for hal_idx, halogen in enumerate(c["halogens"]):
            new_carbons = copy.deepcopy(mol.carbons)
            # ハロゲンを除去
            new_carbons[idx]["halogens"].pop(hal_idx)
            # 求核剤を付加
            fragment = nucleophile_info["fragment"]
            new_carbons[idx]["substituents"].append(fragment)

            new_mol = Molecule(new_carbons)
            # 水素数を再計算
            for c2 in new_mol.carbons:
                bonds = (len(c2["neighbors"]) + len(c2["halogens"])
                         + len(c2["substituents"]) + len(c2["double_bond_to"]))
                c2["h_count"] = 4 - bonds

            smiles = new_mol.to_smiles()
            canonical = _canonical_key(new_mol)
            if canonical in seen:
                continue
            seen.add(canonical)

            c_count = new_mol.carbon_count
            base_name = CARBON_PREFIX.get(c_count, f"C{c_count}")
            product_name = f"{base_name}{nucleophile_info['product_suffix']} (SN2)"

            results.append({
                "smiles": smiles,
                "name": product_name,
                "carbon_class": carbon_class,
                "reaction_type": "SN2",
                "leaving_group": halogen,
            })

    return results


# --- E2 ---
# SMARTS 相当: [C:1]([H])-[C:2](X) + Base → [C:1]=[C:2] + HX + Base-H
# 条件: 温度 ≥ 60, β-水素が存在する(隣接炭素に H がある)

def e2_elimination(mol: Molecule) -> list[dict]:
    """
    E2 脱離反応を実行する。
    ハロゲンが結合している炭素(α炭素)の隣接炭素(β炭素)から H を抜去し、
    二重結合を形成する。

    ザイツェフ則: 多置換アルケンを優先的に生成。

    Returns: list of product dicts
    """
    results: list[dict] = []
    seen: set[str] = set()

    for alpha_idx, alpha_c in enumerate(mol.carbons):
        if not alpha_c["halogens"]:
            continue

        # 各ハロゲンについて脱離を実行
        for hal_idx, halogen in enumerate(alpha_c["halogens"]):
            # β炭素 (α炭素に隣接する炭素) を探す
            for beta_idx in alpha_c["neighbors"]:
                beta_c = mol.carbons[beta_idx]

                # β炭素に水素がなければ脱離できない
                if beta_c["h_count"] <= 0:
                    continue

                new_carbons = copy.deepcopy(mol.carbons)
                # α炭素からハロゲンを除去
                new_carbons[alpha_idx]["halogens"].pop(hal_idx)
                # β炭素から水素を除去
                new_carbons[beta_idx]["h_count"] -= 1
                # α-β 間に二重結合を形成
                new_carbons[alpha_idx]["double_bond_to"].append(beta_idx)
                new_carbons[beta_idx]["double_bond_to"].append(alpha_idx)

                new_mol = Molecule(new_carbons)
                # 水素数を再計算 (二重結合分を考慮)
                for c2 in new_mol.carbons:
                    bonds = (len(c2["neighbors"]) + len(c2["halogens"])
                             + len(c2["substituents"])
                             + len(c2["double_bond_to"]))  # 二重結合 = 追加1本
                    c2["h_count"] = 4 - bonds

                smiles = new_mol.to_smiles()
                canonical = _canonical_key(new_mol)
                if canonical in seen:
                    continue
                seen.add(canonical)

                # ザイツェフ判定: 二重結合炭素に結合した非H置換基数
                substitution_degree = (
                    len(new_mol.carbons[alpha_idx]["neighbors"])
                    + len(new_mol.carbons[beta_idx]["neighbors"])
                    - 2  # α-β 間の結合を除く
                )

                c_count = new_mol.carbon_count
                base_name = ALKENE_PREFIX.get(c_count, f"C{c_count}アルケン")
                product_name = f"{base_name} (E2, ザイツェフ度={substitution_degree})"

                results.append({
                    "smiles": smiles,
                    "name": product_name,
                    "carbon_class": mol.classify_carbon(alpha_idx),
                    "reaction_type": "E2",
                    "leaving_group": halogen,
                    "zaitsev_degree": substitution_degree,
                })

    # ザイツェフ則: 多置換アルケンを先頭に
    results.sort(key=lambda r: r.get("zaitsev_degree", 0), reverse=True)
    return results


# =====================================================================
#  SECTION 4:  反応タイプの自動判定
# =====================================================================

# グリニャール試薬のパターン検出
def _is_grignard_reagent(smiles: str) -> bool:
    """SMILESがグリニャール試薬 (R-Mg-X) かどうかを判定する。"""
    return "[Mg]" in smiles or "Mg" in smiles

def _parse_grignard(smiles: str) -> Optional[dict]:
    """
    グリニャール試薬のSMILESをパースする。
    例: [CH3][Mg][Cl] → {"r_group": "C", "carbon_count": 1, "halogen": "Cl"}
    例: [CH3][CH2][Mg][Cl] → {"r_group": "CC", "carbon_count": 2, "halogen": "Cl"}
    """
    import re as _re
    # パターン: [CHn]...[Mg][X]
    pattern = _re.compile(r'^((?:\[CH\d?\])+)\[Mg\]\[(Cl|Br|I)\]$')
    m = pattern.match(smiles)
    if not m:
        return None
    ch_groups = _re.findall(r'\[CH(\d?)\]', m.group(1))
    carbon_count = len(ch_groups)
    r_group = "C" * carbon_count
    halogen = m.group(2)
    return {"r_group": r_group, "carbon_count": carbon_count, "halogen": halogen}


def detect_reaction_type(
    reagent_1: str,
    reagent_2: str,
    condition_light: bool,
    temperature: Optional[int],
    solvent_type: Optional[str],
    catalyst_alcl3: bool = False,
) -> str:
    """
    入力試薬から反応タイプを自動判定する。

    Returns:
        "radical_halogenation" | "substitution_elimination" |
        "grignard_preparation" | "grignard_reaction" |
        "eas_acylation" | "nanoputian_assembly" | "unknown"
    """
    # ナノプシャン組立: NANO_HEAD + NANO_BODY
    if set([reagent_1, reagent_2]) == {"NANO_HEAD", "NANO_BODY"}:
        return "nanoputian_assembly"

    # ハロゲン分子 (X₂) + アルカン + 光 → ラジカルハロゲン化
    if reagent_2 in HALOGEN_MAP:
        return "radical_halogenation"

    # マグネシウム → グリニャール試薬調製
    if reagent_2 == "Mg":
        return "grignard_preparation"

    # グリニャール試薬 + 求電子剤 → グリニャール反応
    if _is_grignard_reagent(reagent_1):
        return "grignard_reaction"

    # 芳香族 + 塩化アシル + AlCl3 触媒 → EAS
    if catalyst_alcl3 and ("c1ccccc1" in reagent_1 or "c1ccccc1" in reagent_2):
        return "eas_acylation"

    # 求核剤/塩基 + ハロアルカン → 置換 or 脱離
    if reagent_2 in NUCLEOPHILE_BASE_MAP:
        haloalkane = parse_haloalkane_smiles(reagent_1)
        if haloalkane is not None:
            return "substitution_elimination"

    # === インディゴルート反応 ===
    # ニトロ化: ベンゼン + HNO3 (H2SO4触媒) → ニトロベンゼン
    if reagent_2 == "HNO3" and (reagent_1 in ("c1ccccc1", "C1=CC=CC=C1")):
        return "nitration"
    
    # 還元: ニトロベンゼン + Sn/HCl → アニリン
    if reagent_1 == "NITROBENZENE" and reagent_2 == "Sn_HCl":
        return "reduction_nitro"
    
    # Vilsmeier-Haack: アニリン + DMF (POCl3触媒) → 2-アミノベンズアルデヒド
    if reagent_1 == "ANILINE" and reagent_2 == "DMF":
        return "vilsmeier_haack"
    
    # ジアゾ化: 2-アミノベンズアルデヒド + NaNO2 → 2-ニトロベンズアルデヒド
    if reagent_1 == "2_AMINOBENZALDEHYDE" and reagent_2 == "NaNO2":
        return "diazotization"
    
    # アルドール縮合: 2-ニトロベンズアルデヒド + アセトン → インドキシル前駆体
    if reagent_1 == "2_NITROBENZALDEHYDE" and reagent_2 == "ACETONE":
        return "aldol_condensation"
    
    # 酸化二量化: インドキシル + 空気 → インディゴ
    if reagent_1 == "INDOXYL" and reagent_2 == "AIR_O2":
        return "oxidative_dimerization"

    return "unknown"


# =====================================================================
#  SECTION 5:  ヘルパー関数
# =====================================================================

def _canonical_key(mol: Molecule) -> str:
    """分子の正規化キーを生成してトポロジカルな重複を検出する。"""
    descriptors = []
    for c in mol.carbons:
        desc = (
            len(c["neighbors"]),
            tuple(sorted(c["halogens"])),
            tuple(sorted(c.get("substituents", []))),
            tuple(sorted(c.get("double_bond_to", []))),
            c["h_count"],
        )
        descriptors.append(desc)
    descriptors.sort()
    return str(descriptors)


def _name_haloalkane(mol: Molecule, halogen: str, carbon_class: str) -> str:
    """ラジカルハロゲン化生成物のゲーム向け日本語名。"""
    c_count = mol.carbon_count
    base = CARBON_PREFIX.get(c_count, f"C{c_count}アルカン")
    hal_prefix = HALOGEN_JP.get(halogen, halogen)
    class_jp = {"primary": "第一級", "secondary": "第二級", "tertiary": "第三級"}
    class_label = class_jp.get(carbon_class, "")

    total_hal = sum(len(c["halogens"]) for c in mol.carbons)
    if total_hal == 1:
        return f"{hal_prefix}{base} ({class_label})"
    else:
        num_map = {2: "ジ", 3: "トリ", 4: "テトラ"}
        num_prefix = num_map.get(total_hal, f"{total_hal}-")
        return f"{num_prefix}{hal_prefix}{base}"


# =====================================================================
#  SECTION 6:  Pydantic models
# =====================================================================

class ReactRequest(BaseModel):
    """反応リクエスト (全Phase統合)"""
    reagent_1: str = Field(
        ...,
        examples=["C", "CC", "CCCl", "CC(Cl)C", "c1ccccc1"],
        description="基質の SMILES (アルカン, ハロアルカン, 芳香族, グリニャール試薬, 特殊トークン)",
    )
    reagent_2: str = Field(
        ...,
        examples=["ClCl", "BrBr", "O", "Mg", "CC(=O)Cl", "NANO_BODY"],
        description="試薬の SMILES (ハロゲン分子, 求核剤, Mg, 塩化アシル, 特殊トークン)",
    )
    condition_light: bool = Field(
        ...,
        examples=[True],
        description="光 (hν) 条件の有無 (Phase 0 用)",
    )
    # --- Phase 1 追加パラメータ ---
    temperature: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        examples=[25, 80],
        description="反応温度 (0-100℃)。Phase 1 で使用。",
    )
    solvent_type: Optional[str] = Field(
        None,
        examples=["protic", "aprotic"],
        description="溶媒タイプ: 'protic' (プロトン性) or 'aprotic' (非プロトン性)。Phase 1 で使用。",
    )
    # --- Phase 2/3 追加パラメータ ---
    catalyst: Optional[str] = Field(
        None,
        description="追加の触媒指定 (Sn, Fe, Pt/Heat, Pd/Cu, Acid 等)",
    )


class ResultStatus(str, Enum):
    SUCCESS = "success"
    NO_REACTION = "no_reaction"
    GAME_OVER = "game_over"


class ProductInfo(BaseModel):
    """生成物情報"""
    smiles: str
    name: str
    carbon_class: str = Field(description="置換された炭素の級数")
    reaction_type: Optional[str] = Field(
        None, description="反応タイプ (SN2 / E2 / radical)",
    )


class ReactResponse(BaseModel):
    """反応レスポンス"""
    status: ResultStatus
    message: str
    reagent_1_smiles: str
    reagent_2_smiles: str
    reaction_type: Optional[str] = Field(
        None, description="判定された反応タイプ",
    )
    products: list[ProductInfo] = Field(default_factory=list)
    byproducts: list[str] = Field(
        default_factory=list,
        description="副生成物の SMILES / 名称リスト",
    )
    # 後方互換
    hx_byproduct: Optional[str] = Field(None, description="副生成物 HX の SMILES")
    # Phase 1 追加
    condition_summary: Optional[str] = Field(
        None, description="適用された反応条件のサマリー",
    )
    # --- Tier 2 AI 推論追加フィールド ---
    tier: Optional[str] = Field(
        None, description="処理 Tier: 'tier1_rule' or 'tier2_ai'",
    )
    ai_latency_seconds: Optional[float] = Field(
        None, description="Tier 2 AI 推論にかかった時間 (秒)",
    )
    ai_model: Optional[str] = Field(
        None, description="使用された AI モデル名",
    )


# =====================================================================
#  SECTION 7:  エンドポイント
# =====================================================================

# ---------------------------------------------------------------------------
# SMILES 正規化ヘルパー
# ---------------------------------------------------------------------------
def canonicalize_smiles(smiles: str) -> str:
    """
    SMILES 文字列を正規化する。
    パース可能なら Molecule を経由して canonical SMILES を生成。
    パース不可能な場合は元の文字列をそのまま返す。
    特殊トークン (NANO_HEAD, NANO_BODY, etc.) はそのまま返す。
    """
    # 特殊トークンはそのまま返す
    special_tokens = {"NANO_HEAD", "NANO_BODY", "NANOPUTIAN", "HEAT", "SOLVENT",
                      "MYSTERY", "ALL", "AlCl3", "Mg", "O=O", "C=O", "HCl",
                      "HBr", "ClCl", "BrBr",
                      # インディゴルート
                      "HNO3", "NITROBENZENE", "Sn_HCl", "ANILINE", "DMF",
                      "2_AMINOBENZALDEHYDE", "NaNO2", "2_NITROBENZALDEHYDE",
                      "ACETONE", "INDOXYL", "AIR_O2", "INDIGO", "DYE_MASTER"}
    if smiles in special_tokens:
        return smiles

    # ハロアルカン (Cl, Br, F, I を含む) をまず試す
    mol = parse_haloalkane_smiles(smiles)
    if mol is not None:
        return mol.to_smiles()

    # アルカンを試す
    mol = parse_alkane_smiles(smiles)
    if mol is not None:
        return mol.to_smiles()

    # パース不可能 → 元の SMILES をそのまま返す
    return smiles


@app.post("/canonicalize")
async def canonicalize_endpoint(body: dict):
    """SMILES を正規化 (canonical) 形式に変換する。"""
    smiles = body.get("smiles", "")
    canonical = canonicalize_smiles(smiles)
    return {"original": smiles, "canonical": canonical}


@app.get("/")
async def root():
    """ヘルスチェック & API 情報"""
    hf_configured = bool(os.environ.get("HF_TOKEN"))
    return {
        "service": "ChemFactory Tier 1+2 Hybrid API",
        "status": "running",
        "version": "0.3.1",
        "description": "Tier 1 ルールベース + Tier 2 AI 推論ハイブリッド反応シミュレーター",
        "tier2_ai_enabled": hf_configured,
        "tier2_ai_note": "HF_TOKEN 環境変数を設定すると Tier 2 AI 推論が有効になります" if not hf_configured else "Tier 2 AI 推論が有効です",
        "endpoints": {
            "POST /react": "化学反応を実行 (Tier 1: ルールベース → Tier 2: AI フォールバック)",
            "POST /canonicalize": "SMILES を正規化形式に変換",
            "GET /docs": "Swagger UI (API ドキュメント)",
        },
    }


# ---------------------------------------------------------------------------
# Open World Reaction Registry (Tier 1.5)
# ---------------------------------------------------------------------------
def get_reaction_key(r1: str, r2: str, catalyst: str = None) -> tuple[str, str, str]:
    sorted_r = sorted([r1, r2])
    cat = catalyst if catalyst else "None"
    return (sorted_r[0], sorted_r[1], cat)

OPEN_WORLD_REACTIONS = {
    # --------------------------------------------------------
    # Tier 1: アスピリンルート
    # --------------------------------------------------------
    get_reaction_key("Oc1ccccc1", "O=C=O", "NaOH"): {
        "product": "OC1=CC=CC=C1C(=O)O",
        "product_name": "サリチル酸",
        "message": "✨ コルベ・シュミット反応成功！フェノールからサリチル酸を合成した。",
        "byproducts": [],
        "reaction_type": "kolbe_schmitt"
    },
    get_reaction_key("OC1=CC=CC=C1C(=O)O", "CC(=O)OC(C)=O", "H2SO4"): {
        "product": "CC(=O)Oc1ccccc1C(=O)O",
        "product_name": "💊 アスピリン",
        "message": "✨ エステル化成功！世界初の人工合成医薬品「アスピリン」が完成した！",
        "byproducts": ["CC(=O)O"], # 酢酸
        "reaction_type": "esterification"
    },
    # --------------------------------------------------------
    # Tier 1: パラセタモールルート
    # --------------------------------------------------------
    get_reaction_key("Oc1ccccc1", "HNO3", "Acid (酸)"): {
        "product": "O=[N+]([O-])c1ccc(O)cc1",
        "product_name": "p-ニトロフェノール",
        "message": "✨ ニトロ化成功！フェノールからp-ニトロフェノールを合成した。",
        "byproducts": ["H2O"],
        "reaction_type": "nitration"
    },
    get_reaction_key("O=[N+]([O-])c1ccc(O)cc1", "[H][H]", "Pd/C"): {
        "product": "Nc1ccc(O)cc1",
        "product_name": "p-アミノフェノール",
        "message": "✨ 接触還元成功！p-ニトロフェノールからp-アミノフェノールを合成した。",
        "byproducts": ["H2O", "H2O"],
        "reaction_type": "reduction"
    },
    get_reaction_key("O=[N+]([O-])c1ccc(O)cc1", "[H][H]", "Sn (スズ)"): {
        "product": "Nc1ccc(O)cc1",
        "product_name": "p-アミノフェノール",
        "message": "✨ スズ還元成功！p-ニトロフェノールからp-アミノフェノールを合成した。",
        "byproducts": ["H2O", "H2O"],
        "reaction_type": "reduction"
    },
    get_reaction_key("Nc1ccc(O)cc1", "CC(=O)OC(C)=O", "None"): {
        "product": "CC(=O)Nc1ccc(O)cc1",
        "product_name": "💊 パラセタモール",
        "message": "✨ アミド化成功！解熱鎮痛薬「パラセタモール」が完成した！",
        "byproducts": ["CC(=O)O"], # 酢酸
        "reaction_type": "amidation"
    },
    # --------------------------------------------------------
    # Tier 2: クマリンルート
    # --------------------------------------------------------
    get_reaction_key("Oc1ccccc1", "C(Cl)(Cl)Cl", "NaOH"): {
        "product": "O=Cc1ccccc1O",
        "product_name": "サリチルアルデヒド",
        "message": "✨ Reimer-Tiemann反応成功！フェノールのオルト位がホルミル化された。",
        "byproducts": ["NaCl"],
        "reaction_type": "reimer_tiemann"
    },
    get_reaction_key("O=Cc1ccccc1O", "CC(=O)OC(C)=O", "NaOH"): {
        "product": "O=C1C=Cc2ccccc2O1",
        "product_name": "🌸 クマリン",
        "message": "✨ Perkin反応とラクトン化が連続進行！甘い香りのクマリンが完成した！",
        "byproducts": ["H2O"],
        "reaction_type": "perkin_condensation"
    },
    # Base触媒の表記ゆれ用
    get_reaction_key("O=Cc1ccccc1O", "CC(=O)OC(C)=O", "Base"): {
        "product": "O=C1C=Cc2ccccc2O1",
        "product_name": "🌸 クマリン",
        "message": "✨ Perkin反応とラクトン化が連続進行！甘い香りのクマリンが完成した！",
        "byproducts": ["H2O"],
        "reaction_type": "perkin_condensation"
    },
    # --------------------------------------------------------
    # Tier 2: リドカインルート
    # --------------------------------------------------------
    get_reaction_key("Cc1cccc(C)c1N", "ClCC(=O)Cl", "None"): {
        "product": "Cc1cccc(C)c1NC(=O)CCl",
        "product_name": "クロロアセトアミド中間体",
        "message": "✨ アミド結合形成！リドカイン合成の第一段階をクリアした。",
        "byproducts": ["HCl"],
        "reaction_type": "amidation"
    },
    get_reaction_key("Cc1cccc(C)c1NC(=O)CCl", "CCNCC", "None"): {
        "product": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
        "product_name": "💊 リドカイン",
        "message": "✨ S_N2反応成功！優れた局所麻酔薬「リドカイン」が完成した！",
        "byproducts": ["HCl"],
        "reaction_type": "sn2_substitution"
    },
    # --------------------------------------------------------
    # Tier 2: フルオレセインルート
    # --------------------------------------------------------
    get_reaction_key("O=C1OC(=O)c2ccccc12", "Oc1cc(O)ccc1", "ZnCl2"): {
        "product": "O=C1OC2(c3ccccc13)c4ccc(O)cc4Oc5cc(O)ccc25",
        "product_name": "☢️ フルオレセイン",
        "message": "✨ Friedel-Crafts型縮合成功！ブラックライトで鮮やかに光る「フルオレセイン」が完成した！",
        "byproducts": ["H2O"],
        "reaction_type": "friedel_crafts_condensation"
    },
    # --------------------------------------------------------
    # Tier 3: イブプロフェンルート
    # --------------------------------------------------------
    get_reaction_key("CC(C)Cc1ccccc1", "CC(=O)Cl", "AlCl3"): {
        "product": "CC(C)Cc1ccc(C(C)=O)cc1",
        "product_name": "p-イソブチルアセトフェノン",
        "message": "✨ Friedel-Craftsアシル化成功！イブプロフェン合成の骨格ができた。",
        "byproducts": ["HCl"],
        "reaction_type": "friedel_crafts_acylation"
    },
    get_reaction_key("CC(C)Cc1ccc(C(C)=O)cc1", "[C-]#[O+]", "Pd"): {
        "product": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "product_name": "💊 イブプロフェン",
        "message": "✨ BHCプロセスによるアトムエコノミーの高い反応が進行！鎮痛薬「イブプロフェン」が完成した！",
        "byproducts": [],
        "reaction_type": "catalytic_carbonylation"
    },
    # --------------------------------------------------------
    # Tier 3: サッカリンルート
    # --------------------------------------------------------
    get_reaction_key("Cc1ccccc1", "O=S(=O)(Cl)O", "None"): {
        "product": "Cc1ccccc1S(=O)(=O)Cl",
        "product_name": "o-トルエンスルホニルクロリド",
        "message": "✨ クロロスルホン化進行。オルト置換体を抽出した。",
        "byproducts": ["H2O"],
        "reaction_type": "chlorosulfonation"
    },
    get_reaction_key("Cc1ccccc1S(=O)(=O)Cl", "N", "None"): {
        "product": "Cc1ccccc1S(N)(=O)=O",
        "product_name": "o-トルエンスルホンアミド",
        "message": "✨ アミド化成功。閉環への準備が整った。",
        "byproducts": ["HCl"],
        "reaction_type": "amidation"
    },
    get_reaction_key("Cc1ccccc1S(N)(=O)=O", "O=[Mn](=O)(=O)[O-]", "Acid"): {
        "product": "O=C1NS(=O)(=O)c2ccccc12",
        "product_name": "🍬 サッカリン",
        "message": "✨ メチル基の酸化と分子内閉環が連続進行！人工甘味料「サッカリン」が完成した！",
        "byproducts": ["H2O"],
        "reaction_type": "oxidation_cyclization"
    },
    # Acid触媒の表記ゆれ用
    get_reaction_key("Cc1ccccc1S(N)(=O)=O", "O=[Mn](=O)(=O)[O-]", "Acid (酸)"): {
        "product": "O=C1NS(=O)(=O)c2ccccc12",
        "product_name": "🍬 サッカリン",
        "message": "✨ メチル基の酸化と分子内閉環が連続進行！人工甘味料「サッカリン」が完成した！",
        "byproducts": ["H2O"],
        "reaction_type": "oxidation_cyclization"
    },
    # --------------------------------------------------------
    # Tier 4: ナイロン6,6
    # --------------------------------------------------------
    get_reaction_key("O=C(Cl)CCCCC(=O)Cl", "NCCCCCCN", "None"): {
        "product": "O=C(CCCCC(=O)NCCCCCCN)NCCCCCCN",
        "product_name": "🧵 ナイロン6,6",
        "message": "✨ 重縮合反応成功！単一の分子から「物質(マテリアル)」へのパラダイムシフト、「ナイロン6,6」が完成した！",
        "byproducts": ["HCl"],
        "reaction_type": "interfacial_polycondensation"
    },
    # --------------------------------------------------------
    # Tier 4: キュバンルート
    # --------------------------------------------------------
    get_reaction_key("O=C1C=CC=C1", "O=C1C=CC=C1", "UV Lamp"): {
        "product": "O=C1C2C3C4C1C5C2C3C45",
        "product_name": "ケージドジオン中間体",
        "message": "✨ [2+2]付加環化成功！ひずみの大きいケージ骨格が形成された。",
        "byproducts": [],
        "reaction_type": "photochemical_cycloaddition"
    },
        "product": "C12C3C4C1C5C2C3C45",
        "product_name": "🧊 キュバン",
        "message": "✨ Favorskii転位と脱炭酸が完了！プラトンの立体、究極のひずみ分子「キュバン」が完成した！",
        "byproducts": ["O=C=O"],
        "reaction_type": "favorskii_rearrangement"
    },
    # --------------------------------------------------------
    # Tier 5: ナノプシャン (Final)
    # --------------------------------------------------------
    get_reaction_key("c1ccccc1I", "C#Cc1ccc(C#C)cc1", "Pd/Cu"): {
        "product": "c1ccccc1C#Cc2ccc(C#C)cc2",
        "product_name": "ナノプシャン上半身",
        "message": "✨ Sonogashiraカップリング成功！頭部と胴体が結合し、ナノプシャンの上半身が完成した。",
        "byproducts": ["HI"],
        "reaction_type": "sonogashira_coupling"
    },
    get_reaction_key("Ic1cc(I)ccc1", "CCC(CC)C#C", "Pd/Cu"): {
        "product": "CCC(CC)C#Cc1cc(I)ccc1",
        "product_name": "ナノプシャン下半身",
        "message": "✨ Sonogashiraカップリング成功！骨盤と脚部が結合し、ナノプシャンの下半身が完成した。",
        "byproducts": ["HI"],
        "reaction_type": "sonogashira_coupling"
    },
    get_reaction_key("c1ccccc1C#Cc2ccc(C#C)cc2", "CCC(CC)C#Cc1cc(I)ccc1", "Pd/Cu"): {
        "product": "c1ccccc1C#Cc2ccc(C#Cc3cc(C#CC(CC)CC)ccc3)cc2",
        "product_name": "🕺 ナノプシャン",
        "message": "✨ 究極のクロスカップリングが完了！！分子サイズの小人「ナノプシャン」が誕生した！",
        "byproducts": ["HI"],
        "reaction_type": "sonogashira_coupling"
    }
}


@app.post("/react", response_model=ReactResponse)
async def react(req: ReactRequest):
    """
    化学反応エンドポイント (Phase 0 + Phase 1 統合)。

    【Phase 0: ラジカルハロゲン化】
      R-H + X₂ --hν→ R-X + HX
      - reagent_1: アルカン (例: "C", "CC")
      - reagent_2: ハロゲン分子 (例: "ClCl", "BrBr")
      - condition_light: true

    【Phase 1: 置換/脱離反応】
      - reagent_1: ハロアルカン (例: "CCCl", "CC(Cl)C")
      - reagent_2: 求核剤/塩基 (例: "O", "[O-]")
      - temperature: 0-100 (60 以上 → E2 優先, 60 未満 → SN2 候補)
      - solvent_type: "protic" or "aprotic" (aprotic + 低温 → SN2)
    """
    # ==================================================================
    # カスタム触媒レシピ (新機能)
    # ==================================================================
    import copy
    r1, r2 = req.reagent_1, req.reagent_2
    cat = req.catalyst or ""
    reagents = {r1, r2}
    
    custom_res = None
    if reagents == {"C", "C"} and cat.startswith("Pt"):
        custom_res = (["C#C"], ["H2"], "✨ メタンの脱水素カップリングによりアセチレンを合成しました！", "catalytic_coupling", "アセチレン", "N/A")
    elif reagents == {"C#C", "C#C"} and cat.startswith("Fe"):
        custom_res = (["c1ccccc1"], [], "✨ アセチレンの三量化環化によりベンゼン環が生成されました！", "cyclotrimerization", "ベンゼン", "N/A")
    elif reagents == {"NITROBENZENE", "HCl"} and cat.startswith("Sn"):
        custom_res = (["ANILINE"], ["H2O"], "✨ 塩化水素とスズを利用した還元反応によりアニリンが合成されました！", "reduction_nitro", "アニリン", "N/A")
    elif reagents == {"CC", "HCl"}:
        custom_res = (["CCCl"], [], "✨ エタンと塩化水素からクロロエタンを合成しました！", "hydrohalogenation", "クロロエタン", "primary")
    elif reagents == {"CC(O)C", "O=O"} and cat.startswith("Pd"):
        custom_res = (["CC(=O)C"], ["H2O"], "✨ 2-プロパノールの触媒的酸化によりアセトンが合成されました！", "oxidation", "アセトン", "secondary")
    elif (reagents == {"C", "N"} or reagents == {"C", "[NH3]"}):
        custom_res = (["CNC"], [], "✨ メタンとアンモニアからジメチルアミンを合成しました！", "amination", "ジメチルアミン", "secondary")
    elif reagents == {"CNC", "O=CO"} and cat.startswith("Acid"):
        custom_res = (["O=CN(C)C"], ["H2O"], "✨ 酸触媒を用いたアミド化反応によりDMF（ジメチルホルムアミド）が生成されました！", "amidation", "DMF", "N/A")

    if custom_res:
        prs_smi, byprs, msg, rxn_type, pname, pclass = custom_res
        prs = [ProductInfo(smiles=s, name=pname, carbon_class=pclass, reaction_type=rxn_type) for s in prs_smi]
        return ReactResponse(
            status=ResultStatus.SUCCESS,
            message=msg,
            reagent_1_smiles=r1,
            reagent_2_smiles=r2,
            reaction_type=rxn_type,
            products=prs,
            byproducts=byprs,
            condition_summary=f"Catalyst: {cat}" if cat else "No specific catalyst",
            tier="tier1_rule"
        )

    # ==================================================================
    # Open World Reaction Registry (Tier 1.5)
    # ==================================================================
    target_registry_key = get_reaction_key(r1, r2, cat)
    recipe = OPEN_WORLD_REACTIONS.get(target_registry_key)
    
    # 触媒の表記ゆれ吸収 (Acid や Sn の部分一致等)
    if not recipe:
        sorted_r = sorted([r1, r2])
        for (k1, k2, kcat), v in OPEN_WORLD_REACTIONS.items():
            if k1 == sorted_r[0] and k2 == sorted_r[1]:
                # 触媒がNone指定なら、リクエストもNoneでなければならない
                if kcat == "None" and (not cat or cat == "None"):
                    recipe = v; break
                # 触媒が指定されている場合、部分一致を許容する
                elif kcat != "None" and cat and (kcat in cat or cat in kcat or cat.startswith(kcat.split('(')[0].strip())):
                    recipe = v; break

    if recipe:
        return ReactResponse(
            status=ResultStatus.SUCCESS,
            message=recipe["message"],
            reagent_1_smiles=r1,
            reagent_2_smiles=r2,
            reaction_type=recipe.get("reaction_type", "open_world_recipe"),
            products=[ProductInfo(smiles=recipe["product"], name=recipe["product_name"])],
            byproducts=recipe.get("byproducts", []),
            condition_summary=f"Catalyst: {cat or 'None'}",
            tier="tier1_registry",
            rarity="SSR" if "💊" in recipe["product_name"] else "SR",
            flavor_text=recipe.get("message", "")
        )

    # ==================================================================
    # 反応タイプの自動判定
    # ==================================================================
    rxn_type = detect_reaction_type(
        req.reagent_1, req.reagent_2,
        req.condition_light, req.temperature, req.solvent_type,
        req.catalyst == "AlCl3" if req.catalyst else False,
    )

    # ==================================================================
    # Phase 0: ラジカルハロゲン化
    # ==================================================================
    if rxn_type == "radical_halogenation":
        return _handle_radical_halogenation(req)

    # ==================================================================
    # Phase 1: SN2 / E2
    # ==================================================================
    if rxn_type == "substitution_elimination":
        return _handle_substitution_elimination(req)

    # ==================================================================
    # Phase 2: グリニャール試薬
    # ==================================================================
    if rxn_type == "grignard_preparation":
        return _handle_grignard_preparation(req)

    if rxn_type == "grignard_reaction":
        return _handle_grignard_reaction(req)

    # ==================================================================
    # Phase 3: 芳香族求電子置換 (EAS)
    # ==================================================================
    if rxn_type == "eas_acylation":
        return _handle_eas_acylation(req)

    # ==================================================================
    # Phase 4: ナノプシャン組立
    # ==================================================================
    if rxn_type == "nanoputian_assembly":
        return _handle_nanoputian_assembly(req)

    # ==================================================================
    # Phase 5: インディゴルート反応 (トークンベース)
    # ==================================================================
    if rxn_type in ("nitration", "reduction_nitro", "vilsmeier_haack",
                     "diazotization", "aldol_condensation", "oxidative_dimerization"):
        return _handle_indigo_reactions(req, rxn_type)

    # ==================================================================
    # Tier 1 で未定義 → Tier 2 AI 推論フォールバック
    # ==================================================================
    logger.info(
        f"Tier 1 未対応: {req.reagent_1} + {req.reagent_2} → Tier 2 AI へフォールバック"
    )
    return await _handle_tier2_ai_fallback(req)


# ---------------------------------------------------------------------------
# Phase 0 ハンドラー
# ---------------------------------------------------------------------------

def _handle_radical_halogenation(req: ReactRequest) -> ReactResponse:
    """Phase 0: ラジカルハロゲン化の処理。"""

    # 光条件チェック
    if not req.condition_light:
        return ReactResponse(
            status=ResultStatus.NO_REACTION,
            message="💡 光 (hν) がありません！ラジカル反応には光エネルギーが必要です。",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="radical_halogenation",
            condition_summary="光なし → 反応不可",
        )

    # 禁止ハロゲンチェック
    if req.reagent_2 in FORBIDDEN_HALOGENS:
        status = (ResultStatus.GAME_OVER if req.reagent_2 == "FF"
                  else ResultStatus.NO_REACTION)
        return ReactResponse(
            status=status,
            message=FORBIDDEN_HALOGENS[req.reagent_2],
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="radical_halogenation",
            products=([ProductInfo(smiles="GAME_OVER_EXPLOSION",
                                   name="💥", carbon_class="N/A",
                                   reaction_type="explosion")]
                      if req.reagent_2 == "FF" else []),
        )

    # アルカンのパース
    mol = parse_alkane_smiles(req.reagent_1)
    if mol is None:
        return ReactResponse(
            status=ResultStatus.NO_REACTION,
            message=f"⚠️ '{req.reagent_1}' は有効なアルカンではありません。"
                    f"例: C (メタン), CC (エタン), CCC (プロパン)",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="radical_halogenation",
        )

    # 反応実行
    halogen = HALOGEN_MAP[req.reagent_2]
    raw_products = radical_halogenation(mol, halogen)

    if not raw_products:
        return ReactResponse(
            status=ResultStatus.NO_REACTION,
            message="🔬 水素が残っていないため反応が進行しませんでした。",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="radical_halogenation",
        )

    hx_display = {"Cl": "HCl", "Br": "HBr", "F": "HF", "I": "HI"}
    product_infos = [
        ProductInfo(smiles=p["smiles"], name=p["name"],
                    carbon_class=p["carbon_class"],
                    reaction_type="radical")
        for p in raw_products
    ]

    return ReactResponse(
        status=ResultStatus.SUCCESS,
        message=f"✨ ラジカルハロゲン化成功！{len(product_infos)} 種類の位置異性体が生成されました。",
        reagent_1_smiles=req.reagent_1,
        reagent_2_smiles=req.reagent_2,
        reaction_type="radical_halogenation",
        products=product_infos,
        byproducts=[hx_display.get(halogen, f"H{halogen}")],
        hx_byproduct=hx_display.get(halogen, f"H{halogen}"),
        condition_summary=f"hν (光) + {req.reagent_2}",
    )


# ---------------------------------------------------------------------------
# Phase 1 ハンドラー
# ---------------------------------------------------------------------------

def _handle_substitution_elimination(req: ReactRequest) -> ReactResponse:
    """Phase 1: SN2/E2 反応の処理。"""

    nuc_info = NUCLEOPHILE_BASE_MAP[req.reagent_2]
    mol = parse_haloalkane_smiles(req.reagent_1)

    if mol is None:
        return ReactResponse(
            status=ResultStatus.NO_REACTION,
            message=f"⚠️ '{req.reagent_1}' は有効なハロアルカンではありません。"
                    f"例: CCCl (クロロエタン), CC(Cl)C (2-クロロプロパン)",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="substitution_elimination",
        )

    # デフォルト値
    temp = req.temperature if req.temperature is not None else 25
    solvent = req.solvent_type if req.solvent_type else "aprotic"

    # ------------------------------------------------------------------
    # 分岐ロジック: 温度 × 溶媒 → 反応タイプ決定
    # ------------------------------------------------------------------
    #
    #  ┌───────────────┬──────────────────┬──────────────────┐
    #  │               │  temp < 60       │  temp >= 60      │
    #  ├───────────────┼──────────────────┼──────────────────┤
    #  │ aprotic       │  SN2 優先        │  E2 優先         │
    #  │ (非プロトン性) │                  │                  │
    #  ├───────────────┼──────────────────┼──────────────────┤
    #  │ protic        │  SN1/SN2 混合    │  E2 優先         │
    #  │ (プロトン性)  │  → 簡易版: SN2   │                  │
    #  └───────────────┴──────────────────┴──────────────────┘

    if temp >= 60:
        # --- E2 脱離反応が優先 ---
        rxn_label = "E2"
        condition_desc = f"高温 ({temp}°C) → E2 脱離が優先"

        products = e2_elimination(mol)

        if not products:
            # β-水素がない → SN2 にフォールバック
            rxn_label = "SN2 (E2フォールバック)"
            condition_desc += " → β-水素なし、SN2 へフォールバック"
            products = sn2_substitution(mol, nuc_info)

        if not products:
            return ReactResponse(
                status=ResultStatus.NO_REACTION,
                message="🔬 反応条件は揃っていますが、適切な基質構造がなく反応が進行しませんでした。"
                        f"(第三級炭素では SN2 不可、β-水素なしでは E2 不可)",
                reagent_1_smiles=req.reagent_1,
                reagent_2_smiles=req.reagent_2,
                reaction_type=rxn_label,
                condition_summary=condition_desc,
            )

        # 脱離するハロゲンから HX を特定
        leaving = products[0].get("leaving_group", "X")
        hx = f"H{leaving}"

        product_infos = [
            ProductInfo(
                smiles=p["smiles"], name=p["name"],
                carbon_class=p["carbon_class"],
                reaction_type=p.get("reaction_type", rxn_label),
            )
            for p in products
        ]

        byproducts_list = [hx]
        if products[0].get("reaction_type") == "E2":
            byproducts_list.append(f"{nuc_info['name']}+H (塩基の共役酸)")

        return ReactResponse(
            status=ResultStatus.SUCCESS,
            message=f"🔥 {rxn_label} 反応成功！{len(product_infos)} 種類の生成物。"
                    f"高温 ({temp}°C) により脱離が優先されました。",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type=rxn_label,
            products=product_infos,
            byproducts=byproducts_list,
            hx_byproduct=hx,
            condition_summary=condition_desc,
        )

    else:
        # --- 低温: SN2 置換反応が候補 ---
        if solvent == "aprotic":
            # 非プロトン性極性溶媒 + 低温 → SN2 優先
            rxn_label = "SN2"
            condition_desc = f"低温 ({temp}°C) + 非プロトン性溶媒 → SN2 置換が優先"

            products = sn2_substitution(mol, nuc_info)

            if not products:
                return ReactResponse(
                    status=ResultStatus.NO_REACTION,
                    message="⚠️ SN2 反応が進行しませんでした。"
                            "第三級炭素のハロアルカンでは立体障害により SN2 は不可能です。"
                            "温度を上げて E2 脱離を試してみてください。",
                    reagent_1_smiles=req.reagent_1,
                    reagent_2_smiles=req.reagent_2,
                    reaction_type=rxn_label,
                    condition_summary=condition_desc,
                )

            leaving = products[0].get("leaving_group", "X")
            product_infos = [
                ProductInfo(
                    smiles=p["smiles"], name=p["name"],
                    carbon_class=p["carbon_class"],
                    reaction_type="SN2",
                )
                for p in products
            ]

            return ReactResponse(
                status=ResultStatus.SUCCESS,
                message=f"🧪 SN2 置換反応成功！求核剤がハロゲンを置換しました。"
                        f"生成物: {len(product_infos)} 種類。",
                reagent_1_smiles=req.reagent_1,
                reagent_2_smiles=req.reagent_2,
                reaction_type=rxn_label,
                products=product_infos,
                byproducts=[f"{leaving}⁻ (脱離したハロゲン化物イオン)"],
                condition_summary=condition_desc,
            )

        else:
            # プロトン性溶媒 + 低温 → 反応は遅い / 混合物
            rxn_label = "SN2 (低効率)"
            condition_desc = (
                f"低温 ({temp}°C) + プロトン性溶媒 → "
                "求核剤が溶媒和されて求核性が低下。反応効率が悪化。"
            )

            products = sn2_substitution(mol, nuc_info)

            if not products:
                return ReactResponse(
                    status=ResultStatus.NO_REACTION,
                    message="⚠️ プロトン性溶媒中では求核剤の反応性が低下し、"
                            "反応が進行しませんでした。非プロトン性溶媒 (aprotic) を使ってみてください。",
                    reagent_1_smiles=req.reagent_1,
                    reagent_2_smiles=req.reagent_2,
                    reaction_type=rxn_label,
                    condition_summary=condition_desc,
                )

            product_infos = [
                ProductInfo(
                    smiles=p["smiles"],
                    name=p["name"] + " ⚠️低収率",
                    carbon_class=p["carbon_class"],
                    reaction_type="SN2 (低効率)",
                )
                for p in products
            ]

            return ReactResponse(
                status=ResultStatus.SUCCESS,
                message=f"🧪 SN2 反応が一応進行しましたが、プロトン性溶媒のため収率は低いです。"
                        f"非プロトン性極性溶媒 (aprotic) に切り替えると効率が上がります！",
                reagent_1_smiles=req.reagent_1,
                reagent_2_smiles=req.reagent_2,
                reaction_type=rxn_label,
                products=product_infos,
                byproducts=[f"{products[0].get('leaving_group', 'X')}⁻ (低効率)"],
                condition_summary=condition_desc,
            )


# =====================================================================
#  SECTION 8:  Phase 2 – グリニャール試薬
# =====================================================================

def _handle_grignard_preparation(req: ReactRequest) -> ReactResponse:
    """Phase 2: グリニャール試薬の調製 (R-X + Mg → R-Mg-X)"""
    solvent = req.solvent_type if req.solvent_type else "aprotic"

    # プロトン性溶媒ではグリニャール試薬が分解する
    if solvent == "protic":
        return ReactResponse(
            status=ResultStatus.GAME_OVER,
            message=(
                "💥 グリニャール試薬がプロトン性溶媒 (水, アルコール等) と激しく反応！\n"
                "炭素上のδ-がプロトンと酸塩基反応を起こし、元のアルカンに戻ってしまいました。\n"
                "⚠️ 必ず乾燥した非プロトン性溶媒 (エーテル等) を使用してください。"
            ),
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="grignard_preparation_failure",
            products=[
                ProductInfo(
                    smiles=TAR_SMILES,
                    name=TAR_NAME,
                    carbon_class="N/A",
                    reaction_type="decomposition",
                )
            ],
            condition_summary="プロトン性溶媒 → グリニャール試薬分解",
        )

    # ハロアルカンをパースして R 基を特定
    mol = parse_haloalkane_smiles(req.reagent_1)
    if mol is None:
        return ReactResponse(
            status=ResultStatus.NO_REACTION,
            message=f"⚠️ '{req.reagent_1}' は有効なハロアルカンではありません。グリニャール試薬にはC-X結合が必要です。",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="grignard_preparation",
        )

    # ハロゲンを特定
    halogen = None
    for c in mol.carbons:
        if c["halogens"]:
            halogen = c["halogens"][0]
            break
    if halogen is None:
        return ReactResponse(
            status=ResultStatus.NO_REACTION,
            message="⚠️ ハロゲンが見つかりません。",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="grignard_preparation",
        )

    # グリニャール試薬の SMILES を生成: [CHn][CHn]...[Mg][X]
    n_carbons = mol.carbon_count
    ch_groups = []
    for i in range(n_carbons):
        c = mol.carbons[i]
        # ハロゲンを除いた水素数 (C-X が C-Mg-X になるので H は増えない)
        h_on_c = c["h_count"]
        if h_on_c == 0:
            ch_groups.append("[C]")
        elif h_on_c == 1:
            ch_groups.append("[CH]")
        elif h_on_c == 2:
            ch_groups.append("[CH2]")
        else:
            ch_groups.append("[CH3]")

    grignard_smiles = "".join(ch_groups) + f"[Mg][{halogen}]"

    c_count = mol.carbon_count
    base_name = CARBON_PREFIX.get(c_count, f"C{c_count}")
    product_name = f"{base_name}マグネシウム{HALOGEN_JP.get(halogen, halogen)}化物 (グリニャール試薬)"

    return ReactResponse(
        status=ResultStatus.SUCCESS,
        message=(
            f"✨ グリニャール試薬調製成功！\n"
            f"R-X + Mg → R-Mg-X\n"
            f"非常に強力な求核剤/塩基ができました。水に触れさせないよう注意！"
        ),
        reagent_1_smiles=req.reagent_1,
        reagent_2_smiles=req.reagent_2,
        reaction_type="grignard_preparation",
        products=[
            ProductInfo(
                smiles=grignard_smiles,
                name=product_name,
                carbon_class="grignard",
                reaction_type="grignard_preparation",
            )
        ],
        condition_summary=f"R-X + Mg → R-Mg-X (非プロトン性溶媒中)",
    )


def _handle_grignard_reaction(req: ReactRequest) -> ReactResponse:
    """Phase 2: グリニャール反応 (R-Mg-X + 求電子剤 → 生成物)"""
    grignard = _parse_grignard(req.reagent_1)
    if grignard is None:
        return ReactResponse(
            status=ResultStatus.NO_REACTION,
            message=f"⚠️ '{req.reagent_1}' は有効なグリニャール試薬ではありません。",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="grignard_reaction",
        )

    r2 = req.reagent_2
    n_c = grignard["carbon_count"]
    r_group = grignard["r_group"]

    # プロトン性溶媒チェック
    solvent = req.solvent_type if req.solvent_type else "aprotic"
    if solvent == "protic":
        return ReactResponse(
            status=ResultStatus.GAME_OVER,
            message="💥 グリニャール試薬がプロトン性溶媒と反応して分解しました！",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="grignard_decomposition",
            products=[ProductInfo(smiles=TAR_SMILES, name=TAR_NAME, carbon_class="N/A", reaction_type="decomposition")],
        )

    # --- グリニャール + 酸素 (O=O) → アルコール ---
    if r2 == "O=O":
        # R-Mg-X + 1/2 O₂ → R-OH
        product_smiles = r_group + "O"  # CO, CCO, etc.
        c_count = n_c
        base_name = CARBON_PREFIX.get(c_count, f"C{c_count}")
        product_name = f"{base_name}オール (グリニャール酸化)"
        return ReactResponse(
            status=ResultStatus.SUCCESS,
            message=f"✨ グリニャール試薬の酸化成功！\nR-Mg-X + O₂ → R-O-Mg-X → (加水分解) → R-OH\nアルコールが生成されました。",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="grignard_oxidation",
            products=[ProductInfo(smiles=product_smiles, name=product_name, carbon_class="primary", reaction_type="grignard_oxidation")],
            byproducts=[f"Mg({grignard['halogen']})OH (マグネシウム塩)"],
            condition_summary="R-Mg-X + O₂ → R-OH (酸化)",
        )

    # --- グリニャール + ホルムアルデヒド (C=O) → 1級アルコール (炭素鎖延長) ---
    if r2 == "C=O":
        # R-Mg-X + HCHO → R-CH₂-OH (after hydrolysis)
        product_smiles = r_group + "CO"  # CCO, CCCO, etc.
        c_count = n_c + 1
        base_name = CARBON_PREFIX.get(c_count, f"C{c_count}")
        product_name = f"{base_name}オール (求核付加)"
        return ReactResponse(
            status=ResultStatus.SUCCESS,
            message=f"✨ 求核付加反応成功！\nR-Mg-X + HCHO → R-CH₂-OH\n炭素鎖が1つ延長されたアルコールが生成されました。",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="grignard_nucleophilic_addition",
            products=[ProductInfo(smiles=product_smiles, name=product_name, carbon_class="primary", reaction_type="nucleophilic_addition")],
            byproducts=[f"Mg({grignard['halogen']})OH"],
            condition_summary="R-Mg-X + HCHO → R-CH₂-OH (求核付加)",
        )

    # 未対応の求電子剤
    return ReactResponse(
        status=ResultStatus.NO_REACTION,
        message=f"⚠️ グリニャール試薬と '{r2}' の反応はサポートされていません。",
        reagent_1_smiles=req.reagent_1,
        reagent_2_smiles=req.reagent_2,
        reaction_type="grignard_reaction",
    )


# =====================================================================
#  SECTION 9:  Phase 3 – 芳香族求電子置換 (EAS)
# =====================================================================

def _handle_eas_acylation(req: ReactRequest) -> ReactResponse:
    """Phase 3: Friedel-Crafts アシル化 (ArH + RCOCl + AlCl₃ → ArCOR + HCl)"""
    catalyst = getattr(req, 'catalyst_AlCl3', False)
    if not catalyst:
        return ReactResponse(
            status=ResultStatus.NO_REACTION,
            message="⚠️ Friedel-Crafts 反応には AlCl₃ 触媒が必要です！触媒を ON にしてください。",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="eas_acylation",
        )

    # ベンゼン + 塩化アセチル → アセトフェノン
    ar = req.reagent_1 if "c1ccccc1" in req.reagent_1 else req.reagent_2
    acyl = req.reagent_2 if req.reagent_2 != ar else req.reagent_1

    if acyl == "CC(=O)Cl":
        product_smiles = "CC(=O)c1ccccc1"
        product_name = "アセトフェノン (Friedel-Crafts アシル化)"
    else:
        product_smiles = f"{acyl.replace('Cl', '')}c1ccccc1"
        product_name = f"芳香族ケトン (EAS)"

    return ReactResponse(
        status=ResultStatus.SUCCESS,
        message=(
            f"✨ Friedel-Crafts アシル化成功！\n"
            f"ArH + RCOCl + AlCl₃ → ArCOR + HCl\n"
            f"芳香環にアシル基が導入されました。"
        ),
        reagent_1_smiles=req.reagent_1,
        reagent_2_smiles=req.reagent_2,
        reaction_type="eas_acylation",
        products=[
            ProductInfo(
                smiles=product_smiles,
                name=product_name,
                carbon_class="aromatic",
                reaction_type="eas_acylation",
            )
        ],
        byproducts=["HCl (塩化水素)"],
        condition_summary="ArH + RCOCl + AlCl₃ → ArCOR + HCl",
    )


# =====================================================================
#  SECTION 10: Phase 4 – ナノプシャン組立
# =====================================================================

def _handle_nanoputian_assembly(req: ReactRequest) -> ReactResponse:
    """Phase 4: ナノプシャン組立 (NANO_HEAD + NANO_BODY → NANOPUTIAN)"""
    return ReactResponse(
        status=ResultStatus.SUCCESS,
        message=(
            "🎉 ナノプシャン全合成達成！！！\n\n"
            "君がパズルで組み上げてきたすべての反応 —\n"
            "ラジカルハロゲン化、SN2置換、E2脱離、グリニャール試薬、求核付加、Friedel-Craftsアシル化 —\n"
            "これらの知識を総動員して、ついにナノメートルサイズの人型分子が完成しました！\n\n"
            "おめでとう！君は立派な有機化学者だ！"
        ),
        reagent_1_smiles=req.reagent_1,
        reagent_2_smiles=req.reagent_2,
        reaction_type="nanoputian_total_synthesis",
        products=[
            ProductInfo(
                smiles="NANOPUTIAN",
                name="🧬 ナノプシャン (NanoPutian)",
                carbon_class="total_synthesis",
                reaction_type="total_synthesis",
            )
        ],
        condition_summary="NANO_HEAD + NANO_BODY → NANOPUTIAN (全合成)",
    )


# =====================================================================
#  SECTION 10b: Phase 5 – インディゴルート反応 (トークンベース)
# =====================================================================

# インディゴルート反応の定義マップ
INDIGO_REACTIONS: dict[str, dict] = {
    "nitration": {
        "product": "NITROBENZENE",
        "product_name": "ニトロベンゼン (Nitrobenzene)",
        "byproducts": ["H2O"],
        "message": (
            "⚡ ニトロ化成功！\n\n"
            "硫酸 (H₂SO₄) が触媒として NO₂⁺ イオンを生成し、\n"
            "ベンゼン環の π 電子がこれを攻撃する。\n"
            "これが「芳香族求電子置換反応 (EAS)」のニトロ化だ！\n\n"
            "🧪 生成物: ニトロベンゼン (淡黄色の液体、アーモンドの匂い)"
        ),
        "condition": "c1ccccc1 + HNO3 → C₆H₅NO₂ + H₂O (EAS ニトロ化)",
    },
    "reduction_nitro": {
        "product": "ANILINE",
        "product_name": "アニリン (Aniline)",
        "byproducts": ["SnCl2", "H2O"],
        "message": (
            "🔄 還元成功！\n\n"
            "スズ (Sn) と塩酸 (HCl) がニトロ基 (-NO₂) を\n"
            "アミノ基 (-NH₂) に変換した。\n"
            "これは Béchamp 還元として知られる古典的な方法だ。\n\n"
            "🧪 生成物: アニリン (アミン特有の匂いを持つ液体)\n"
            "💡 ヒント: アミノ基は高い求核性と電子供与性を持つ重要な官能基"
        ),
        "condition": "C₆H₅NO₂ + Sn/HCl → C₆H₅NH₂ (Béchamp 還元)",
    },
    "vilsmeier_haack": {
        "product": "2_AMINOBENZALDEHYDE",
        "product_name": "2-アミノベンズアルデヒド (2-Aminobenzaldehyde)",
        "byproducts": ["POCl3_waste"],
        "message": (
            "✨ Vilsmeier-Haack 反応成功！\n\n"
            "DMF (ジメチルホルムアミド) と POCl₃ から\n"
            "反応性の高い「Vilsmeier 試薬」が生成され、\n"
            "アニリンのオルト位にアルデヒド基 (-CHO) を導入した。\n\n"
            "🧪 生成物: 2-アミノベンズアルデヒド\n"
            "💡 ヒント: -NH₂ 基がオルト/パラ位に電子を供与するため、\n"
            "    オルト位が選択的にホルミル化される"
        ),
        "condition": "C₆H₅NH₂ + DMF/POCl₃ → 2-NH₂-C₆H₄-CHO (Vilsmeier-Haack)",
    },
    "diazotization": {
        "product": "2_NITROBENZALDEHYDE",
        "product_name": "2-ニトロベンズアルデヒド (2-Nitrobenzaldehyde)",
        "byproducts": ["N2", "H2O"],
        "message": (
            "🌀 ジアゾ化 → Sandmeyer 型変換成功！\n\n"
            "亜硝酸ナトリウム (NaNO₂) でアミノ基をジアゾニウム塩に変換し、\n"
            "さらにニトロ基 (-NO₂) に変換した。\n"
            "ジアゾ化は窒素ガス (N₂) の放出を駆動力とする反応だ。\n\n"
            "🧪 生成物: 2-ニトロベンズアルデヒド (Baeyer-Drewson 法の基質)\n"
            "💡 これがインディゴ合成の核心的な出発原料！"
        ),
        "condition": "2-NH₂-C₆H₄-CHO + NaNO₂ → 2-NO₂-C₆H₄-CHO + N₂ (ジアゾ化 + Sandmeyer)",
    },
    "aldol_condensation": {
        "product": "INDOXYL",
        "product_name": "インドキシル (Indoxyl)",
        "byproducts": ["H2O", "NaOH"],
        "message": (
            "🔮 Baeyer-Drewson 反応成功！\n\n"
            "2-ニトロベンズアルデヒドとアセトンが\n"
            "NaOH の塩基条件下でアルドール縮合を起こし、\n"
            "同時にニトロ基の還元と環化が進行して\n"
            "複素環「インドール骨格」が一気に構築された！\n\n"
            "🧪 生成物: インドキシル (3-ヒドロキシインドール)\n"
            "💡 これが酸化されるとインディゴになる最後の中間体"
        ),
        "condition": "2-NO₂-C₆H₄-CHO + CH₃COCH₃ + NaOH → Indoxyl (Baeyer-Drewson)",
    },
    "oxidative_dimerization": {
        "product": "INDIGO",
        "product_name": "🔵 インディゴ (Indigo)",
        "byproducts": ["H2O"],
        "message": (
            "🎉🔵 インディゴ合成達成！！！\n\n"
            "インドキシルが空気中の酸素 (O₂) で酸化され、\n"
            "2分子が C=C 二重結合で連結して「インディゴ」が完成！\n\n"
            "✨ 巨大な共役系 (交互に並ぶ単結合と二重結合) が\n"
            "   可視光の赤色成分を吸収し、美しい藍色を発色する。\n\n"
            "🧪 この染料は5000年以上の歴史を持ち、\n"
            "   ジーンズの「デニムブルー」の正体はインディゴ染料だ！\n\n"
            "おめでとう！君はインディゴの全合成に成功した！"
        ),
        "condition": "2 × Indoxyl + O₂ → Indigo + H₂O (酸化二量化)",
    },
}


def _handle_indigo_reactions(req: ReactRequest, rxn_type: str) -> ReactResponse:
    """Phase 5: インディゴルート反応 (トークンベースの特殊反応)"""
    rxn_info = INDIGO_REACTIONS.get(rxn_type)
    if rxn_info is None:
        return ReactResponse(
            status=ResultStatus.NO_REACTION,
            message="❌ 未定義のインディゴルート反応です。",
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type=rxn_type,
        )

    return ReactResponse(
        status=ResultStatus.SUCCESS,
        message=rxn_info["message"],
        reagent_1_smiles=req.reagent_1,
        reagent_2_smiles=req.reagent_2,
        reaction_type=rxn_type,
        products=[
            ProductInfo(
                smiles=rxn_info["product"],
                name=rxn_info["product_name"],
                carbon_class="indigo_route",
                reaction_type=rxn_type,
            )
        ],
        byproducts=rxn_info.get("byproducts", []),
        condition_summary=rxn_info["condition"],
        tier="tier1_rule",
    )


# =====================================================================
#  SECTION 11: Tier 2 – AI 推論フォールバック (Hugging Face Inference API)
# =====================================================================

# --- 設定 ---
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# 使用する HF モデル (反応予測用)
# - 本番: "ncf/rxn-forward-prediction" (Molecular Transformer ベース)
# - 代替: 任意の text2text-generation モデル
HF_MODEL_ID: str = os.environ.get(
    "HF_MODEL_ID",
    "ncf/rxn-forward-prediction",
)
HF_API_URL: str = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_TIMEOUT_SECONDS: float = 15.0

# ゲームオーバー用ダミー生成物 (タール / 炭化物)
TAR_SMILES: str = "C1=CC=CC=C1"  # ベンゼン (タールの簡略表現)
TAR_NAME: str = "🏭 タール / 炭化物 (Carbonized Residue)"


async def _call_hf_inference(reaction_smiles: str) -> dict:
    """
    Hugging Face Inference API に反応 SMILES を送信し、
    予測生成物を取得する。

    Args:
        reaction_smiles: "reactant1.reactant2>>" 形式の入力

    Returns:
        dict with keys:
            - "success": bool
            - "predicted_smiles": str (予測された生成物 SMILES)
            - "raw_output": any (API の生レスポンス)
            - "latency_seconds": float
            - "error": Optional[str]
    """
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": reaction_smiles,
        "parameters": {
            "max_new_tokens": 256,
            "top_k": 5,          # Top-K 推論で「意図しない副産物」を生成
            "temperature": 0.8,  # 多様性を上げる
        },
    }

    start_time = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                HF_API_URL,
                json=payload,
                headers=headers,
            )

        elapsed = time.perf_counter() - start_time
        logger.info(f"HF API response: status={resp.status_code}, latency={elapsed:.2f}s")

        if resp.status_code != 200:
            error_detail = resp.text[:500]
            logger.warning(f"HF API error {resp.status_code}: {error_detail}")
            return {
                "success": False,
                "predicted_smiles": "",
                "raw_output": error_detail,
                "latency_seconds": elapsed,
                "error": f"HTTP {resp.status_code}: {error_detail}",
            }

        result = resp.json()

        # HF text2text-generation の出力形式を処理
        predicted = ""
        if isinstance(result, list) and len(result) > 0:
            # [{"generated_text": "..."}, ...]
            if isinstance(result[0], dict):
                predicted = result[0].get("generated_text", "").strip()
            elif isinstance(result[0], str):
                predicted = result[0].strip()
        elif isinstance(result, dict):
            predicted = result.get("generated_text", "").strip()
        elif isinstance(result, str):
            predicted = result.strip()

        # 空なら失敗扱い
        if not predicted:
            return {
                "success": False,
                "predicted_smiles": "",
                "raw_output": result,
                "latency_seconds": elapsed,
                "error": "AI モデルが空の出力を返しました",
            }

        return {
            "success": True,
            "predicted_smiles": predicted,
            "raw_output": result,
            "latency_seconds": elapsed,
            "error": None,
        }

    except httpx.TimeoutException:
        elapsed = time.perf_counter() - start_time
        logger.error(f"HF API timeout after {elapsed:.2f}s")
        return {
            "success": False,
            "predicted_smiles": "",
            "raw_output": None,
            "latency_seconds": elapsed,
            "error": f"タイムアウト ({HF_TIMEOUT_SECONDS}秒超過)",
        }

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"HF API exception: {e}")
        return {
            "success": False,
            "predicted_smiles": "",
            "raw_output": None,
            "latency_seconds": elapsed,
            "error": f"{type(e).__name__}: {str(e)}",
        }


async def _handle_tier2_ai_fallback(req: ReactRequest) -> ReactResponse:
    """
    Tier 2: AI 推論フォールバックハンドラー。
    ルールベースで対応できない反応を AI に推論させる。
    """

    # HF_TOKEN が未設定の場合のフォールバック
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set — Tier 2 AI disabled, returning mock response")
        return _tier2_mock_response(req)

    # 反応 SMILES を構築: "reagent_1.reagent_2>>" (RXNSMILES 形式)
    reaction_input = f"{req.reagent_1}.{req.reagent_2}>>"
    logger.info(f"Tier 2 AI inference: input='{reaction_input}'")

    # AI 推論実行
    ai_result = await _call_hf_inference(reaction_input)

    # --- AI が失敗した場合: タール化ゲームオーバー ---
    if not ai_result["success"]:
        logger.warning(
            f"Tier 2 AI failed: {ai_result['error']} → タール化ゲームオーバー"
        )
        return ReactResponse(
            status=ResultStatus.GAME_OVER,
            message=(
                f"💀 AI 推論が失敗しました: {ai_result['error']}\n"
                "反応が暴走し、全てがタール（炭化物）になりました…\n"
                "🏭 ゲームオーバー: 実験室が煤だらけです！"
            ),
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type="tier2_ai_carbonization",
            products=[
                ProductInfo(
                    smiles=TAR_SMILES,
                    name=TAR_NAME,
                    carbon_class="N/A",
                    reaction_type="carbonization",
                )
            ],
            byproducts=["CO2 (二酸化炭素)", "H2O (水蒸気)", "煤 (Soot)"],
            tier="tier2_ai",
            ai_latency_seconds=ai_result["latency_seconds"],
            ai_model=HF_MODEL_ID,
            condition_summary=f"Tier 2 AI 推論失敗 → タール化",
        )

    # --- AI が成功した場合 ---
    predicted_smiles = ai_result["predicted_smiles"]
    latency = ai_result["latency_seconds"]

    # AI の予測結果を複数の生成物に分割 (ドット区切り)
    product_smiles_list = [
        s.strip() for s in predicted_smiles.replace(".", " ").split()
        if s.strip()
    ]
    if not product_smiles_list:
        product_smiles_list = [predicted_smiles]

    product_infos = [
        ProductInfo(
            smiles=smi,
            name=f"🤖 AI 予測生成物 #{i+1}",
            carbon_class="AI-predicted",
            reaction_type="ai_prediction",
        )
        for i, smi in enumerate(product_smiles_list)
    ]

    return ReactResponse(
        status=ResultStatus.SUCCESS,
        message=(
            f"🤖 Tier 2 AI 推論成功！({latency:.2f}秒)\n"
            f"モデル '{HF_MODEL_ID}' が {len(product_infos)} 個の生成物を予測しました。\n"
            "⚠️ AI 予測のため、意図しない副産物が含まれる可能性があります。"
        ),
        reagent_1_smiles=req.reagent_1,
        reagent_2_smiles=req.reagent_2,
        reaction_type="tier2_ai_prediction",
        products=product_infos,
        tier="tier2_ai",
        ai_latency_seconds=latency,
        ai_model=HF_MODEL_ID,
        condition_summary=(
            f"Tier 1 未対応 → Tier 2 AI ({HF_MODEL_ID}) にフォールバック | "
            f"推論時間: {latency:.2f}秒"
        ),
    )


def _tier2_mock_response(req: ReactRequest) -> ReactResponse:
    """
    HF_TOKEN 未設定時のモック Tier 2 レスポンス。
    未知の反応 = 常にタール化 (ゲーム内で M11/M13 を確実にクリアさせるため)
    """
    return ReactResponse(
        status=ResultStatus.SUCCESS,
        message=(
            "🏭 未知の反応条件のため、反応が暴走しタール化しました。\n"
            "副反応が連鎖し、黒色の炭化物が沈殿しています…\n"
            "（これもある意味、化学を学ぶ第一歩です！）"
        ),
        reagent_1_smiles=req.reagent_1,
        reagent_2_smiles=req.reagent_2,
        reaction_type="carbonization",
        products=[
            ProductInfo(
                smiles=TAR_SMILES,
                name=TAR_NAME,
                carbon_class="N/A",
                reaction_type="carbonization",
            ),
        ],
        byproducts=["CO2 (二酸化炭素)", "H2O (水蒸気)", "煤 (Soot)"],
        tier="tier2_mock",
        ai_latency_seconds=0.0,
        ai_model="mock (rule-based tar)",
        condition_summary="未対応の反応 → タール化",
    )


# =====================================================================
# Run
# =====================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
