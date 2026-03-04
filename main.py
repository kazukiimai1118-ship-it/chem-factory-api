"""
ChemFactory – Phase 0 + Phase 1 API (Tier 1 + Tier 2 Hybrid)
==============================================================
Tier 1: ローカルルールベース処理 (Pure Python, RDKit-free)
Tier 2: 外部 AI 推論フォールバック (Hugging Face Inference API)

Phase 0: アルカンのラジカルハロゲン化  R-H + X₂ --hν→ R-X + HX
Phase 1: ハロアルカンの置換/脱離反応
         - SN2 (低温 + 非プロトン性溶媒) R-X + Nu⁻ → R-Nu + X⁻
         - E2  (高温)                     R-X + Base → Alkene + HX + Base-H
Tier 2:  未知の組み合わせ → HF 反応予測 AI へフォールバック

アーキテクチャ:
  1. /react リクエスト受信
  2. Tier 1 ルールベースで反応タイプ判定
  3. Tier 1 で対応不可 → Tier 2 AI 推論 (HF Inference API)
  4. Tier 2 タイムアウト/エラー → タール化ゲームオーバー
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
        """分子を SMILES 文字列に変換する。"""
        if self.carbon_count == 0:
            return ""
        visited = [False] * self.carbon_count
        return self._build_smiles(0, visited)

    def _build_smiles(self, idx: int, visited: list[bool]) -> str:
        visited[idx] = True
        c = self.carbons[idx]

        # ハロゲン置換基
        hal_str = ""
        for h in c["halogens"]:
            if len(h) == 1:
                hal_str += h
            else:
                hal_str += f"({h})"

        # その他の置換基 (OH, NH2, etc.)
        sub_str = ""
        for s in c.get("substituents", []):
            if len(s) == 1:
                sub_str += s
            else:
                sub_str += f"({s})"

        # 隣接する未訪問の炭素
        unvisited_neighbors = [n for n in c["neighbors"] if not visited[n]]

        if not unvisited_neighbors:
            return f"C{hal_str}{sub_str}"

        main = unvisited_neighbors[0]
        branches = unvisited_neighbors[1:]

        # 二重結合判定
        bond_char = "=" if main in c.get("double_bond_to", []) else ""

        result = f"C{hal_str}{sub_str}"
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

def detect_reaction_type(
    reagent_1: str,
    reagent_2: str,
    condition_light: bool,
    temperature: Optional[int],
    solvent_type: Optional[str],
) -> str:
    """
    入力試薬から反応タイプを自動判定する。

    Returns:
        "radical_halogenation" | "substitution_elimination" | "unknown"
    """
    # ハロゲン分子 (X₂) + アルカン + 光 → ラジカルハロゲン化
    if reagent_2 in HALOGEN_MAP:
        return "radical_halogenation"

    # 求核剤/塩基 + ハロアルカン → 置換 or 脱離
    if reagent_2 in NUCLEOPHILE_BASE_MAP:
        haloalkane = parse_haloalkane_smiles(reagent_1)
        if haloalkane is not None:
            return "substitution_elimination"

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
    """反応リクエスト (Phase 0 + Phase 1 統合)"""
    reagent_1: str = Field(
        ...,
        examples=["C", "CC", "CCCl", "CC(Cl)C"],
        description="基質の SMILES (アルカン or ハロアルカン)",
    )
    reagent_2: str = Field(
        ...,
        examples=["ClCl", "BrBr", "O", "[O-]"],
        description="試薬の SMILES (ハロゲン分子 or 求核剤/塩基)",
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

@app.get("/")
async def root():
    """ヘルスチェック & API 情報"""
    hf_configured = bool(os.environ.get("HF_TOKEN"))
    return {
        "service": "ChemFactory Tier 1+2 Hybrid API",
        "status": "running",
        "version": "0.3.0",
        "description": "Tier 1 ルールベース + Tier 2 AI 推論ハイブリッド反応シミュレーター",
        "tier2_ai_enabled": hf_configured,
        "tier2_ai_note": "HF_TOKEN 環境変数を設定すると Tier 2 AI 推論が有効になります" if not hf_configured else "Tier 2 AI 推論が有効です",
        "endpoints": {
            "POST /react": "化学反応を実行 (Tier 1: ルールベース → Tier 2: AI フォールバック)",
            "GET /docs": "Swagger UI (API ドキュメント)",
        },
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
    # 反応タイプの自動判定
    # ==================================================================
    rxn_type = detect_reaction_type(
        req.reagent_1, req.reagent_2,
        req.condition_light, req.temperature, req.solvent_type,
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
#  SECTION 8:  Tier 2 – AI 推論フォールバック (Hugging Face Inference API)
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
    開発・テスト時に使用する。
    """
    import hashlib

    # 入力の組み合わせからダミー生成物を決定論的に生成
    seed = hashlib.md5(
        f"{req.reagent_1}:{req.reagent_2}".encode()
    ).hexdigest()

    # ダミー生成物テーブル (シードの先頭文字で選択)
    mock_products_table = {
        "0": ("C(=O)O", "🎲 酢酸 (Acetic Acid) [Mock]"),
        "1": ("CCO", "🎲 エタノール [Mock]"),
        "2": ("CC=O", "🎲 アセトアルデヒド [Mock]"),
        "3": ("C1CCCCC1", "🎲 シクロヘキサン [Mock]"),
        "4": ("CC(=O)C", "🎲 アセトン [Mock]"),
        "5": ("C=CC", "🎲 プロペン [Mock]"),
        "6": ("CCCC", "🎲 ブタン [Mock]"),
        "7": ("CC(O)C", "🎲 2-プロパノール [Mock]"),
        "8": ("ClC=C", "🎲 塩化ビニル [Mock]"),
        "9": ("C#N", "🎲 シアン化水素 [Mock]"),
        "a": ("CC(=O)O", "🎲 酢酸 [Mock]"),
        "b": ("C1=CC=CC=C1O", "🎲 フェノール [Mock]"),
        "c": ("CCOC", "🎲 ジエチルエーテル [Mock]"),
        "d": ("C(Cl)(Cl)Cl", "🎲 クロロホルム [Mock]"),
        "e": ("CC#C", "🎲 プロピン [Mock]"),
        "f": (TAR_SMILES, TAR_NAME),
    }

    chosen = mock_products_table.get(seed[0], (TAR_SMILES, TAR_NAME))

    return ReactResponse(
        status=ResultStatus.SUCCESS,
        message=(
            "🎲 Tier 2 モックモード (HF_TOKEN 未設定)\n"
            "AI の代わりにダミー生成物を返しています。\n"
            "本番で使用するには環境変数 HF_TOKEN を設定してください。"
        ),
        reagent_1_smiles=req.reagent_1,
        reagent_2_smiles=req.reagent_2,
        reaction_type="tier2_mock",
        products=[
            ProductInfo(
                smiles=chosen[0],
                name=chosen[1],
                carbon_class="mock",
                reaction_type="mock_prediction",
            ),
        ],
        byproducts=["❓ (モックのため副生成物は不明)"],
        tier="tier2_mock",
        ai_latency_seconds=0.0,
        ai_model="mock (HF_TOKEN not set)",
        condition_summary="Tier 1 未対応 → Tier 2 モック (HF_TOKEN 未設定)",
    )


# =====================================================================
# Run
# =====================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
