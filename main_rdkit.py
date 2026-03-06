"""
ChemFactory – Phase 0 + Phase 1 API (RDKit + HF AI Hybrid Version)
===================================================================
Tier 1: RDKit の ReactionFromSmarts を用いたローカルルールベース判定
Tier 2: Hugging Face Inference API を用いた AI 推論フォールバック

Python 3.12以下など、RDKit が利用できる環境向けの実装。
"""

from __future__ import annotations

import logging
import os
import time
import base64
import io
from enum import Enum
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 実行環境で RDKit が利用可能かどうかに依存します
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
except ImportError:
    raise ImportError("このスクリプトを実行するには RDKit が必要です。(pip install rdkit)")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("chemfactory_rdkit")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ChemFactory – RDKit + AI Hybrid API",
    description="Tier 1: RDKit ルールベース / Tier 2: HF AI フォールバック",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Tier 1: 反応定義 (RDKit ReactionFromSmarts)
# ---------------------------------------------------------------------------

def get_radical_halogenation_rxn(halogen: str) -> AllChem.ChemicalReaction:
    valid_halo = "Cl" if halogen == "ClCl" else "Br"
    smarts = f"[C:1]([H])>>[C:1]{valid_halo}"
    rxn = AllChem.ReactionFromSmarts(smarts)
    rxn.Initialize()
    return rxn

rxn_sn2 = AllChem.ReactionFromSmarts("[C:1]-[Cl,Br,I]>>[C:1]-O")
rxn_sn2.Initialize()

rxn_e2 = AllChem.ReactionFromSmarts("[H][C:1]-[C:2]-[Cl,Br,I]>>[C:1]=[C:2]")
rxn_e2.Initialize()

rxn_grignard = AllChem.ReactionFromSmarts("[C:1]-[Cl,Br,I:2]>>[C:1]-[Mg]-[*:2]")
rxn_grignard.Initialize()

# Phase 3: 芳香族求電子置換反応 (Friedel-Crafts アシル化のようなダミーアート分子生成用)
rxn_eas = AllChem.ReactionFromSmarts("[c:1][H]>>[c:1]C(=O)C")
rxn_eas.Initialize()

# Phase 2.1: グリニャール試薬による炭素鎖延長 / 酸化
rxn_grignard_carbonyl = AllChem.ReactionFromSmarts("[C:1]-[Mg]-[*:2].[C:3]=[O:4]>>[C:1]-[C:3]-[O:4]")
rxn_grignard_carbonyl.Initialize()

rxn_grignard_o2 = AllChem.ReactionFromSmarts("[C:1]-[Mg]-[*:2].[O:3]=[O:4]>>[C:1]-[O:3]")
rxn_grignard_o2.Initialize()

# ---------------------------------------------------------------------------
# Tier 2: Hugging Face AI 設定
# ---------------------------------------------------------------------------

HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
HF_MODEL_ID: str = os.environ.get(
    "HF_MODEL_ID",
    "ncf/rxn-forward-prediction"
)
HF_API_URL: str = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_TIMEOUT_SECONDS: float = 15.0

# ダミー生成物 (タール化)
TAR_SMILES = "C1=CC=CC=C1"
TAR_NAME = "🏭 タール / 炭化物"


# ---------------------------------------------------------------------------
# Open World Reaction Registry
# ---------------------------------------------------------------------------
def _canon(s: str) -> str:
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else s
    except:
        return s

def get_reaction_key(r1: str, r2: str, catalyst: str = None) -> tuple[str, str, str]:
    c1 = _canon(r1)
    c2 = _canon(r2)
    sorted_r = sorted([c1, c2])
    cat = catalyst if catalyst else "None"
    return (sorted_r[0], sorted_r[1], cat)

OPEN_WORLD_REACTIONS = {
    # --------------------------------------------------------
    # Tier 1: アスピリンルート
    # --------------------------------------------------------
    get_reaction_key("c1ccccc1O", "O=C=O", "NaOH"): {
        "product": "O=C(O)c1ccccc1O",
        "product_name": "サリチル酸",
        "message": "✨ コルベ・シュミット反応成功！フェノールからサリチル酸を合成した。",
        "byproducts": [],
        "reaction_type": "kolbe_schmitt"
    },
    get_reaction_key("O=C(O)c1ccccc1O", "CC(=O)OC(C)=O", "H2SO4"): {
        "product": "CC(=O)Oc1ccccc1C(=O)O",
        "product_name": "💊 アスピリン",
        "message": "✨ エステル化成功！世界初の人工合成医薬品「アスピリン」が完成した！",
        "byproducts": ["CC(=O)O"], # 酢酸
        "reaction_type": "esterification"
    },
    # --------------------------------------------------------
    # Tier 1: パラセタモールルート
    # --------------------------------------------------------
    get_reaction_key("c1ccccc1O", "HNO3", "Acid"): {
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
    get_reaction_key("O=[N+]([O-])c1ccc(O)cc1", "[H][H]", "Sn(スズ)"): {
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


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ReactRequest(BaseModel):
    reagent_1: str = Field(..., examples=["C", "CC", "CCCl", "c1ccccc1"])
    reagent_2: str = Field(..., examples=["ClCl", "BrBr", "O", "CC(=O)Cl"])
    condition_light: bool = Field(..., examples=[True])
    temperature: Optional[int] = Field(None, ge=0, le=100)
    solvent_type: Optional[str] = Field(None, examples=["protic", "aprotic"])
    catalyst: Optional[str] = Field(None, description="追加の触媒指定")

class ResultStatus(str, Enum):
    SUCCESS = "success"
    NO_REACTION = "no_reaction"
    GAME_OVER = "game_over"

class ProductInfo(BaseModel):
    smiles: str
    name: Optional[str] = None
    carbon_class: Optional[str] = None
    reaction_type: Optional[str] = None

class ReactResponse(BaseModel):
    status: ResultStatus
    message: str
    reagent_1_smiles: str
    reagent_2_smiles: str
    reaction_type: Optional[str] = None
    products: list[ProductInfo] = Field(default_factory=list)
    byproducts: list[str] = Field(default_factory=list)
    hx_byproduct: Optional[str] = None
    condition_summary: Optional[str] = None
    tier: Optional[str] = Field(None, description="tier1_rule or tier2_ai")
    ai_latency_seconds: Optional[float] = Field(None)
    ai_model: Optional[str] = Field(None)
    image_base64: Optional[str] = Field(None, description="Base64 encoded PNG image")
    rarity: Optional[str] = Field(None, description="ゲーム内のレア度 (N, SR, SSRなど)")
    flavor_text: Optional[str] = Field(None, description="フレーバーテキスト")
    quiz_question: Optional[str] = Field(None, description="学習用クイズの問題文")
    quiz_options: Optional[list[str]] = Field(None, description="クイズの選択肢")
    quiz_correct_index: Optional[int] = Field(None, description="正解のインデックス(0-2)")


# ---------------------------------------------------------------------------
# ヘルパー関数: Image Generation
# ---------------------------------------------------------------------------
def generate_image_base64(smiles: str) -> Optional[str]:
    """SMILES文字列からPIL画像を生成し、Base64文字列に変換して返す"""
    if not smiles:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 化学的に無効なSMILESの場合は None (null) を返す
            return None
            
        img = Draw.MolToImage(mol)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return img_base64
    except Exception as e:
        logger.warning(f"Image generation failed for SMILES '{smiles}': {e}")
        return None


# ---------------------------------------------------------------------------
# ヘルパー関数: Tier 1 実行
# ---------------------------------------------------------------------------
def execute_reaction(rxn: AllChem.ChemicalReaction, reactant_mol: Chem.Mol) -> list[str]:
    mol_h = Chem.AddHs(reactant_mol)
    product_sets = rxn.RunReactants((mol_h,))
    
    seen = set()
    unique_products = []
    for products in product_sets:
        for p in products:
            try:
                Chem.SanitizeMol(p)
                p_no_h = Chem.RemoveHs(p)
                smi = Chem.MolToSmiles(p_no_h)
                if smi not in seen:
                    seen.add(smi)
                    unique_products.append(smi)
            except Exception:
                continue
    return unique_products

def execute_reaction_2(rxn: AllChem.ChemicalReaction, mol1: Chem.Mol, mol2: Chem.Mol) -> list[str]:
    """2つの基質を用いた反応処理"""
    mol_h1 = Chem.AddHs(mol1)
    mol_h2 = Chem.AddHs(mol2)
    product_sets = rxn.RunReactants((mol_h1, mol_h2))
    
    seen = set()
    unique_products = []
    for products in product_sets:
        for p in products:
            try:
                Chem.SanitizeMol(p)
                p_no_h = Chem.RemoveHs(p)
                smi = Chem.MolToSmiles(p_no_h)
                if smi not in seen:
                    seen.add(smi)
                    unique_products.append(smi)
            except Exception:
                continue
    return unique_products

# ---------------------------------------------------------------------------
# ヘルパー関数: Tier 2 実行
# ---------------------------------------------------------------------------
async def _call_hf_inference(reaction_smiles: str) -> dict:
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": reaction_smiles,
        "parameters": {"max_new_tokens": 256, "top_k": 5, "temperature": 0.8},
    }

    start_time = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT_SECONDS) as client:
            resp = await client.post(HF_API_URL, json=payload, headers=headers)
        elapsed = time.perf_counter() - start_time
        
        if resp.status_code != 200:
            return {"success": False, "error": resp.text[:200], "latency": elapsed}

        result = resp.json()
        predicted = ""
        if isinstance(result, list) and len(result) > 0:
            predicted = result[0].get("generated_text", "") if isinstance(result[0], dict) else str(result[0])
        elif isinstance(result, dict):
            predicted = result.get("generated_text", "")

        return {
            "success": bool(predicted),
            "predicted_smiles": predicted.strip(),
            "latency": elapsed,
            "error": "Empty output" if not predicted else None
        }
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return {"success": False, "error": str(e), "latency": elapsed}

async def handle_tier2_fallback(req: ReactRequest) -> ReactResponse:
    if not HF_TOKEN:
        # モック機能
        import hashlib
        seed = hashlib.md5(f"{req.reagent_1}:{req.reagent_2}".encode()).hexdigest()
        fake = TAR_SMILES if seed[0] < '8' else "CCO"
        
        img_b64 = generate_image_base64(fake)
        return ReactResponse(
            status=ResultStatus.SUCCESS if fake != TAR_SMILES else ResultStatus.GAME_OVER,
            message="🎲 Tier 2 モック (HF_TOKEN未設定)",
            reagent_1_smiles=req.reagent_1, reagent_2_smiles=req.reagent_2,
            reaction_type="tier2_mock", products=[ProductInfo(smiles=fake, name="Mock Product")],
            tier="tier2_mock", ai_latency_seconds=0.0, ai_model="mock",
            image_base64=img_b64,
            rarity="SSR",
            flavor_text="未知の組み合わせにより誕生したキモい副産物。取り扱い注意。",
            quiz_question="想定外の副産物が生まれました。化学において最も重要な心構えは？",
            quiz_options=["運を天に任せる", "安全第一で原因を記録する", "そのまま放置する"],
            quiz_correct_index=1
        )

    # 推論実行
    reaction_input = f"{req.reagent_1}.{req.reagent_2}>>"
    ai_result = await _call_hf_inference(reaction_input)

    lat = ai_result.get("latency", 0.0)
    
    if not ai_result["success"]:
        img_b64 = generate_image_base64(TAR_SMILES)
        return ReactResponse(
            status=ResultStatus.GAME_OVER,
            message=f"💀 AI 推論失敗（タール化）: {ai_result['error']}",
            reagent_1_smiles=req.reagent_1, reagent_2_smiles=req.reagent_2,
            reaction_type="tier2_ai_carbonization", products=[ProductInfo(smiles=TAR_SMILES, name=TAR_NAME)],
            tier="tier2_ai", ai_latency_seconds=lat, ai_model=HF_MODEL_ID,
            image_base64=img_b64,
            rarity="SSR",
            flavor_text="未知の組み合わせにより誕生したキモい副産物。取り扱い注意。",
            quiz_question="想定外の副産物が生まれました。化学において最も重要な心構えは？",
            quiz_options=["運を天に任せる", "安全第一で原因を記録する", "そのまま放置する"],
            quiz_correct_index=1
        )

    pred_smi = ai_result["predicted_smiles"]
    img_b64 = generate_image_base64(pred_smi)
    return ReactResponse(
        status=ResultStatus.SUCCESS,
        message=f"🤖 Tier 2 AI 推論成功！意図しない副産物の可能性があります。",
        reagent_1_smiles=req.reagent_1, reagent_2_smiles=req.reagent_2,
        reaction_type="tier2_ai_prediction", products=[ProductInfo(smiles=pred_smi, name="AI Pred Product")],
        tier="tier2_ai", ai_latency_seconds=lat, ai_model=HF_MODEL_ID,
        image_base64=img_b64,
        rarity="SSR",
        flavor_text="未知の組み合わせにより誕生したキモい副産物。取り扱い注意。",
        quiz_question="想定外の副産物が生まれました。化学において最も重要な心構えは？",
        quiz_options=["運を天に任せる", "安全第一で原因を記録する", "そのまま放置する"],
        quiz_correct_index=1
    )


# ---------------------------------------------------------------------------
# エンドポイント
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "RDKit + HF AI Hybrid API", "hf_enabled": bool(HF_TOKEN)}

@app.post("/react", response_model=ReactResponse)
async def react(req: ReactRequest):
    # ==================================================================
    # カスタム触媒レシピ (新機能)
    # ==================================================================
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
            tier="tier1_rule",
            image_base64=generate_image_base64(prs_smi[0]) if prs_smi else None
        )

    reagent1_mol = Chem.MolFromSmiles(req.reagent_1)
    if reagent1_mol is None:
        return ReactResponse(
            status=ResultStatus.NO_REACTION,
            message="無効なSMILESが含まれています。",
            reagent_1_smiles=req.reagent_1, reagent_2_smiles=req.reagent_2,
            image_base64=None
        )

    temp = req.temperature if req.temperature is not None else 25
    solvent = req.solvent_type if req.solvent_type else "aprotic"
    
    # ----------------------------------------------------------------------
    # Tier 1 ルールベース判定
    # ----------------------------------------------------------------------
    products_smiles = []
    reaction_type_str = ""
    message = ""
    tier1_handled = False
    hx_byproduct = None

    # Phase 1: 置換 or 脱離
    if req.reagent_2 in ["O", "[O-]"]:
        tier1_handled = True
        if temp >= 60:
            reaction_type_str = "E2 Elimination"
            products_smiles = execute_reaction(rxn_e2, reagent1_mol)
            if not products_smiles:
                reaction_type_str = "SN2 Substitution (E2 Fallback)"
                products_smiles = execute_reaction(rxn_sn2, reagent1_mol)
        else:
            reaction_type_str = "SN2 Substitution"
            products_smiles = execute_reaction(rxn_sn2, reagent1_mol)

    # Phase 0: ラジカルハロゲン化
    elif req.reagent_2 in ["ClCl", "BrBr"]:
        tier1_handled = True
        if req.condition_light:
            reaction_type_str = "Radical Halogenation"
            rxn = get_radical_halogenation_rxn(req.reagent_2)
            products_smiles = execute_reaction(rxn, reagent1_mol)
            hx_byproduct = "HCl" if req.reagent_2 == "ClCl" else "HBr"

    # Phase 2: グリニャール試薬生成 (Mg挿入)
    elif req.reagent_2 == "Mg":
        if solvent == "protic":
            return ReactResponse(
                status=ResultStatus.GAME_OVER,
                message="プクプクと泡を立ててMgが溶けてしまった…（溶媒の選択ミス）",
                reagent_1_smiles=req.reagent_1, reagent_2_smiles=req.reagent_2,
                reaction_type="Grignard Formation Failure", products=[],
                tier="tier1_rule",
                image_base64=None
            )
        else:
            tier1_handled = True
            reaction_type_str = "Grignard Reagent Formation"
            products_smiles = execute_reaction(rxn_grignard, reagent1_mol)

    # Phase 2.1: 炭素鎖の延長 (アルコール等の生成)
    elif "[Mg]" in req.reagent_1 and req.reagent_2 in ["C=O", "O=O"]:
        if solvent != "aprotic":
            return ReactResponse(
                status=ResultStatus.GAME_OVER,
                message="水などのプロトン性溶媒によりグリニャール試薬が完全に失活した…",
                reagent_1_smiles=req.reagent_1, reagent_2_smiles=req.reagent_2,
                reaction_type="Grignard Extension Failure", products=[],
                tier="tier1_rule",
                image_base64=None
            )
        else:
            tier1_handled = True
            reaction_type_str = "Grignard Chain Extension"
            reagent2_mol = Chem.MolFromSmiles(req.reagent_2)
            if req.reagent_2 == "C=O":
                products_smiles = execute_reaction_2(rxn_grignard_carbonyl, reagent1_mol, reagent2_mol)
            else:
                products_smiles = execute_reaction_2(rxn_grignard_o2, reagent1_mol, reagent2_mol)
            # RDKit上は酸素がアニオン化するため、Hをつけてアルコール(-OH)にする簡易補正
            products_smiles = [s.replace("[O-]", "O") for s in products_smiles]

    # Phase 3: 芳香族求電子置換反応 (Friedel-Crafts アシル化)
    elif req.reagent_1 == "c1ccccc1" and req.reagent_2 == "CC(=O)Cl":
        if solvent != "aprotic":
            return ReactResponse(
                status=ResultStatus.GAME_OVER,
                message="水などのプロトン性溶媒により触媒が失活し、反応が進まなかった…",
                reagent_1_smiles=req.reagent_1, reagent_2_smiles=req.reagent_2,
                reaction_type="EAS Failure (Solvent)", products=[],
                tier="tier1_rule",
                image_base64=None
            )
        elif req.catalyst != "AlCl3":
            return ReactResponse(
                status=ResultStatus.NO_REACTION,
                message="ベンゼン環は安定なため、強力なルイス酸触媒がないと反応しません。",
                reagent_1_smiles=req.reagent_1, reagent_2_smiles=req.reagent_2,
                reaction_type="EAS Failure (No Catalyst)", products=[],
                tier="tier1_rule",
                image_base64=None
            )
        else:
            tier1_handled = True
            reaction_type_str = "Electrophilic Aromatic Substitution"
            # アセトフェノンをベースに見栄えを拡張したダミーのアート分子を固定で生成
            art_molecule_smiles = "CC(=O)c1ccc(C(=O)C)cc1" # p-Diacetylbenzene
            products_smiles = [art_molecule_smiles]
            hx_byproduct = "HCl"

    if tier1_handled:
        if not products_smiles:
            # RDKitの条件に合わない無茶な組み合わせは、副反応が暴走しタール化したものとする
            return ReactResponse(
                status=ResultStatus.GAME_OVER,
                message="反応条件に合わない無茶な組み合わせです。副反応が暴走しタールになりました。",
                reagent_1_smiles=req.reagent_1, reagent_2_smiles=req.reagent_2,
                reaction_type="Tar Formation", products=[ProductInfo(smiles=TAR_SMILES, name=TAR_NAME)],
                tier="tier1_fallback",
                image_base64=generate_image_base64(TAR_SMILES)
            )
            
        img_b64 = generate_image_base64(products_smiles[0])
        product_infos = [ProductInfo(smiles=smi) for smi in products_smiles]
        
        # レア度とフレーバーテキストの判定
        if reaction_type_str == "Electrophilic Aromatic Substitution":
            rarity = "UR"
            flavor_text = "配向性のパズルを解き明かし、ついに目的のアート分子が完成した！全合成クリア！"
        elif reaction_type_str == "Grignard Reagent Formation":
            rarity = "SR"
            flavor_text = "強力な求核剤、グリニャール試薬が完成した。これで炭素骨格を拡張できる。"
        elif reaction_type_str == "Grignard Chain Extension":
            rarity = "SSR"
            flavor_text = "炭素鎖の延長に成功した。これでより大きな骨格を作ることができる。"
        elif temp >= 80 and reaction_type_str == "E2 Elimination":
            rarity = "SR"
            flavor_text = "高温環境下での過激な脱離反応により生成。フラスコが少し焦げた。"
        else:
            rarity = "N"
            flavor_text = "教科書通りの安全な生成物。実験は成功だ。"
            
        # クイズ情報の判定
        quiz_question = None
        quiz_options = None
        quiz_correct_index = None
        
        if "Radical" in reaction_type_str:
            quiz_question = "この反応で最初に発生する不安定な中間体は？"
            quiz_options = ["カルボカチオン", "カルバニオン", "ラジカル"]
            quiz_correct_index = 2
        elif "Substitution" in reaction_type_str or "SN2" in reaction_type_str:
            quiz_question = "この反応における求核剤（攻撃する側）の役割を果たしたのは？"
            quiz_options = ["溶媒", "塩基/アニオン", "光"]
            quiz_correct_index = 1
        elif "Grignard" in reaction_type_str:
            quiz_question = "この反応において非プロトン性溶媒（Aprotic）が必須な理由は？"
            quiz_options = ["水と反応して失活するのを防ぐため", "温度を急激に上げるため", "色を綺麗にするため"]
            quiz_correct_index = 0
        elif "Electrophilic" in reaction_type_str:
            quiz_question = "ベンゼン環の求電子置換反応において、AlCl3触媒の役割は？"
            quiz_options = ["溶解度を上げるため", "求電子剤を強力に活性化するため", "ベンゼン環を直接壊すため"]
            quiz_correct_index = 1
        elif "Elimination" in reaction_type_str:
            quiz_question = "脱離反応が起きやすくなる条件は？"
            quiz_options = ["低温", "高温", "中性条件"]
            quiz_correct_index = 1
        
        return ReactResponse(
            status=ResultStatus.SUCCESS,
            message=f"🔬 Tier 1 ({reaction_type_str}) が成功しました！",
            reagent_1_smiles=req.reagent_1, reagent_2_smiles=req.reagent_2,
            reaction_type=reaction_type_str, products=product_infos,
            hx_byproduct=hx_byproduct,
            tier="tier1_rule",
            image_base64=img_b64,
            rarity=rarity,
            flavor_text=flavor_text,
            quiz_question=quiz_question,
            quiz_options=quiz_options,
            quiz_correct_index=quiz_correct_index
        )
        
        
    # ----------------------------------------------------------------------
    # Tier 1.5 Open World Reaction Registry 判定
    # ----------------------------------------------------------------------
    registry_key = get_reaction_key(req.reagent_1, req.reagent_2, req.catalyst)
    if registry_key in OPEN_WORLD_REACTIONS:
        recipe = OPEN_WORLD_REACTIONS[registry_key]
        img_b64 = generate_image_base64(recipe["product"])
        return ReactResponse(
            status=ResultStatus.SUCCESS,
            message=recipe["message"],
            reagent_1_smiles=req.reagent_1,
            reagent_2_smiles=req.reagent_2,
            reaction_type=recipe.get("reaction_type", "open_world_recipe"),
            products=[ProductInfo(smiles=recipe["product"], name=recipe["product_name"])],
            byproducts=recipe.get("byproducts", []),
            condition_summary=f"Catalyst: {req.catalyst or 'None'}",
            tier="tier1_registry",
            image_base64=img_b64,
            rarity="SSR" if "💊" in recipe["product_name"] else "SR",
            flavor_text=recipe.get("message", "")
        )

    # ----------------------------------------------------------------------
    # Tier 2 AI推論フォールバック
    # ----------------------------------------------------------------------
    logger.info(f"Tier 1未対応 → Tier 2へ: {req.reagent_1} + {req.reagent_2}")
    return await handle_tier2_fallback(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_rdkit:app", host="0.0.0.0", port=8001, reload=True)

