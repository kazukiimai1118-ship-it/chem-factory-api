import os
import sys
import json
import asyncio
import httpx
import logging
import hashlib
from typing import List, Dict

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# パス設定 (backend/chem-factory-api をカレントディレクトリとして実行することを想定)
sys.path.append(os.getcwd())

# 必要な定数とプンプト (main.py からコピー、またはインポートを試みる)
try:
    from main import MECHANISM_PROMPT, HF_API_URL, HF_TOKEN, save_reaction_cache, load_reaction_cache, _call_llm_structured_inference
except ImportError:
    # インポート失敗時のフォールバック (基本はインポートを想定)
    logger.error("Could not import from main.py. Please run from the backend/chem-factory-api directory.")
    sys.exit(1)

# 事前生成する反応リスト
COMMON_REACTIONS = [
    {
        "r1": "C", # Methane
        "r2": "ClCl", # Chlorine
        "cat": "light",
        "description": "Radical Chlorination of Methane"
    },
    {
        "r1": "CC", # Ethane
        "r2": "BrBr", # Bromine
        "cat": "light",
        "description": "Radical Bromination of Ethane"
    },
    {
        "r1": "CCl", # Chloromethane
        "r2": "[OH-]", # Hydroxide
        "cat": "NaOH",
        "description": "SN2 Reaction (Chloromethane -> Methanol)"
    },
    {
        "r1": "CCBr", # Bromoethane
        "r2": "[OH-]", # Hydroxide
        "cat": "NaOH",
        "description": "SN2 Reaction (Bromoethane -> Ethanol)"
    },
    {
        "r1": "C=C", # Ethene
        "r2": "HH", # Hydrogen
        "cat": "Pt",
        "description": "Hydrogenation of Ethene"
    },
    {
        "r1": "c1ccccc1", # Benzene
        "r2": "ClCl", # Chlorine
        "cat": "FeCl3",
        "description": "Electrophilic Aromatic Substitution (Benzene -> Chlorobenzene)"
    }
]

async def generate_cache():
    logger.info("Starting Batch Reaction Mechanism Cache Generation...")
    
    if not HF_TOKEN:
        logger.error("HF_TOKEN is not set. Cannot call LLM API.")
        return

    success_count = 0
    fail_count = 0
    skip_count = 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        for reaction in COMMON_REACTIONS:
            r1 = reaction["r1"]
            r2 = reaction["r2"]
            cat = reaction["cat"]
            desc = reaction["description"]

            logger.info(f"Processing: {desc} ({r1} + {r2}, cat={cat})")

            # 既存キャッシュの確認
            existing = load_reaction_cache(r1, r2, cat)
            if existing:
                logger.info(f"  -> Cache already exists. Skipping.")
                skip_count += 1
                continue

            # 推論実行 (main.py の改良されたリトライ機構付き関数を呼び出し)
            try:
                result = await _call_llm_structured_inference(r1, r2, cat)
                if result.get("success"):
                    # キャッシュ保存 (関数内で処理されるが明示的にログ)
                    save_reaction_cache(r1, r2, cat, result)
                    logger.info(f"  -> SUCCESS: Cache generated (Latency: {result.get('latency', 0):.2f}s)")
                    success_count += 1
                else:
                    logger.error(f"  -> FAILED: {result.get('error')}")
                    fail_count += 1
            except Exception as e:
                logger.error(f"  -> EXCEPTION: {e}")
                fail_count += 1
            
            # API 負荷軽減のためのスリープ
            await asyncio.sleep(1.0)

    logger.info("========================================")
    logger.info(f"Cache Generation Completed.")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed:  {fail_count}")
    logger.info(f"Skipped: {skip_count}")
    logger.info("========================================")

if __name__ == "__main__":
    asyncio.run(generate_cache())
