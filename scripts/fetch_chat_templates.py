#!/usr/bin/env python3
"""
Fetch chat templates from popular HuggingFace models.

Downloads only tokenizer_config.json files (~1-10KB each) and extracts
chat templates. No model weights are downloaded.

Purpose: Collect real-world templates to analyze what Jinja2 features
are actually used, then write targeted compatibility tests.

Usage:
    python scripts/fetch_chat_templates.py

    # With HF token for gated models (recommended)
    HF_TOKEN=hf_xxx python scripts/fetch_chat_templates.py

    # Force re-download all (skip cache)
    python scripts/fetch_chat_templates.py --force

    # Analyze collected templates using Jinja2 AST
    python scripts/fetch_chat_templates.py --analyze

Output:
    cache/templates/<org>--<model>.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# =============================================================================
# Diverse model list - prioritize architectural diversity over size variations
# =============================================================================

MODELS = [
    # =========================================================================
    # Qwen family (ChatML-based, Alibaba) - OPEN
    # =========================================================================
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/QwQ-32B",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-Audio-7B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen-7B-Chat",

    # =========================================================================
    # Mistral family - OPEN
    # =========================================================================
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Codestral-22B-v0.1",
    "mistralai/Mistral-Nemo-Instruct-2407",

    # =========================================================================
    # Microsoft Phi family - OPEN
    # =========================================================================
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3.5-MoE-instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "microsoft/phi-4",

    # =========================================================================
    # DeepSeek family - OPEN
    # =========================================================================
    "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/DeepSeek-V2.5",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/deepseek-math-7b-instruct",

    # =========================================================================
    # Yi family (01.AI) - OPEN
    # =========================================================================
    "01-ai/Yi-6B-Chat",
    "01-ai/Yi-34B-Chat",
    "01-ai/Yi-1.5-9B-Chat",
    "01-ai/Yi-Coder-9B-Chat",
    "01-ai/Yi-VL-6B",

    # =========================================================================
    # HuggingFace models - OPEN
    # =========================================================================
    "HuggingFaceH4/zephyr-7b-beta",
    "HuggingFaceH4/zephyr-7b-alpha",
    "HuggingFaceH4/starchat2-15b-v0.1",
    "HuggingFaceH4/mistral-7b-sft-beta",
    "HuggingFaceM4/Idefics3-8B-Llama3",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM-1.7B-Instruct",

    # =========================================================================
    # Small/Efficient models - OPEN
    # =========================================================================
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "stabilityai/stablelm-2-zephyr-1_6b",
    "stabilityai/stablelm-zephyr-3b",
    "stabilityai/stable-code-instruct-3b",
    "Felladrin/Minueza-32M-UltraChat",
    "microsoft/DialoGPT-medium",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",

    # =========================================================================
    # NousResearch / Hermes - OPEN
    # =========================================================================
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "NousResearch/Hermes-2-Pro-Mistral-7B",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "NousResearch/Genstruct-7B",

    # =========================================================================
    # Teknium / OpenHermes - OPEN
    # =========================================================================
    "teknium/OpenHermes-2.5-Mistral-7B",
    "teknium/OpenHermes-2-Mistral-7B",

    # =========================================================================
    # OpenChat - OPEN
    # =========================================================================
    "openchat/openchat-3.5-0106",
    "openchat/openchat-3.6-8b-20240522",
    "openchat/openchat_3.5",

    # =========================================================================
    # Code specialized models - OPEN
    # =========================================================================
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "bigcode/starcoder2-15b-instruct-v0.1",
    "bigcode/starcoder2-7b-instruct-v0.1",
    "m-a-p/OpenCodeInterpreter-DS-6.7B",
    "WizardLM/WizardCoder-15B-V1.0",
    "deepseek-ai/deepseek-coder-1.3b-instruct",

    # =========================================================================
    # Multimodal / Vision-Language - OPEN
    # =========================================================================
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    "THUDM/cogvlm2-llama3-chat-19B",
    "allenai/Molmo-7B-D-0924",
    "Qwen/Qwen2-VL-2B-Instruct",

    # =========================================================================
    # IBM Granite - OPEN
    # =========================================================================
    "ibm-granite/granite-3.0-8b-instruct",
    "ibm-granite/granite-3.1-8b-instruct",
    "ibm-granite/granite-20b-code-instruct",
    "ibm-granite/granite-3.0-2b-instruct",

    # =========================================================================
    # Chinese models - OPEN
    # =========================================================================
    "THUDM/chatglm3-6b",
    "THUDM/glm-4-9b-chat",
    "internlm/internlm2-chat-7b",
    "internlm/internlm2_5-7b-chat",
    "FlagAlpha/Llama3-Chinese-8B-Instruct",
    "shenzhi-wang/Llama3-8B-Chinese-Chat",

    # =========================================================================
    # Japanese models - OPEN
    # =========================================================================
    "llm-jp/llm-jp-13b-instruct-full-dolly-ichikara_004_001_single-oasst-oasst2-v2.0",
    "elyza/Llama-3-ELYZA-JP-8B",
    "lightblue/suzume-llama-3-8B-multilingual",

    # =========================================================================
    # Korean models - OPEN
    # =========================================================================
    "beomi/Llama-3-Open-Ko-8B-Instruct-preview",
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    "maywell/Synatra-7B-v0.3-dpo",

    # =========================================================================
    # European language models - OPEN
    # =========================================================================
    "occiglot/occiglot-7b-eu5-instruct",
    "BSC-LT/salamandra-7b-instruct",
    "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct",
    "jphme/em_german_leo_mistral",
    "mhenrichsen/danskgpt-tiny-chat",
    "Unbabel/TowerInstruct-7B-v0.2",

    # =========================================================================
    # NVIDIA models - OPEN
    # =========================================================================
    "nvidia/Llama3-ChatQA-1.5-8B",
    "nvidia/Nemotron-Mini-4B-Instruct",
    "nvidia/Mistral-NeMo-Minitron-8B-Instruct",
    "nvidia/OpenMath2-Llama3.1-8B",

    # =========================================================================
    # Mamba / SSM models - OPEN
    # =========================================================================
    "state-spaces/mamba-2.8b-hf",
    "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-790m-hf",
    "CobraMamba/mamba-gpt-3b-v4",
    "Q-bert/Mamba-130M",
    "clibrain/mamba-2.8b-instruct-openhermes",

    # =========================================================================
    # RWKV models - OPEN
    # =========================================================================
    "RWKV/rwkv-5-world-3b",
    "RWKV/v6-Finch-1B6-HF",
    "RWKV/rwkv-6-world-1b6",
    "RWKV/rwkv-6-world-3b",

    # =========================================================================
    # Allen AI / OLMo - OPEN
    # =========================================================================
    "allenai/OLMo-7B-Instruct",
    "allenai/OLMo-2-7B-Instruct",
    "allenai/tulu-2-7b",
    "allenai/tulu-2-dpo-7b",
    "allenai/Molmo-7B-D-0924",

    # =========================================================================
    # Medical/Scientific models - OPEN
    # =========================================================================
    "BioMistral/BioMistral-7B",
    "epfl-llm/meditron-7b",

    # =========================================================================
    # Math models - OPEN
    # =========================================================================
    "AI-MO/NuminaMath-7B-TIR",
    "TIGER-Lab/MAmmoTH2-8B",

    # =========================================================================
    # Reasoning/Agent models - OPEN
    # =========================================================================
    "Nexusflow/Starling-LM-7B-beta",
    "berkeley-nest/Starling-LM-7B-alpha",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "NexaAIDev/Octopus-v2",

    # =========================================================================
    # Mixture of Experts - OPEN
    # =========================================================================
    "DiscoResearch/DiscoLM-mixtral-8x7b-v2",
    "cognitivecomputations/dolphin-2.7-mixtral-8x7b",
    "cognitivecomputations/dolphin-2.9.1-llama-3-8b",
    "cognitivecomputations/dolphin-2.9-llama3-8b",

    # =========================================================================
    # Intel models - OPEN
    # =========================================================================
    "Intel/neural-chat-7b-v3-3",
    "Intel/neural-chat-7b-v3-1",

    # =========================================================================
    # Function calling / Tool use - OPEN
    # =========================================================================
    "meetkai/functionary-small-v2.4",
    "meetkai/functionary-small-v2.5",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",

    # =========================================================================
    # Falcon - OPEN
    # =========================================================================
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-mamba-7b-instruct",

    # =========================================================================
    # Southeast Asian models - OPEN
    # =========================================================================
    "aisingapore/sea-lion-7b-instruct",
    "SeaLLMs/SeaLLMs-v3-7B-Chat",
    "sail/Sailor-7B-Chat",

    # =========================================================================
    # Orion / Starling models - OPEN
    # =========================================================================
    "OrionStarAI/Orion-14B-Chat",
    "Deci/DeciLM-7B-instruct",

    # =========================================================================
    # Creative / Roleplay - OPEN
    # =========================================================================
    "Sao10K/L3-8B-Stheno-v3.2",
    "dreamgen/opus-v1.2-llama-3-8b",

    # =========================================================================
    # Recent open models (2024-2025) - OPEN
    # =========================================================================
    "mlabonne/NeuralHermes-2.5-Mistral-7B",
    "Open-Orca/Mistral-7B-OpenOrca",
    "MediaTek-Research/Breeze-7B-Instruct-v1_0",
    "TokenBender/evolvedSeeker_1_3",

    # =========================================================================
    # Arcee AI - OPEN
    # =========================================================================
    "arcee-ai/Arcee-Lite",
    "arcee-ai/SuperNova-Lite",

    # =========================================================================
    # Abacus AI - OPEN
    # =========================================================================
    "abacusai/Smaug-Llama-3-70B-Instruct",
    "abacusai/Dracarys-Llama-3.1-70B-Instruct",

    # =========================================================================
    # Together AI - OPEN
    # =========================================================================
    "togethercomputer/StripedHyena-Nous-7B",
    "togethercomputer/Llama-3-8B-Instruct-Gradient-1048k",

    # =========================================================================
    # Reflection / Reasoning - OPEN
    # =========================================================================
    "mattshumer/Reflection-Llama-3.1-70B",
    "Gryphe/MythoMax-L2-13b",

    # =========================================================================
    # LMSys / Arena models - OPEN
    # =========================================================================
    "lmsys/fastchat-t5-3b-v1.0",

    # =========================================================================
    # X.AI / Grok (open weights) - OPEN
    # =========================================================================
    "xai-org/grok-1",

    # =========================================================================
    # WizardLM variants - OPEN
    # =========================================================================
    "WizardLM/WizardCoder-15B-V1.0",
    "WizardLM/WizardMath-7B-V1.1",

    # =========================================================================
    # Undi95 / merges - OPEN
    # =========================================================================
    "Undi95/Toppy-M-7B",
    "Undi95/MXLewd-L2-20B",

    # =========================================================================
    # More diverse fine-tunes - OPEN
    # =========================================================================
    "garage-bAInd/Platypus2-70B-instruct",
    "pankajmathur/orca_mini_v3_7b",
    "jondurbin/airoboros-l2-13b-3.1.1",
    "CalderaAI/30B-Lazarus",
    "ehartford/WizardLM-33B-V1.0-Uncensored",
    "ehartford/dolphin-2.0-mistral-7b",
    "NousResearch/Nous-Capybara-7B-V1.9",
    "Open-Orca/OpenOrca-Platypus2-13B",
    "migtissera/Synthia-7B-v1.2",
    "teknium/CollectiveCognition-v1.1-Mistral-7B",

    # =========================================================================
    # More Llama 3 fine-tunes - OPEN
    # =========================================================================
    "NousResearch/Meta-Llama-3-8B-Instruct",
    "cognitivecomputations/dolphin-2.9.2-qwen2-7b",
    "ajibawa-2023/Code-Llama-3-8B",

    # =========================================================================
    # More multimodal - OPEN
    # =========================================================================
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    "Qwen/Qwen2-VL-2B-Instruct",

    # =========================================================================
    # Upstage - OPEN
    # =========================================================================
    "upstage/SOLAR-10.7B-Instruct-v1.0",

    # =========================================================================
    # EleutherAI - OPEN
    # =========================================================================
    "EleutherAI/llemma_7b",
    "EleutherAI/gpt-neo-2.7B",

    # =========================================================================
    # More Chinese - OPEN
    # =========================================================================
    "FlagAlpha/Llama2-Chinese-7b-Chat",
    "LinkSoul/Chinese-Llama-2-7b",

    # =========================================================================
    # More code models - OPEN
    # =========================================================================
    "Phind/Phind-CodeLlama-34B-v2",
    "WizardLM/WizardCoder-Python-34B-V1.0",

    # =========================================================================
    # Instruct variants - OPEN
    # =========================================================================
    "garage-bAInd/Camel-Platypus2-13B",
    "timdettmers/guanaco-33b-merged",
    "YeungNLP/firefly-llama2-7b-chat",

    # =========================================================================
    # 2025 / Latest models - OPEN
    # =========================================================================
    # DeepSeek R1 series (Jan 2025)
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Zero",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",

    # Qwen 2.5 latest (2024-2025)
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",

    # Qwen3 (2025)
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",

    # Mistral latest (2024-2025)
    "mistralai/Mistral-Large-Instruct-2411",
    "mistralai/Mistral-Small-Instruct-2409",
    "mistralai/Pixtral-Large-Instruct-2411",

    # Microsoft Phi-4 (Dec 2024)
    "microsoft/phi-4-mini-instruct",

    # NVIDIA latest (2024-2025)
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "nvidia/Llama-3.1-Nemotron-51B-Instruct",
    "nvidia/Nemotron-4-340B-Instruct",

    # Allen AI latest (2024-2025)
    "allenai/OLMo-2-13B-Instruct",
    "allenai/tulu-3-8b",
    "allenai/tulu-3-70b",

    # SmolLM2 (Dec 2024)
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",

    # Falcon 3 (Dec 2024)
    "tiiuae/Falcon3-10B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "tiiuae/Falcon3-1B-Instruct",
    "tiiuae/Falcon3-Mamba-7B-Instruct",

    # IBM Granite latest (2024-2025)
    "ibm-granite/granite-3.1-2b-instruct",
    "ibm-granite/granite-3.2-8b-instruct",
    "ibm-granite/granite-3.2-2b-instruct",

    # Alibaba latest
    "alibaba-nlp/gte-Qwen2-7B-instruct",

    # Arcee latest (2024-2025)
    "arcee-ai/Arcee-Nova",
    "arcee-ai/Arcee-Spark",

    # Cohere Aya (multilingual)
    "CohereForAI/aya-expanse-32b",

    # GLM-4 latest
    "THUDM/glm-4-9b-chat-1m",

    # InternLM latest (2024-2025)
    "internlm/internlm2_5-20b-chat",
    "internlm/internlm3-8b-instruct",

    # Yi latest (2024-2025)
    "01-ai/Yi-1.5-34B-Chat",
    "01-ai/Yi-Lightning",

    # Gemma 2 open versions (via community)
    "unsloth/gemma-2-9b-it-bnb-4bit",

    # Llama 3.1/3.2/3.3 community fine-tunes
    "NousResearch/Hermes-3-Llama-3.2-3B",
    "cognitivecomputations/dolphin-2.9.4-llama3.1-8b",
    "Sao10K/L3.1-8B-Celeste-v1",

    # Code/Math specialized latest
    "Qwen/Qwen2.5-Math-72B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",

    # Vision models latest
    "Qwen/Qwen2-VL-72B-Instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "openbmb/MiniCPM-o-2_6",

    # Mamba 2 / Jamba
    "ai21labs/Jamba-v0.1",
    "ai21labs/AI21-Jamba-1.5-Large",
    "ai21labs/AI21-Jamba-1.5-Mini",

    # Snowflake
    "Snowflake/snowflake-arctic-instruct",

    # Databricks DBRX
    "databricks/dbrx-instruct",

    # xAI Grok
    "xai-org/grok-1",

    # Command R+ latest
    "CohereForAI/c4ai-command-r-plus-08-2024",

    # Mixtral community fine-tunes
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",

    # StableLM latest
    "stabilityai/stablelm-2-12b-chat",

    # Exaone (LG AI)
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",

    # Hugging Face models latest
    "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",

    # More community fine-tunes (popular on HF)
    "NousResearch/Hermes-3-Llama-3.1-70B",
    "Sao10K/Fimbulvetr-11B-v2",
    "mlabonne/Llama-3.1-70B-Instruct-lorablated",
    "arcee-ai/Llama-3.1-SuperNova-Lite",
    "Gryphe/Pantheon-RP-1.6-12b-Llama-3",

    # More Chinese models (2024-2025)
    "01-ai/Yi-34B-Chat-200K",
    "THUDM/chatglm-6b",
    "Baichuan2-7B-Chat",

    # Medical/Scientific (2024-2025)
    "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
    "epfl-llm/meditron-70b",

    # Latest merges and experiments
    "teknium/OpenHermes-2.5-Mistral-7B-16k",
    "NousResearch/Yarn-Mistral-7b-128k",
    "cognitivecomputations/WizardLM-2-8x22B",
]


def fetch_trending_models(token: str | None = None, limit: int = 200) -> list[str]:
    """Fetch trending text-generation models from HuggingFace API."""
    url = f"https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trending&limit={limit}"

    headers = {"User-Agent": "talu-template-fetcher/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=30) as response:
            models = json.loads(response.read().decode())
            return [m["id"] for m in models if "/" in m.get("id", "")]
    except (HTTPError, URLError, TimeoutError) as e:
        print(f"⚠ Failed to fetch trending models: {e}")
        return []


def fetch_recently_updated_models(token: str | None = None, limit: int = 200) -> list[str]:
    """Fetch recently updated text-generation models from HuggingFace API."""
    url = f"https://huggingface.co/api/models?pipeline_tag=text-generation&sort=lastModified&direction=-1&limit={limit}"

    headers = {"User-Agent": "talu-template-fetcher/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=30) as response:
            models = json.loads(response.read().decode())
            return [m["id"] for m in models if "/" in m.get("id", "")]
    except (HTTPError, URLError, TimeoutError) as e:
        print(f"⚠ Failed to fetch recent models: {e}")
        return []


def fetch_most_liked_models(token: str | None = None, limit: int = 200) -> list[str]:
    """Fetch most liked text-generation models from HuggingFace API."""
    url = f"https://huggingface.co/api/models?pipeline_tag=text-generation&sort=likes&direction=-1&limit={limit}"

    headers = {"User-Agent": "talu-template-fetcher/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=30) as response:
            models = json.loads(response.read().decode())
            return [m["id"] for m in models if "/" in m.get("id", "")]
    except (HTTPError, URLError, TimeoutError) as e:
        print(f"⚠ Failed to fetch liked models: {e}")
        return []


def fetch_tokenizer_config(model_id: str, token: str | None = None) -> dict | None:
    """Fetch tokenizer_config.json from HuggingFace."""
    url = f"https://huggingface.co/{model_id}/raw/main/tokenizer_config.json"

    headers = {"User-Agent": "talu-template-fetcher/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as response:
            content = response.read().decode()
            if not content.strip():
                return {"_error": "Empty response", "_model": model_id}
            return json.loads(content)
    except HTTPError as e:
        return {"_error": f"HTTP {e.code}", "_model": model_id}
    except (URLError, TimeoutError) as e:
        return {"_error": str(e), "_model": model_id}
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON: {e}", "_model": model_id}


def extract_chat_template(config: dict) -> str | None:
    """Extract chat template from tokenizer config."""
    if "_error" in config:
        return None

    template = config.get("chat_template")
    if not template:
        return None

    # Handle list format (multiple templates)
    if isinstance(template, list):
        # Look for default template
        for t in template:
            if isinstance(t, dict):
                if t.get("name") == "default" or not t.get("name"):
                    return t.get("template")
        # Fall back to first template
        if template and isinstance(template[0], dict):
            return template[0].get("template")
        return None

    return template if isinstance(template, str) else None


def analyze_template_with_jinja2(template: str) -> dict:
    """Analyze template using Jinja2's AST parser for accurate feature detection."""
    try:
        from jinja2 import Environment, nodes
    except ImportError:
        # Fall back to regex if jinja2 not available
        return analyze_template_regex(template)

    features = set()

    try:
        env = Environment()
        ast = env.parse(template)
    except Exception as e:
        # Template has syntax error, note it but continue
        return {"_parse_error": str(e)}

    def visit(node):
        """Recursively visit AST nodes and collect features."""
        node_type = type(node).__name__

        # Control flow
        if isinstance(node, nodes.For):
            features.add("for_loop")
            if node.else_:
                features.add("for_else")
            if node.test:
                features.add("for_if_filter")
            if node.recursive:
                features.add("for_recursive")
        elif isinstance(node, nodes.If):
            features.add("if_statement")
        elif isinstance(node, nodes.Macro):
            features.add("macro")
        elif isinstance(node, nodes.CallBlock):
            features.add("call_block")
        elif isinstance(node, nodes.FilterBlock):
            features.add("filter_block")
        elif isinstance(node, nodes.With):
            features.add("with_statement")
        elif isinstance(node, nodes.Block):
            features.add("block")
        elif isinstance(node, nodes.Extends):
            features.add("extends")
        elif isinstance(node, nodes.Include):
            features.add("include")
        elif isinstance(node, nodes.Import):
            features.add("import")
        elif isinstance(node, nodes.FromImport):
            features.add("from_import")

        # Assignment
        elif isinstance(node, nodes.Assign):
            features.add("set_statement")
        elif isinstance(node, nodes.AssignBlock):
            features.add("set_block")

        # Output
        elif isinstance(node, nodes.Output):
            pass  # Basic output
        elif isinstance(node, nodes.MarkSafe):
            features.add("mark_safe")

        # Expressions
        elif isinstance(node, nodes.Name):
            name = node.name
            # Check for common variables
            if name == "loop":
                features.add("loop_variable")
            elif name == "caller":
                features.add("caller")
            elif name == "varargs":
                features.add("varargs")
            elif name == "kwargs":
                features.add("kwargs")
        elif isinstance(node, nodes.Getattr):
            # Check for loop.X attributes
            if isinstance(node.node, nodes.Name) and node.node.name == "loop":
                features.add(f"loop_{node.attr}")
        elif isinstance(node, nodes.Getitem):
            features.add("getitem")
            # Check for slice
            if isinstance(node.arg, nodes.Slice):
                features.add("slice")
            # Check for negative index
            elif isinstance(node.arg, nodes.Neg):
                features.add("negative_index")
            elif isinstance(node.arg, nodes.Const) and isinstance(node.arg.value, int) and node.arg.value < 0:
                features.add("negative_index")

        # Calls
        elif isinstance(node, nodes.Call):
            if isinstance(node.node, nodes.Name):
                func_name = node.node.name
                features.add(f"func_{func_name}")
            elif isinstance(node.node, nodes.Getattr):
                method_name = node.node.attr
                features.add(f"method_{method_name}")

        # Filters
        elif isinstance(node, nodes.Filter):
            features.add(f"filter_{node.name}")

        # Tests
        elif isinstance(node, nodes.Test):
            features.add(f"test_{node.name}")

        # Operators
        elif isinstance(node, nodes.And):
            features.add("op_and")
        elif isinstance(node, nodes.Or):
            features.add("op_or")
        elif isinstance(node, nodes.Not):
            features.add("op_not")
        elif isinstance(node, nodes.Compare):
            features.add("comparison")
            for op in node.ops:
                op_type = type(op).__name__.lower()
                features.add(f"op_{op_type}")
        elif isinstance(node, nodes.Add):
            features.add("op_add")
        elif isinstance(node, nodes.Sub):
            features.add("op_sub")
        elif isinstance(node, nodes.Mul):
            features.add("op_mul")
        elif isinstance(node, nodes.Div):
            features.add("op_div")
        elif isinstance(node, nodes.FloorDiv):
            features.add("op_floordiv")
        elif isinstance(node, nodes.Mod):
            features.add("op_mod")
        elif isinstance(node, nodes.Pow):
            features.add("op_pow")
        elif isinstance(node, nodes.Concat):
            features.add("op_concat")
        elif isinstance(node, nodes.Neg):
            features.add("op_neg")
        elif isinstance(node, nodes.Pos):
            features.add("op_pos")

        # Conditionals
        elif isinstance(node, nodes.CondExpr):
            features.add("ternary")

        # Literals
        elif isinstance(node, nodes.List):
            features.add("list_literal")
        elif isinstance(node, nodes.Dict):
            features.add("dict_literal")
        elif isinstance(node, nodes.Tuple):
            features.add("tuple_literal")

        # Slice
        elif isinstance(node, nodes.Slice):
            features.add("slice")

        # Recurse into children
        for child in node.iter_child_nodes():
            visit(child)

    visit(ast)

    # Also detect some text patterns that AST doesn't capture well
    if "{#" in template:
        features.add("comments")
    if "{%-" in template or "-%}" in template:
        features.add("whitespace_control_block")
    if "{{-" in template or "-}}" in template:
        features.add("whitespace_control_output")

    # Detect special template patterns
    if "bos_token" in template:
        features.add("bos_token")
    if "eos_token" in template:
        features.add("eos_token")
    if "add_generation_prompt" in template:
        features.add("generation_prompt")
    if "tool" in template.lower():
        features.add("tool_related")
    if "<think>" in template.lower() or "thinking" in template.lower():
        features.add("thinking_tags")

    return {f: True for f in sorted(features)}


def analyze_template_regex(template: str) -> dict:
    """Fallback regex-based analysis if jinja2 not available."""
    import re
    features = {}

    patterns = {
        "for_loop": r"\{%[-\s]*for\s",
        "if_statement": r"\{%[-\s]*if\s",
        "set_statement": r"\{%[-\s]*set\s",
        "macro": r"\{%[-\s]*macro\s",
        "whitespace_control": r"\{%-|\{\{-",
    }

    for name, pattern in patterns.items():
        if re.search(pattern, template):
            features[name] = True

    return features


def get_cache_path(model_id: str) -> Path:
    """Get cache file path for a model."""
    cache_dir = Path(__file__).parent.parent / "cache" / "templates"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = model_id.replace("/", "--") + ".json"
    return cache_dir / filename


def fetch_all(token: str | None = None, force: bool = False, include_api: bool = True) -> dict:
    """Fetch all templates and save to cache."""
    results = {"success": [], "cached": [], "failed": [], "no_template": []}

    # Start with static list
    all_models = list(MODELS)

    # Add models from HuggingFace API if enabled
    if include_api:
        print("Fetching model lists from HuggingFace API...")

        # Fetch trending, recent, and most liked models
        trending = fetch_trending_models(token, limit=300)
        recent = fetch_recently_updated_models(token, limit=300)
        liked = fetch_most_liked_models(token, limit=300)

        print(f"  Trending: {len(trending)} models")
        print(f"  Recent: {len(recent)} models")
        print(f"  Liked: {len(liked)} models")

        # Merge all lists (preserving order, removing duplicates)
        seen = set(all_models)
        for model_list in [trending, recent, liked]:
            for model_id in model_list:
                if model_id not in seen:
                    all_models.append(model_id)
                    seen.add(model_id)

        print(f"  Total unique models: {len(all_models)}")
        print()

    for i, model_id in enumerate(all_models, 1):
        filepath = get_cache_path(model_id)

        # Skip if already cached (unless force)
        if filepath.exists() and not force:
            print(f"[{i}/{len(all_models)}] {model_id}... cached ✓")
            results["cached"].append(model_id)
            continue

        print(f"[{i}/{len(all_models)}] {model_id}...", end=" ", flush=True)

        config = fetch_tokenizer_config(model_id, token)
        template = extract_chat_template(config)

        if template:
            data = {
                "model_id": model_id,
                "template": template,
                "bos_token": config.get("bos_token", ""),
                "eos_token": config.get("eos_token", ""),
                "features": analyze_template_with_jinja2(template),
            }
            filepath.write_text(json.dumps(data, indent=2))
            print(f"✓ ({len(template)} chars, {len(data['features'])} features)")
            results["success"].append(model_id)
        elif "_error" in config:
            print(f"✗ {config['_error']}")
            results["failed"].append(model_id)
        else:
            print("⚠ no chat_template")
            results["no_template"].append(model_id)

    return results


def analyze_all():
    """Analyze all cached templates and print feature usage."""
    cache_dir = Path(__file__).parent.parent / "cache" / "templates"

    if not cache_dir.exists():
        print("No cached templates found. Run without --analyze first.")
        return

    feature_counts = Counter()
    templates = []
    parse_errors = []

    for filepath in sorted(cache_dir.glob("*.json")):
        data = json.loads(filepath.read_text())
        templates.append(data)

        features = data.get("features", {})
        if "_parse_error" in features:
            parse_errors.append((data["model_id"], features["_parse_error"]))
        else:
            for feature in features:
                feature_counts[feature] += 1

    print(f"\n{'=' * 70}")
    print(f"JINJA2 FEATURE USAGE ACROSS {len(templates)} TEMPLATES (AST analysis)")
    print(f"{'=' * 70}\n")

    # Group by category
    categories = {
        "Control Flow": ["for_loop", "for_else", "for_if_filter", "for_recursive",
                        "if_statement", "macro", "call_block", "filter_block",
                        "with_statement", "block", "extends", "include",
                        "import", "from_import"],
        "Assignment": ["set_statement", "set_block"],
        "Loop Variables": [k for k in feature_counts if k.startswith("loop_")],
        "Filters": sorted([k for k in feature_counts if k.startswith("filter_")]),
        "Functions": sorted([k for k in feature_counts if k.startswith("func_")]),
        "Methods": sorted([k for k in feature_counts if k.startswith("method_")]),
        "Tests": sorted([k for k in feature_counts if k.startswith("test_")]),
        "Operators": sorted([k for k in feature_counts if k.startswith("op_")]),
        "Access Patterns": ["getitem", "slice", "negative_index"],
        "Literals": ["list_literal", "dict_literal", "tuple_literal"],
        "Whitespace": ["whitespace_control_block", "whitespace_control_output"],
        "Template Patterns": ["bos_token", "eos_token", "generation_prompt",
                             "tool_related", "thinking_tags", "comments",
                             "ternary", "comparison"],
    }

    for category, features in categories.items():
        relevant = [(f, feature_counts[f]) for f in features if feature_counts[f] > 0]
        if relevant:
            print(f"{category}:")
            for feature, count in sorted(relevant, key=lambda x: -x[1]):
                pct = count * 100 // len(templates) if templates else 0
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"  {feature:30} {bar} {count:3} ({pct:2}%)")
            print()

    # List rare features
    print(f"\n{'=' * 70}")
    print("RARE FEATURES (used by 1-5 models) - potential compatibility gaps")
    print(f"{'=' * 70}\n")

    rare_features = [(f, c) for f, c in feature_counts.items() if 1 <= c <= 5]
    for feature, count in sorted(rare_features, key=lambda x: (x[1], x[0])):
        models = [t["model_id"] for t in templates if feature in t.get("features", {})]
        print(f"{feature} ({count}):")
        for m in models[:5]:
            print(f"  - {m}")
        print()

    # Parse errors
    if parse_errors:
        print(f"\n{'=' * 70}")
        print(f"PARSE ERRORS ({len(parse_errors)} templates)")
        print(f"{'=' * 70}\n")
        for model_id, error in parse_errors[:10]:
            print(f"  {model_id}: {error[:60]}...")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total templates:    {len(templates)}")
    print(f"  Parse errors:       {len(parse_errors)}")
    print(f"  Unique features:    {len(feature_counts)}")
    print(f"  Common (>50%):      {sum(1 for c in feature_counts.values() if c > len(templates)//2)}")
    print(f"  Rare (1-5 models):  {len(rare_features)}")


def main():
    parser = argparse.ArgumentParser(description="Fetch and analyze HuggingFace chat templates")
    parser.add_argument("--analyze", action="store_true", help="Analyze cached templates with Jinja2 AST")
    parser.add_argument("--force", action="store_true", help="Re-download all (ignore cache)")
    args = parser.parse_args()

    if args.analyze:
        analyze_all()
        return 0

    token = os.environ.get("HF_TOKEN")
    if token:
        print("✓ Using HF_TOKEN for authentication\n")
    else:
        print("⚠ No HF_TOKEN set - some gated models may fail\n")

    results = fetch_all(token, force=args.force)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  New downloads: {len(results['success']):3}")
    print(f"  From cache:    {len(results['cached']):3}")
    print(f"  No template:   {len(results['no_template']):3}")
    print(f"  Failed:        {len(results['failed']):3}")
    print(f"  Total models:  {len(MODELS):3}")
    print(f"\nTemplates saved to: cache/templates/")
    print(f"Run with --analyze to see feature usage statistics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
