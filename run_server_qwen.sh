#!/bin/bash

# ============================================================
# DMTS Qwen Streaming Server (CUDA + vLLM)
# ============================================================
# This starts dmts_qwen.py and can optionally boot a local vLLM service.
#
# Usage:
#   chmod +x run_server_qwen.sh
#   ./run_server_qwen.sh
#
# Optional env vars:
#   START_VLLM=true
#   CUDA_VISIBLE_DEVICES=0
#   TENSOR_PARALLEL_SIZE=1
#   GPU_MEMORY_UTILIZATION=0.90
#   MAX_MODEL_LEN=4096
#   VLLM_PORT=8000

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================================
# Model paths
# ============================================================
STORAGE_PATH="./models"
QWEN_MODEL="${STORAGE_PATH}/Qwen3-ASR-1.7B"
DIARIZATION_MODEL="${STORAGE_PATH}/XTTS-v2"

# Translation models
NLLB_600M="${STORAGE_PATH}/nllb-200-distilled-600M"
NLLB_3_3B="${STORAGE_PATH}/nllb-200-3.3B"
HUNYUAN_MODEL="${STORAGE_PATH}/Hunyuan-MT-7B"

# ============================================================
# Server configuration
# ============================================================
PORT=8890
TARGET_LANGUAGE="eng_Latn"
TRANSLATION_BACKEND="nllb"   # nllb | hybrid | hunyuan
ENABLE_TRANSLATION="true"
ENABLE_DIARIZATION="true"
ASR_LANGUAGE=""

# Qwen streaming knobs
QWEN_REALTIME_INTERVAL_MS=700
QWEN_FINAL_SILENCE_MS=1200
QWEN_MIN_REALTIME_AUDIO_MS=400
QWEN_MIN_FINAL_AUDIO_MS=600
QWEN_REALTIME_WINDOW_SEC=8.0

MAX_ACTIVE_SESSIONS=64
MAX_SESSION_QUEUE_CHUNKS=256
MAX_CONCURRENT_ASR_REQUESTS=8

# ============================================================
# vLLM CUDA scaling configuration
# ============================================================
START_VLLM="${START_VLLM:-false}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

QWEN_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}"

mkdir -p "${SCRIPT_DIR}/saved_audio"
mkdir -p "${SCRIPT_DIR}/logs"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DMTS Qwen Streaming Server${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Port: ${PORT}"
echo -e "  Qwen API: ${QWEN_API_BASE}"
echo -e "  Qwen model: ${QWEN_MODEL}"
echo -e "  Translation backend: ${TRANSLATION_BACKEND}"
echo -e "  Max active sessions: ${MAX_ACTIVE_SESSIONS}"
echo -e "  Max concurrent ASR reqs: ${MAX_CONCURRENT_ASR_REQUESTS}"
echo -e "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

TRANSLATION_ARGS=""
if [ "${ENABLE_TRANSLATION}" = "true" ]; then
  TRANSLATION_ARGS="--enable_translation --translation_target_language ${TARGET_LANGUAGE} --translation_backend ${TRANSLATION_BACKEND}"
  if [ "${TRANSLATION_BACKEND}" = "nllb" ]; then
    TRANSLATION_ARGS="${TRANSLATION_ARGS} --translation_model_realtime ${NLLB_600M} --translation_model_full ${NLLB_3_3B}"
  elif [ "${TRANSLATION_BACKEND}" = "hybrid" ]; then
    TRANSLATION_ARGS="${TRANSLATION_ARGS} --translation_model ${HUNYUAN_MODEL} --translation_model_realtime ${NLLB_600M} --translation_model_full ${NLLB_3_3B}"
  elif [ "${TRANSLATION_BACKEND}" = "hunyuan" ]; then
    TRANSLATION_ARGS="${TRANSLATION_ARGS} --translation_model ${HUNYUAN_MODEL} --skip_realtime_translation"
  fi
else
  TRANSLATION_ARGS="--disable_translation"
fi

DIARIZATION_ARGS="--disable_diarization"
if [ "${ENABLE_DIARIZATION}" = "true" ]; then
  DIARIZATION_ARGS="--enable_diarization --diarization_model_path ${DIARIZATION_MODEL}"
fi

cleanup() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo -e "${YELLOW}Stopping vLLM (PID ${VLLM_PID})...${NC}"
    kill "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT

if [ "${START_VLLM}" = "true" ]; then
  echo -e "${YELLOW}Starting local vLLM (CUDA enabled)...${NC}"
  export CUDA_VISIBLE_DEVICES

  # Requires: pip install vllm
  vllm serve "${QWEN_MODEL}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --dtype auto \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" &

  VLLM_PID=$!
  echo -e "${GREEN}vLLM started (PID ${VLLM_PID})${NC}"
  echo -e "${YELLOW}Waiting for vLLM warm-up...${NC}"
  sleep 10
fi

echo -e "${YELLOW}Starting dmts_qwen.py...${NC}"
cd "${SCRIPT_DIR}"
python dmts_qwen.py \
  --port "${PORT}" \
  --qwen_api_base "${QWEN_API_BASE}" \
  --qwen_model "${QWEN_MODEL}" \
  --asr_language "${ASR_LANGUAGE}" \
  --qwen_realtime_interval_ms "${QWEN_REALTIME_INTERVAL_MS}" \
  --qwen_final_silence_ms "${QWEN_FINAL_SILENCE_MS}" \
  --qwen_min_realtime_audio_ms "${QWEN_MIN_REALTIME_AUDIO_MS}" \
  --qwen_min_final_audio_ms "${QWEN_MIN_FINAL_AUDIO_MS}" \
  --qwen_realtime_window_sec "${QWEN_REALTIME_WINDOW_SEC}" \
  --max_active_sessions "${MAX_ACTIVE_SESSIONS}" \
  --max_session_queue_chunks "${MAX_SESSION_QUEUE_CHUNKS}" \
  --max_concurrent_asr_requests "${MAX_CONCURRENT_ASR_REQUESTS}" \
  ${TRANSLATION_ARGS} \
  ${DIARIZATION_ARGS} \
  --audio-log-dir "./saved_audio" \
  --transcription-log "./logs/transcript_qwen.log"
