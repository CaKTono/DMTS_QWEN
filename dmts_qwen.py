#!/usr/bin/env python3
"""
DMTS Qwen Streaming Server

A Qwen-native DMTS server path that preserves DMTS websocket contracts while
removing RealtimeSTT dependency. Audio is ingested in realtime, grouped into
per-client streaming sessions, and transcribed through a local Qwen/vLLM
compatible HTTP endpoint.

Key design points:
- DMTS-style endpoints: /, /control, /data, /stream
- Multi-client safe: one session state per websocket connection
- Shared ASR concurrency limiter for GPU protection
- Translation pipeline compatibility (NLLB/Hunyuan/Hybrid)
- Optional diarization pipeline compatibility
- Verification intentionally disabled in this path
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import aiohttp_cors
import httpx
import numpy as np
from aiohttp import WSMsgType, web
from scipy.signal import resample

from language_codes import normalize_language_code


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# Global runtime state
args_global: Optional[argparse.Namespace] = None
session_manager: Optional["QwenSessionManager"] = None
translation_manager: Any = None
target_translation_language: Optional[str] = None
shared_executor: Optional[ThreadPoolExecutor] = None
full_sentence_processor_thread: Optional["FullSentenceProcessorThread"] = None

# Connection registries
data_connections = set()
control_connections = set()
stream_connections = set()

# Async message buses
audio_queue: asyncio.Queue = asyncio.Queue()
translation_queue: asyncio.Queue = asyncio.Queue()

# Thread queue for final sentence packets
full_sentence_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# Logging flags
DEBUG_LOGGING = False
EXTENDED_LOGGING = False
LOG_INCOMING_CHUNKS = False


def debug_print(message: str) -> None:
    if DEBUG_LOGGING:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        thread_name = threading.current_thread().name
        print(f"[DEBUG][{timestamp}][{thread_name}] {message}", file=sys.stderr)


def preprocess_text(text: str) -> str:
    text = text.lstrip()
    if text.startswith("..."):
        text = text[3:]
    if text.endswith("...'."):
        text = text[:-1]
    if text.endswith("...'"):
        text = text[:-1]
    text = text.lstrip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def format_timestamp_ns(timestamp_ns: int) -> str:
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000
    dt = datetime.fromtimestamp(seconds)
    time_str = dt.strftime("%H:%M:%S")
    milliseconds = remainder_ns // 1_000_000
    return f"{time_str}.{milliseconds:03d}"


def decode_and_resample(audio_data: bytes, original_sample_rate: int, target_sample_rate: int) -> bytes:
    if original_sample_rate == target_sample_rate:
        return audio_data

    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    num_original_samples = len(audio_np)
    num_target_samples = int(num_original_samples * target_sample_rate / original_sample_rate)
    resampled_audio = resample(audio_np, num_target_samples)
    return resampled_audio.astype(np.int16).tobytes()


def pcm16_to_float32(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.array([], dtype=np.float32)
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0


def pcm16_to_wav_bytes(audio_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16 bytes as a WAV payload for HTTP audio APIs."""
    with io.BytesIO() as bio:
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        return bio.getvalue()


def map_asr_language_to_nllb(lang_code: Optional[str]) -> str:
    if not lang_code:
        return "eng_Latn"
    normalized = normalize_language_code(lang_code)
    return normalized or "eng_Latn"


@dataclass
class SessionState:
    session_id: str
    websocket: web.WebSocketResponse
    source: str
    sample_rate: int = 16000
    source_language: str = ""
    last_audio_monotonic: float = 0.0
    last_realtime_emit: float = 0.0
    last_realtime_text: str = ""
    utterance_audio: bytearray = field(default_factory=bytearray)
    rolling_audio: bytearray = field(default_factory=bytearray)
    queue: "asyncio.Queue[Tuple[bytes, int, Optional[str]]]" = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    closed: bool = False
    worker_task: Optional[asyncio.Task] = None


class QwenVLLMBackend:
    """Thin async HTTP client for a vLLM/OpenAI-compatible audio transcription service."""

    def __init__(self, base_url: str, model: str, timeout_sec: float, api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(timeout=self.timeout_sec, headers=headers)

    async def transcribe(
        self,
        audio_pcm16: bytes,
        sample_rate: int,
        language: Optional[str],
        prompt: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        if not audio_pcm16:
            return "", language

        wav_payload = pcm16_to_wav_bytes(audio_pcm16, sample_rate)
        files = {
            "file": ("audio.wav", wav_payload, "audio/wav"),
        }
        data = {
            "model": self.model,
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        # vLLM deployments often expose OpenAI-compatible speech endpoint.
        # If this endpoint differs in your deployment, update this one path.
        url = f"{self.base_url}/v1/audio/transcriptions"
        response = await self.client.post(url, files=files, data=data)
        response.raise_for_status()

        payload = response.json()
        text = payload.get("text", "") or ""
        detected_lang = payload.get("language")
        return text.strip(), detected_lang

    async def close(self) -> None:
        await self.client.aclose()


class QwenSessionManager:
    """Maintains per-client audio sessions and shared ASR concurrency controls."""

    def __init__(
        self,
        backend: QwenVLLMBackend,
        loop: asyncio.AbstractEventLoop,
        args: argparse.Namespace,
        executor: ThreadPoolExecutor,
    ) -> None:
        self.backend = backend
        self.loop = loop
        self.args = args
        self.executor = executor

        self.sessions: Dict[str, SessionState] = {}
        self.sessions_lock = asyncio.Lock()
        self.asr_semaphore = asyncio.Semaphore(args.max_concurrent_asr_requests)
        self.session_closed_event = asyncio.Event()

        self.sentence_index = 0
        self.index_lock = threading.Lock()

        # Tunables mutable through /control
        self.realtime_interval_ms = args.qwen_realtime_interval_ms
        self.final_silence_ms = args.qwen_final_silence_ms
        self.min_realtime_audio_ms = args.qwen_min_realtime_audio_ms
        self.min_final_audio_ms = args.qwen_min_final_audio_ms
        self.realtime_window_sec = args.qwen_realtime_window_sec
        self.asr_language = args.asr_language

    def next_sentence_index(self) -> int:
        with self.index_lock:
            current = self.sentence_index
            self.sentence_index += 1
            return current

    def tunable_parameters(self) -> Dict[str, Any]:
        return {
            "qwen_realtime_interval_ms": self.realtime_interval_ms,
            "qwen_final_silence_ms": self.final_silence_ms,
            "qwen_min_realtime_audio_ms": self.min_realtime_audio_ms,
            "qwen_min_final_audio_ms": self.min_final_audio_ms,
            "qwen_realtime_window_sec": self.realtime_window_sec,
            "asr_language": self.asr_language,
        }

    def set_tunable_parameter(self, name: str, value: Any) -> bool:
        if name == "qwen_realtime_interval_ms":
            self.realtime_interval_ms = int(value)
            return True
        if name == "qwen_final_silence_ms":
            self.final_silence_ms = int(value)
            return True
        if name == "qwen_min_realtime_audio_ms":
            self.min_realtime_audio_ms = int(value)
            return True
        if name == "qwen_min_final_audio_ms":
            self.min_final_audio_ms = int(value)
            return True
        if name == "qwen_realtime_window_sec":
            self.realtime_window_sec = float(value)
            return True
        if name == "asr_language":
            self.asr_language = str(value)
            return True
        return False

    async def create_session(self, ws: web.WebSocketResponse, source: str) -> str:
        async with self.sessions_lock:
            if len(self.sessions) >= self.args.max_active_sessions:
                raise RuntimeError(f"Session limit reached ({self.args.max_active_sessions})")

            session_id = uuid4().hex
            session = SessionState(
                session_id=session_id,
                websocket=ws,
                source=source,
                sample_rate=16000,
                source_language=self.args.asr_language or "",
                last_audio_monotonic=time.monotonic(),
                queue=asyncio.Queue(maxsize=self.args.max_session_queue_chunks),
            )
            session.worker_task = asyncio.create_task(self._session_worker(session), name=f"ASRSession-{session_id[:8]}")
            self.sessions[session_id] = session

        debug_print(f"Created ASR session {session_id} ({source}), total={len(self.sessions)}")
        return session_id

    async def remove_session(self, session_id: str) -> None:
        session: Optional[SessionState] = None
        async with self.sessions_lock:
            session = self.sessions.pop(session_id, None)

        if session is None:
            return

        session.closed = True
        await self._flush_session(session)

        if session.worker_task:
            session.worker_task.cancel()
            try:
                await session.worker_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(f"{bcolors.WARNING}Session worker shutdown error: {exc}{bcolors.ENDC}")

        debug_print(f"Removed ASR session {session_id}, total={len(self.sessions)}")

    async def feed_audio(self, session_id: str, chunk: bytes, sample_rate: int, language_hint: Optional[str] = None) -> None:
        session = self.sessions.get(session_id)
        if not session or session.closed:
            return

        if language_hint:
            session.source_language = language_hint

        if sample_rate <= 0:
            sample_rate = 16000

        if sample_rate != 16000:
            loop = asyncio.get_running_loop()
            chunk = await loop.run_in_executor(self.executor, decode_and_resample, chunk, sample_rate, 16000)
            sample_rate = 16000

        session.sample_rate = sample_rate
        session.last_audio_monotonic = time.monotonic()

        if session.queue.full():
            # Backpressure strategy: drop oldest chunk to preserve recency.
            try:
                session.queue.get_nowait()
                session.queue.task_done()
            except asyncio.QueueEmpty:
                pass

        await session.queue.put((chunk, sample_rate, language_hint))

    async def _transcribe_chunk(
        self,
        audio_pcm16: bytes,
        sample_rate: int,
        language: Optional[str],
        prompt: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        async with self.asr_semaphore:
            return await self.backend.transcribe(
                audio_pcm16=audio_pcm16,
                sample_rate=sample_rate,
                language=language,
                prompt=prompt,
            )

    async def _emit_realtime(self, session: SessionState, text: str, lang: Optional[str]) -> None:
        text = preprocess_text(text)
        if not text:
            return
        if text == session.last_realtime_text:
            return

        session.last_realtime_text = text
        if lang:
            session.source_language = lang

        source_lang_code = map_asr_language_to_nllb(session.source_language or self.asr_language)

        if args_global and args_global.enable_translation:
            await translation_queue.put(
                {
                    "type": "realtime",
                    "text": text,
                    "source_lang": source_lang_code,
                }
            )
        else:
            await audio_queue.put(json.dumps({"type": "realtime", "text": text}))

    async def _flush_session(self, session: SessionState) -> None:
        if not session.utterance_audio:
            return

        sample_rate = session.sample_rate or 16000
        total_samples = len(session.utterance_audio) // 2
        total_ms = (total_samples / sample_rate) * 1000.0
        if total_ms < self.min_final_audio_ms:
            session.utterance_audio.clear()
            return

        audio_bytes = bytes(session.utterance_audio)
        session.utterance_audio.clear()

        language = session.source_language or self.asr_language or None
        try:
            text, detected_lang = await self._transcribe_chunk(
                audio_pcm16=audio_bytes,
                sample_rate=sample_rate,
                language=language,
                prompt=self.args.qwen_final_prompt,
            )
        except Exception as exc:
            print(f"{bcolors.WARNING}Final transcription error ({session.session_id[:8]}): {exc}{bcolors.ENDC}")
            return

        text = preprocess_text(text)
        if not text:
            return

        if detected_lang:
            session.source_language = detected_lang

        source_lang_code = map_asr_language_to_nllb(session.source_language or self.asr_language)
        audio_float = pcm16_to_float32(audio_bytes)

        packet = {
            "index": self.next_sentence_index(),
            "text": text,
            "audio_buffer": audio_float,
            "source_lang": source_lang_code,
            "session_id": session.session_id,
        }
        full_sentence_queue.put(packet)

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n[{timestamp}] {bcolors.OKGREEN}Final sentence ({session.session_id[:8]}): {text}{bcolors.ENDC}")

    async def _session_worker(self, session: SessionState) -> None:
        sample_rate = 16000
        bytes_per_ms = (sample_rate * 2) / 1000.0

        try:
            while not session.closed:
                got_chunk = False
                try:
                    chunk, sample_rate, hint_lang = await asyncio.wait_for(session.queue.get(), timeout=0.10)
                    got_chunk = True
                    if hint_lang:
                        session.source_language = hint_lang

                    session.utterance_audio.extend(chunk)
                    session.rolling_audio.extend(chunk)

                    max_window_bytes = int(self.realtime_window_sec * sample_rate * 2)
                    if len(session.rolling_audio) > max_window_bytes:
                        del session.rolling_audio[:-max_window_bytes]

                    session.last_audio_monotonic = time.monotonic()
                    session.queue.task_done()
                except asyncio.TimeoutError:
                    pass

                now = time.monotonic()

                # Realtime emit cadence
                elapsed_realtime_ms = (now - session.last_realtime_emit) * 1000.0
                current_realtime_ms = len(session.rolling_audio) / bytes_per_ms if bytes_per_ms > 0 else 0
                if got_chunk and elapsed_realtime_ms >= self.realtime_interval_ms and current_realtime_ms >= self.min_realtime_audio_ms:
                    language = session.source_language or self.asr_language or None
                    try:
                        text, detected_lang = await self._transcribe_chunk(
                            audio_pcm16=bytes(session.rolling_audio),
                            sample_rate=sample_rate,
                            language=language,
                            prompt=self.args.qwen_realtime_prompt,
                        )
                        await self._emit_realtime(session, text, detected_lang)
                        session.last_realtime_emit = now
                    except Exception as exc:
                        print(f"{bcolors.WARNING}Realtime transcription error ({session.session_id[:8]}): {exc}{bcolors.ENDC}")

                # Utterance finalization on silence gap
                silence_ms = (now - session.last_audio_monotonic) * 1000.0
                if session.utterance_audio and silence_ms >= self.final_silence_ms:
                    await self._flush_session(session)
                    session.rolling_audio.clear()
                    session.last_realtime_text = ""

        except asyncio.CancelledError:
            pass
        finally:
            if session.utterance_audio:
                await self._flush_session(session)

    async def flush_all_sessions(self) -> None:
        sessions = list(self.sessions.values())
        for session in sessions:
            await self._flush_session(session)

    async def shutdown(self) -> None:
        sessions = list(self.sessions.keys())
        for session_id in sessions:
            await self.remove_session(session_id)


class FullSentenceProcessorThread(threading.Thread):
    """
    Processes finalized sentences, applies diarization, and emits DMTS-compatible
    diarization updates. Verification is intentionally disabled for dmts_qwen.
    """

    def __init__(
        self,
        full_sentence_queue_ref: "queue.Queue[Dict[str, Any]]",
        audio_queue_ref: asyncio.Queue,
        translation_queue_ref: asyncio.Queue,
        shared_executor_ref: ThreadPoolExecutor,
        runtime_args: argparse.Namespace,
        loop: asyncio.AbstractEventLoop,
        tts_model: Any,
    ) -> None:
        super().__init__(name="QwenFullSentenceProcessor")
        self.full_sentence_queue = full_sentence_queue_ref
        self.audio_queue = audio_queue_ref
        self.translation_queue = translation_queue_ref
        self.shared_executor = shared_executor_ref
        self.args = runtime_args
        self.loop = loop
        self.tts_model = tts_model

        self.stop_event = threading.Event()
        self.full_sentences = []
        self.sentence_speakers = []

    def stop(self) -> None:
        self.stop_event.set()

    def _get_speaker_embedding(self, audio_buffer: np.ndarray) -> Optional[np.ndarray]:
        if not self.args.enable_diarization or self.tts_model is None or audio_buffer is None or audio_buffer.size == 0:
            return None

        temp_filename = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_filename = temp_wav.name

            audio_int16 = (audio_buffer * 32767).astype(np.int16)
            with wave.open(temp_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())

            _, speaker_embedding = self.tts_model.get_conditioning_latents(
                audio_path=temp_filename,
                gpt_cond_len=30,
                max_ref_length=60,
            )
            return speaker_embedding.view(-1).cpu().detach().numpy()
        except Exception as exc:
            print(f"{bcolors.WARNING}Diarization embedding error: {exc}{bcolors.ENDC}")
            return None
        finally:
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)

    def _determine_optimal_cluster_count(self, embeddings_scaled: np.ndarray) -> int:
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.metrics import silhouette_score

        num_embeddings = len(embeddings_scaled)
        if num_embeddings <= 1:
            return 1

        kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings_scaled)
        distances = kmeans.transform(embeddings_scaled)
        avg_distance = np.mean(np.min(distances, axis=1))
        if avg_distance < self.args.diarization_speaker_threshold:
            return 1

        max_clusters = min(10, num_embeddings)
        range_clusters = range(2, max_clusters + 1)
        silhouette_scores = []

        for n_clusters in range_clusters:
            hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            cluster_labels = hc.fit_predict(embeddings_scaled)
            if 1 < len(set(cluster_labels)) < num_embeddings:
                score = silhouette_score(embeddings_scaled, cluster_labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)

        optimal_cluster_count = 2
        for i in range(1, len(silhouette_scores)):
            improvement = silhouette_scores[i] - silhouette_scores[i - 1]
            if improvement < self.args.diarization_silhouette_threshold:
                optimal_cluster_count = list(range_clusters)[i - 1]
                break
        return optimal_cluster_count

    def _process_speakers(self) -> None:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler

        embeddings = [emb for _, emb in self.full_sentences if emb is not None]
        if len(embeddings) == 0:
            self.sentence_speakers = []
            return

        embeddings_array = np.array(embeddings)
        if len(embeddings_array) == 1:
            self.sentence_speakers = [0]
            return

        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        optimal_n_clusters = self._determine_optimal_cluster_count(embeddings_scaled)

        hc = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage="ward")
        clusters = hc.fit_predict(embeddings_scaled)

        speaker_map = []
        embedding_idx = 0
        for _, emb in self.full_sentences:
            if emb is not None:
                speaker_map.append(int(clusters[embedding_idx]))
                embedding_idx += 1
            else:
                speaker_map.append(-1)
        self.sentence_speakers = speaker_map

    def _save_audio_file(self, filename: str, audio_bytes: bytes, framerate: int) -> None:
        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(framerate)
                wf.writeframes(audio_bytes)
            if EXTENDED_LOGGING:
                print(f"Saved audio to: {filename}")
        except Exception as exc:
            print(f"{bcolors.WARNING}Background audio save failed: {exc}{bcolors.ENDC}")

    def _process_data_packet(self, data_packet: Dict[str, Any]) -> None:
        sentence_index = data_packet.get("index")
        sentence_text = data_packet.get("text", "")
        audio_buffer = data_packet.get("audio_buffer")
        source_lang = data_packet.get("source_lang", "eng_Latn")

        if self.args.audio_log_dir and isinstance(audio_buffer, np.ndarray) and audio_buffer.size > 0:
            try:
                os.makedirs(self.args.audio_log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(self.args.audio_log_dir, f"transcription_{sentence_index}_{timestamp}.wav")
                audio_int16 = (audio_buffer * 32767).astype(np.int16)
                self.shared_executor.submit(self._save_audio_file, filename, audio_int16.tobytes(), 16000)
            except Exception as exc:
                print(f"{bcolors.WARNING}Could not schedule audio save: {exc}{bcolors.ENDC}")

        old_assignments = list(self.sentence_speakers)
        if self.args.enable_diarization:
            embedding = self._get_speaker_embedding(audio_buffer)
            self.full_sentences.append((sentence_text, embedding))
            self._process_speakers()
        else:
            self.full_sentences.append((sentence_text, None))
            self.sentence_speakers.append(-1)

        updates = []
        for i in range(len(old_assignments)):
            if old_assignments[i] != self.sentence_speakers[i]:
                updates.append({"index": i, "speaker_id": int(self.sentence_speakers[i])})

        new_sentence_speaker_id = int(self.sentence_speakers[-1]) if self.sentence_speakers else -1

        message_data = {
            "type": "diarization_update",
            "new_sentence": {
                "index": sentence_index,
                "text": sentence_text,
                "speaker_id": new_sentence_speaker_id,
            },
            "updates": updates,
        }

        if self.args.enable_translation:
            job = {
                "type": "diarization_update",
                "source_lang": source_lang,
                "data": message_data,
            }
            asyncio.run_coroutine_threadsafe(self.translation_queue.put(job), self.loop)
        else:
            asyncio.run_coroutine_threadsafe(self.audio_queue.put(json.dumps(message_data)), self.loop)

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"\r[{timestamp}] {bcolors.BOLD}Speaker {new_sentence_speaker_id} | Sentence:{bcolors.ENDC} "
            f"{bcolors.OKGREEN}{sentence_text}{bcolors.ENDC}",
            end="",
        )
        if updates:
            print(f" {bcolors.WARNING}({len(updates)} corrections sent){bcolors.ENDC}")

    def run(self) -> None:
        debug_print("FullSentenceProcessorThread started")
        while not self.stop_event.is_set():
            try:
                packet = self.full_sentence_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_data_packet(packet)
            finally:
                self.full_sentence_queue.task_done()
        debug_print("FullSentenceProcessorThread stopped")


def initialize_tts_model(args: argparse.Namespace) -> Any:
    if not args.enable_diarization:
        return None
    if not args.diarization_model_path:
        print(f"{bcolors.WARNING}Diarization enabled but model path missing. Disabling diarization.{bcolors.ENDC}")
        args.enable_diarization = False
        return None

    print(f"{bcolors.OKCYAN}Initializing diarization model from: {args.diarization_model_path}{bcolors.ENDC}")
    try:
        import torch
        from TTS.config import load_config
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models import setup_model as setup_tts_model
        from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig

        config = load_config(os.path.join(args.diarization_model_path, "config.json"))
        model = setup_tts_model(config)

        try:
            torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
        except AttributeError:
            pass

        model.load_checkpoint(config, checkpoint_dir=args.diarization_model_path, eval=True)
        model.to(torch.device("cpu"))
        print(f"{bcolors.OKGREEN}Diarization model initialized on CPU.{bcolors.ENDC}")
        return model
    except Exception as exc:
        print(f"{bcolors.WARNING}Failed to initialize diarization model, disabling diarization: {exc}{bcolors.ENDC}")
        args.enable_diarization = False
        return None


async def handle_index(request: web.Request) -> web.Response:
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "index.html")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")
    except FileNotFoundError:
        return web.Response(status=404, text="Error: index.html not found")


async def broadcast_audio_messages(log_filename: Optional[str] = None) -> None:
    log_file = None
    try:
        if log_filename:
            log_file = open(log_filename, "a", encoding="utf-8")
            print(f"{bcolors.OKGREEN}Logging transcription events to: {log_filename}{bcolors.ENDC}")

        while True:
            message_str = await audio_queue.get()
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            try:
                message_data = json.loads(message_str)
                message_data["server_timestamp"] = timestamp
                final_message = json.dumps(message_data)
            except json.JSONDecodeError:
                final_message = message_str

            if log_file:
                log_file.write(final_message + "\n")
                log_file.flush()

            if data_connections:
                for conn in list(data_connections):
                    try:
                        await conn.send_str(final_message)
                    except Exception as exc:
                        print(f"{bcolors.WARNING}Broadcast send failed, dropping client: {exc}{bcolors.ENDC}")
                        data_connections.discard(conn)
    except asyncio.CancelledError:
        print(f"{bcolors.OKCYAN}Broadcast task cancelled.{bcolors.ENDC}")
    finally:
        if log_file:
            log_file.close()


async def translation_processor_task(executor: ThreadPoolExecutor, loop: asyncio.AbstractEventLoop) -> None:
    global target_translation_language, translation_manager

    while True:
        job = await translation_queue.get()

        if not target_translation_language or not translation_manager:
            if job["type"] == "diarization_update":
                await audio_queue.put(json.dumps(job["data"]))
            else:
                await audio_queue.put(json.dumps(job))
            continue

        if job["type"] == "diarization_update":
            original_data = job["data"]
            new_sentence_text = original_data["new_sentence"]["text"]
            source_language = job["source_lang"]
            try:
                translated_text = await loop.run_in_executor(
                    executor,
                    translation_manager.translate,
                    new_sentence_text,
                    source_language,
                    target_translation_language,
                    "full",
                )
            except Exception as exc:
                print(f"{bcolors.WARNING}Translation error: {exc}{bcolors.ENDC}")
                translated_text = "[Translation Error]"

            final_message_data = original_data
            final_message_data["new_sentence"]["translation"] = {"text": translated_text}

            if "updates" in final_message_data:
                for update in final_message_data["updates"]:
                    if "text" in update and update["text"]:
                        try:
                            update_translation = await loop.run_in_executor(
                                executor,
                                translation_manager.translate,
                                update["text"],
                                source_language,
                                target_translation_language,
                                "full",
                            )
                            update["translation"] = {"text": update_translation}
                        except Exception:
                            update["translation"] = {"text": "[Translation Error]"}

            await audio_queue.put(json.dumps(final_message_data))
            continue

        if job["type"] == "realtime":
            if args_global and args_global.skip_realtime_translation:
                await audio_queue.put(json.dumps({"type": "realtime", "text": job["text"]}))
                continue

            source_language = job["source_lang"]
            try:
                translated_text = await loop.run_in_executor(
                    executor,
                    translation_manager.translate,
                    job["text"],
                    source_language,
                    target_translation_language,
                    "realtime",
                )
            except Exception as exc:
                print(f"{bcolors.WARNING}Realtime translation error: {exc}{bcolors.ENDC}")
                translated_text = "[Translation Error]"

            message_data = {
                "type": "realtime",
                "text": job["text"],
                "translation": {"language": target_translation_language, "text": translated_text},
            }
            await audio_queue.put(json.dumps(message_data))


async def control_handler(request: web.Request) -> web.WebSocketResponse:
    global target_translation_language

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    control_connections.add(ws)

    print(f"{bcolors.OKGREEN}Control client connected{bcolors.ENDC}")

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue

            try:
                command_data = json.loads(msg.data)
            except json.JSONDecodeError:
                await ws.send_json({"status": "error", "message": "Invalid JSON command"})
                continue

            command = command_data.get("command")

            if command == "set_translation_language":
                if not args_global.enable_translation:
                    await ws.send_json({"status": "error", "message": "Translation is disabled"})
                    continue

                lang_code = command_data.get("language", "")
                if not lang_code:
                    target_translation_language = None
                    await ws.send_json({"status": "success", "message": "Translation disabled"})
                else:
                    normalized_lang = normalize_language_code(lang_code)
                    target_translation_language = normalized_lang
                    await ws.send_json(
                        {
                            "status": "success",
                            "message": f"Translation target set to {normalized_lang}",
                            "language": normalized_lang,
                        }
                    )

            elif command == "set_parameter":
                parameter = command_data.get("parameter")
                value = command_data.get("value")
                ok = session_manager.set_tunable_parameter(parameter, value)
                if ok:
                    await ws.send_json({"status": "success", "message": f"Parameter {parameter} set to {value}"})
                else:
                    await ws.send_json({"status": "error", "message": f"Unsupported parameter: {parameter}"})

            elif command == "get_parameter":
                parameter = command_data.get("parameter")
                params = session_manager.tunable_parameters()
                if parameter in params:
                    await ws.send_json({"status": "success", "parameter": parameter, "value": params[parameter]})
                else:
                    await ws.send_json({"status": "error", "message": f"Unsupported parameter: {parameter}"})

            elif command == "call_method":
                method_name = command_data.get("method")
                if method_name == "flush_all_sessions":
                    await session_manager.flush_all_sessions()
                    await ws.send_json({"status": "success", "message": "All sessions flushed"})
                elif method_name == "get_status":
                    await ws.send_json(
                        {
                            "status": "success",
                            "active_sessions": len(session_manager.sessions),
                            "max_active_sessions": args_global.max_active_sessions,
                            "max_concurrent_asr_requests": args_global.max_concurrent_asr_requests,
                            "target_translation_language": target_translation_language,
                        }
                    )
                else:
                    await ws.send_json({"status": "error", "message": f"Unsupported method: {method_name}"})

            else:
                await ws.send_json({"status": "error", "message": f"Unknown command: {command}"})

    except Exception as exc:
        print(f"{bcolors.WARNING}Control handler error: {exc}{bcolors.ENDC}")
    finally:
        control_connections.discard(ws)
        print(f"{bcolors.WARNING}Control client disconnected.{bcolors.ENDC}")

    return ws


async def data_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    data_connections.add(ws)

    try:
        session_id = await session_manager.create_session(ws, source="data")
    except RuntimeError as exc:
        await ws.send_json({"status": "error", "message": str(exc)})
        await ws.close()
        data_connections.discard(ws)
        return ws

    print(f"{bcolors.OKGREEN}Data client connected (session={session_id[:8]}){bcolors.ENDC}")

    try:
        async for msg in ws:
            if msg.type != WSMsgType.BINARY:
                continue

            message = msg.data
            if len(message) < 4:
                continue

            metadata_length = int.from_bytes(message[:4], byteorder="little")
            metadata_json = message[4 : 4 + metadata_length].decode("utf-8")
            metadata = json.loads(metadata_json)

            sample_rate = int(metadata.get("sampleRate", 16000))
            source_language = metadata.get("language")

            if "server_sent_to_stt" in metadata:
                stt_received_ns = time.time_ns()
                metadata["stt_received"] = stt_received_ns
                metadata["stt_received_formatted"] = format_timestamp_ns(stt_received_ns)

            chunk = message[4 + metadata_length :]

            if LOG_INCOMING_CHUNKS:
                print(".", end="", flush=True)

            await session_manager.feed_audio(session_id, chunk, sample_rate, source_language)

    except Exception as exc:
        print(f"{bcolors.WARNING}Data handler error: {exc}{bcolors.ENDC}")
    finally:
        await session_manager.remove_session(session_id)
        data_connections.discard(ws)
        print(f"{bcolors.WARNING}Data client disconnected (session={session_id[:8]}).{bcolors.ENDC}")

    return ws


async def simple_data_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    data_connections.add(ws)
    stream_connections.add(ws)

    try:
        session_id = await session_manager.create_session(ws, source="stream")
    except RuntimeError as exc:
        await ws.send_json({"status": "error", "message": str(exc)})
        await ws.close()
        data_connections.discard(ws)
        stream_connections.discard(ws)
        return ws

    print(f"{bcolors.OKGREEN}External client connected to /stream (session={session_id[:8]}){bcolors.ENDC}")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                await session_manager.feed_audio(session_id, msg.data, 16000, None)
            elif msg.type == WSMsgType.ERROR:
                print(f"{bcolors.WARNING}Stream websocket error: {ws.exception()}{bcolors.ENDC}")
    except Exception as exc:
        print(f"{bcolors.WARNING}Stream handler error: {exc}{bcolors.ENDC}")
    finally:
        await session_manager.remove_session(session_id)
        data_connections.discard(ws)
        stream_connections.discard(ws)
        print(f"{bcolors.WARNING}External client disconnected from /stream (session={session_id[:8]}).{bcolors.ENDC}")

    return ws


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start DMTS Qwen streaming server")

    # Server
    parser.add_argument("--port", type=int, default=8890, help="HTTP/WebSocket port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("--debug_websockets", action="store_true", help="Enable websockets debug logs")
    parser.add_argument("--use_extended_logging", action="store_true", help="Enable verbose logs")
    parser.add_argument("--logchunks", action="store_true", help="Print incoming chunk dots")

    # Qwen/vLLM ASR
    parser.add_argument("--qwen_api_base", type=str, default="http://127.0.0.1:8000", help="Base URL of local Qwen/vLLM service")
    parser.add_argument("--qwen_api_key", type=str, default="", help="Optional API key for ASR service")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-ASR-1.7B", help="Model name expected by ASR service")
    parser.add_argument("--qwen_timeout_sec", type=float, default=90.0, help="HTTP timeout for ASR requests")
    parser.add_argument("--qwen_realtime_prompt", type=str, default="", help="Prompt for realtime transcription")
    parser.add_argument("--qwen_final_prompt", type=str, default="", help="Prompt for final transcription")

    # Streaming controls
    parser.add_argument("--asr_language", type=str, default="", help="Optional source language hint for ASR")
    parser.add_argument("--qwen_realtime_interval_ms", type=int, default=700, help="Realtime transcription cadence")
    parser.add_argument("--qwen_final_silence_ms", type=int, default=1200, help="Silence to finalize sentence")
    parser.add_argument("--qwen_min_realtime_audio_ms", type=int, default=400, help="Min audio needed for realtime call")
    parser.add_argument("--qwen_min_final_audio_ms", type=int, default=600, help="Min audio needed for final sentence")
    parser.add_argument("--qwen_realtime_window_sec", type=float, default=8.0, help="Rolling audio window for realtime")

    # Scalability
    parser.add_argument("--max_active_sessions", type=int, default=64, help="Max concurrent websocket ASR sessions")
    parser.add_argument("--max_session_queue_chunks", type=int, default=256, help="Per-session queued chunks before dropping oldest")
    parser.add_argument("--max_concurrent_asr_requests", type=int, default=8, help="Global concurrent ASR requests")

    # Logging artifacts
    parser.add_argument("--audio-log-dir", type=str, default=None, help="Directory to store finalized sentence audio")
    parser.add_argument("--transcription-log", type=str, default=None, help="JSONL log for outbound events")

    # Translation
    parser.add_argument("--enable_translation", dest="enable_translation", action="store_true", default=True)
    parser.add_argument("--disable_translation", dest="enable_translation", action="store_false")
    parser.add_argument("--translation_target_language", type=str, default="eng_Latn")
    parser.add_argument(
        "--translation_backend",
        type=str,
        default="nllb",
        choices=["nllb", "hunyuan", "hybrid"],
        help='Translation backend: "nllb", "hunyuan", or "hybrid"',
    )
    parser.add_argument("--translation_model", type=str, default="tencent/Hunyuan-MT-7B")
    parser.add_argument("--translation_model_realtime", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--translation_model_full", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--translation_gpu_device", type=int, default=0)
    parser.add_argument("--translation_load_in_8bit", action="store_true", default=False)
    parser.add_argument("--skip_realtime_translation", action="store_true", default=False)

    # Diarization (optional)
    parser.add_argument("--enable_diarization", dest="enable_diarization", action="store_true", default=True)
    parser.add_argument("--disable_diarization", dest="enable_diarization", action="store_false")
    parser.add_argument("--diarization_model_path", type=str, default="")
    parser.add_argument("--diarization_speaker_threshold", type=float, default=17.0)
    parser.add_argument("--diarization_silhouette_threshold", type=float, default=0.0001)

    # Compatibility placeholder (verification disabled)
    parser.add_argument("--enable_verification", action="store_true", default=False, help="No-op for dmts_qwen")

    return parser.parse_args()


async def main_async() -> None:
    global args_global, translation_manager, target_translation_language
    global shared_executor, full_sentence_processor_thread, session_manager
    global DEBUG_LOGGING, EXTENDED_LOGGING, LOG_INCOMING_CHUNKS

    args = parse_arguments()
    args_global = args

    DEBUG_LOGGING = args.debug
    EXTENDED_LOGGING = args.use_extended_logging
    LOG_INCOMING_CHUNKS = args.logchunks

    if args.enable_verification:
        print(f"{bcolors.WARNING}Verification is disabled in dmts_qwen; ignoring --enable_verification.{bcolors.ENDC}")

    ws_logger = logging.getLogger("websockets")
    if args.debug_websockets:
        ws_logger.setLevel(logging.DEBUG)
        ws_logger.propagate = False
    else:
        ws_logger.setLevel(logging.WARNING)
        ws_logger.propagate = True

    loop = asyncio.get_event_loop()
    shared_executor = ThreadPoolExecutor(max_workers=12, thread_name_prefix="QwenJobExecutor")

    # Optional diarization model
    tts_model = initialize_tts_model(args)

    # Optional translation manager
    if args.enable_translation:
        print(f"{bcolors.OKCYAN}Translation enabled using backend: {args.translation_backend}{bcolors.ENDC}")
        if args.translation_backend == "hunyuan":
            from translation.manager_hunyuan import HunyuanTranslationManager

            translation_manager = HunyuanTranslationManager(
                model_path=args.translation_model,
                device="cuda",
                load_in_8bit=args.translation_load_in_8bit,
                gpu_device_index=args.translation_gpu_device,
            )
        elif args.translation_backend == "hybrid":
            from translation.manager_hybrid import HybridTranslationManager

            translation_manager = HybridTranslationManager(
                nllb_realtime_model_path=args.translation_model_realtime,
                nllb_full_model_path=args.translation_model_full,
                hunyuan_model_path=args.translation_model,
                device="cuda",
                hunyuan_load_in_8bit=args.translation_load_in_8bit,
                hunyuan_gpu_device=args.translation_gpu_device,
            )
        else:
            from translation.manager_nllb import TranslationManager

            translation_manager = TranslationManager(
                args.translation_model_realtime,
                args.translation_model_full,
                "cuda",
            )

        target_translation_language = normalize_language_code(args.translation_target_language)
        asyncio.create_task(translation_processor_task(shared_executor, loop))

    # ASR backend + session manager
    backend = QwenVLLMBackend(
        base_url=args.qwen_api_base,
        model=args.qwen_model,
        timeout_sec=args.qwen_timeout_sec,
        api_key=args.qwen_api_key,
    )
    session_manager = QwenSessionManager(backend, loop, args, shared_executor)

    # Final sentence processing thread
    full_sentence_processor_thread = FullSentenceProcessorThread(
        full_sentence_queue_ref=full_sentence_queue,
        audio_queue_ref=audio_queue,
        translation_queue_ref=translation_queue,
        shared_executor_ref=shared_executor,
        runtime_args=args,
        loop=loop,
        tts_model=tts_model,
    )
    full_sentence_processor_thread.start()

    # aiohttp app
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/control", control_handler)
    app.router.add_get("/data", data_handler)
    app.router.add_get("/stream", simple_data_handler)

    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        },
    )
    for route in list(app.router.routes()):
        cors.add(route)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)

    broadcast_task = None
    try:
        await site.start()
        print(f"{bcolors.OKGREEN}dmts_qwen server started: http://localhost:{args.port}{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Endpoints: /control /data /stream{bcolors.ENDC}")
        print(
            f"{bcolors.OKBLUE}ASR backend: {args.qwen_api_base} | model={args.qwen_model} | "
            f"max_sessions={args.max_active_sessions} | max_concurrent_asr={args.max_concurrent_asr_requests}{bcolors.ENDC}"
        )

        broadcast_task = asyncio.create_task(broadcast_audio_messages(args.transcription_log))
        await asyncio.Event().wait()

    except asyncio.CancelledError:
        pass
    finally:
        if broadcast_task:
            broadcast_task.cancel()
            await asyncio.gather(broadcast_task, return_exceptions=True)

        if session_manager:
            await session_manager.shutdown()
            await backend.close()

        if full_sentence_processor_thread and full_sentence_processor_thread.is_alive():
            full_sentence_processor_thread.stop()
            full_sentence_processor_thread.join(timeout=2.0)

        if shared_executor:
            shared_executor.shutdown(wait=True)

        await runner.cleanup()


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}Server interrupted by user.{bcolors.ENDC}")


if __name__ == "__main__":
    main()
