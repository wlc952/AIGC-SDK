#===----------------------------------------------------------------------===#
#
# Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from typing import Callable, Optional, TextIO, Union, List
from dataclasses import dataclass
from functools import lru_cache
from subprocess import CalledProcessError, run
import os
import sys
import zlib
import json
import numpy as np
import numba
import torch
import torch.nn.functional as F
import warnings

from .tokenizer import Tokenizer


def exact_div(x, y):
    assert x % y == 0
    return x // y


def fp16_cast(arr:np.ndarray):
  if arr.dtype == np.float16:
    return arr.view(np.uint16)
  else:
    return arr


def uint16_to_fp16(arr: np.ndarray):
    if arr.dtype == np.uint16:
        return arr.view(np.float16)
    else:
        return arr

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = [80, 128]
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS[0]) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels in N_MELS, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


@lru_cache(maxsize=None)
def mel_filters_np(n_mels: int = N_MELS[0]) -> np.ndarray:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:
    """
    assert n_mels in N_MELS, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    ) as f:
        return f[f"mel_{n_mels}"]


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS[0],
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        # replaces any character not representable using the system default encoding with an '?',
        # avoiding UnicodeEncodeError (https://github.com/openai/whisper/discussions/729).
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        # utf-8 can encode any Unicode code point, so no need to do the round-trip encoding
        return string


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, result: dict, audio_path: str, options: dict):
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        output_path = os.path.join(
            self.output_dir, audio_basename + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options)

    def write_result(self, result: dict, file: TextIO, options: dict):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: TextIO, options: dict):
        for segment in result["segments"]:
            print(segment["text"].strip(), file=file, flush=True)


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(self, result: dict, options: dict):
        raw_max_line_width: Optional[int] = options["max_line_width"]
        max_line_count: Optional[int] = options["max_line_count"]
        highlight_words: bool = options["highlight_words"]
        max_line_width = 1000 if raw_max_line_width is None else raw_max_line_width
        preserve_segments = max_line_count is None or raw_max_line_width is None

        def iterate_subtitles():
            line = ""
            line_len = 0
            line_count = 1
            # the next subtitle to yield (a list of word timings)
            subtitle: List[dict] = []
            last_words: List[dict] = []
            for segment in result["segments"]:
                for word in segment["words"]:
                    word_len = len(word["word"])
                    if (
                        line_len + word_len >= max_line_width
                        and line_count >= max_line_count
                    ):
                        # We're over the limit on both line length and line count
                        # yield the current subtitle
                        yield subtitle
                        subtitle = []
                        line_len = 0
                        line_count = 1
                    elif line_len + word_len >= max_line_width:
                        # We're over the line length limit but not the line count limit
                        # continue to the next line
                        line_len = 0
                        line_count += 1
                    
                    line_len += word_len
                    subtitle.append(word)
                    
            if len(subtitle) > 0:
                yield subtitle

        if "words" in result["segments"][0]:
            for subtitle in iterate_subtitles():
                subtitle_start = self.format_timestamp(subtitle[0]["start"])
                subtitle_end = self.format_timestamp(subtitle[-1]["end"])
                if highlight_words:
                    formatted_words = []
                    for word in subtitle:
                        word_start = self.format_timestamp(word["start"])
                        word_end = self.format_timestamp(word["end"])
                        formatted_words.append(f"<{word_start}>{word['word']}</{word_end}>")
                    subtitle_text = "".join(formatted_words)
                else:
                    subtitle_text = " ".join([word["word"] for word in subtitle])
                yield subtitle_start, subtitle_end, subtitle_text
        else:
            for segment in result["segments"]:
                segment_start = self.format_timestamp(segment["start"])
                segment_end = self.format_timestamp(segment["end"])
                segment_text = segment["text"].strip().replace("-->", "->")
                yield segment_start, segment_end, segment_text

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(self, result: dict, file: TextIO, options: dict):
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result, options):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(self, result: dict, file: TextIO, options: dict):
        for i, (start, end, text) in enumerate(
            self.iterate_result(result, options), start=1
        ):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(self, result: dict, file: TextIO, options: dict):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(segment["text"].strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO, options: dict):
        json.dump(result, file)


def get_writer(
    output_format: str, output_dir: str
) -> Callable[[dict, TextIO, dict], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(result: dict, file: TextIO, options: dict):
            for writer in all_writers:
                writer(result, file, options)

        return write_all

    return writers[output_format](output_dir)


def median_filter(x: torch.Tensor, filter_width: int):
    """Apply a median filter of width `filter_width` along the last dimension of `x`"""
    pad_width = filter_width // 2
    if x.shape[-1] <= pad_width:
        # F.pad requires the padding width to be smaller than the input dimension
        return x

    if (ndim := x.ndim) <= 2:
        # `F.pad` does not support 1D or 2D inputs for reflect padding but supports 3D and 4D
        x = x[None, None, :]

    assert (
        filter_width > 0 and filter_width % 2 == 1
    ), "`filter_width` should be an odd number"

    result = None
    x = F.pad(x, (filter_width // 2, filter_width // 2, 0, 0), mode="reflect")
    if x.is_cuda:
        try:
            from .triton_ops import median_filter_cuda

            result = median_filter_cuda(x, filter_width)
        except (RuntimeError, CalledProcessError):
            warnings.warn(
                "Failed to launch Triton kernels, likely due to missing CUDA toolkit; "
                "falling back to a slower median kernel implementation..."
            )

    if result is None:
        # sort() is faster than torch.median (https://github.com/pytorch/pytorch/issues/51450)
        result = x.unfold(-1, filter_width, 1).sort()[0][..., filter_width // 2]

    if ndim <= 2:
        result = result[0, 0]

    return result


@numba.jit(nopython=True)
def backtrace(trace: np.ndarray):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))

        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")

    result = np.array(result)
    return result[::-1, :].T


@numba.jit(nopython=True, parallel=True)
def dtw_cpu(x: np.ndarray):
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace(trace)


def dtw_cuda(x, BLOCK_SIZE=1024):
    from .triton_ops import dtw_kernel

    M, N = x.shape
    assert M < BLOCK_SIZE, f"M should be smaller than {BLOCK_SIZE=}"

    x_skew = (
        F.pad(x, (0, M + 1), value=np.inf).flatten()[: M * (N + M)].reshape(M, N + M)
    )
    x_skew = x_skew.T.contiguous()
    cost = torch.ones(N + M + 2, M + 2) * np.inf
    cost[0, 0] = 0
    cost = cost.cuda()
    trace = torch.zeros_like(cost, dtype=torch.int32)

    dtw_kernel[(1,)](
        cost,
        trace,
        x_skew,
        x_skew.stride(0),
        cost.stride(0),
        trace.stride(0),
        N,
        M,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    trace = trace.T.flatten()[: (M + 1) * (M + N + 3)].reshape(M + 1, M + N + 3)[
        :, : N + 1
    ]
    return backtrace(trace.cpu().numpy())


def dtw(x: torch.Tensor) -> np.ndarray:
    if x.is_cuda:
        try:
            return dtw_cuda(x)
        except (RuntimeError, CalledProcessError):
            warnings.warn(
                "Failed to launch Triton kernels, likely due to missing CUDA toolkit; "
                "falling back to a slower DTW implementation..."
            )

    return dtw_cpu(x.double().cpu().numpy())


@dataclass
class WordTiming:
    word: str
    tokens: List[int]
    start: float
    end: float
    probability: float


def find_alignment(
    model: "Whisper",
    tokenizer: Tokenizer,
    text_tokens: List[int],
    mel: torch.Tensor,
    num_frames: int,
    *,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
) -> List[WordTiming]:

    if len(text_tokens) == 0:
        return []

    # Get token probabilities using TPU model
    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ]
    )

    # Use TPU model to get logits and token probabilities
    with torch.no_grad():
        audio_features = model.embed_audio(mel.unsqueeze(0))
        logits = model.logits(tokens.unsqueeze(0), audio_features)
        logits = logits[0]
        sampled_logits = logits[len(tokenizer.sot_sequence) :, : tokenizer.eot]
        token_probs = sampled_logits.softmax(dim=-1)
        text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens]
        text_token_probs = text_token_probs.tolist()

    # Simplified alignment matrix based on linear interpolation
    # This is a fallback approach for TPU model without cross attention weights
    matrix = np.zeros((len(text_tokens), num_frames // 2))
    
    # Create a smooth alignment matrix using gaussian distribution
    for i, token in enumerate(text_tokens):
        # Calculate expected position for this token
        expected_pos = (i + 0.5) * (num_frames // 2) / len(text_tokens)
        
        # Create gaussian distribution around expected position
        sigma = (num_frames // 2) / (len(text_tokens) * 3)  # Adjust sigma for smoothness
        for j in range(num_frames // 2):
            # Gaussian weight centered at expected_pos
            weight = np.exp(-0.5 * ((j - expected_pos) / sigma) ** 2)
            matrix[i, j] = weight
    
    # Apply DTW to find optimal alignment
    text_indices, time_indices = dtw(-torch.from_numpy(matrix))

    # Split tokens into words and get word boundaries
    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    # Calculate timing for each word
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    
    # Calculate word probabilities
    word_probabilities = [
        np.mean(text_token_probs[i:j])
        for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
    ]

    return [
        WordTiming(word, tokens, start, end, probability)
        for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities
        )
    ]


def merge_punctuations(alignment: List[WordTiming], prepended: str, appended: str):
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous.word.startswith(" ") and previous.word.strip() in prepended:
            # prepend it to the following word
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens
            previous.word = ""
            previous.tokens = []
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous.word.endswith(" ") and following.word in appended:
            # append it to the previous word
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens
            following.word = ""
            following.tokens = []
        else:
            i = j
        j += 1


def simple_word_alignment(
    tokenizer: Tokenizer,
    text_tokens: List[int],
    segment_start: float,
    segment_end: float,
    num_frames: int,
) -> List[WordTiming]:
    """
    简化的单词对齐算法，不依赖torch，使用线性插值方法
    """
    if len(text_tokens) == 0:
        return []

    # 将tokens分割成单词
    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(words) == 0:
        return []

    # 去除最后的eot token
    words = words[:-1]
    word_tokens = word_tokens[:-1]
    
    if len(words) == 0:
        return []

    # 计算每个单词的时间戳 - 使用线性插值
    segment_duration = segment_end - segment_start
    word_timings = []
    
    for i, (word, tokens) in enumerate(zip(words, word_tokens)):
        if len(words) == 1:
            # 只有一个单词，占据整个segment
            start_time = segment_start
            end_time = segment_end
        else:
            # 多个单词，使用线性插值
            start_ratio = i / len(words)
            end_ratio = (i + 1) / len(words)
            start_time = segment_start + start_ratio * segment_duration
            end_time = segment_start + end_ratio * segment_duration
        
        # 估算概率（简化为固定值）
        probability = 0.8
        
        word_timings.append(WordTiming(
            word=word,
            tokens=tokens,
            start=start_time,
            end=end_time,
            probability=probability
        ))
    
    return word_timings


def add_word_timestamps(
    *,
    segments: List[dict],
    model: "Whisper",
    tokenizer: Tokenizer,
    mel=None,  # 不再使用torch.Tensor
    num_frames: int,
    prepend_punctuations: str = "\"'([{-",
    append_punctuations: str = "\"'.!?:)]}-",
    last_speech_timestamp: float,
    **kwargs,
):
    """
    简化的单词时间戳添加函数，移除torch依赖，使用简化的拆分算法
    """
    if len(segments) == 0:
        return

    # 为每个segment处理单词时间戳
    for segment in segments:
        if "tokens" not in segment:
            continue
            
        text_tokens = [token for token in segment["tokens"] if token < tokenizer.eot]
        
        if len(text_tokens) == 0:
            segment["words"] = []
            continue
        
        # 使用简化的对齐算法
        alignment = simple_word_alignment(
            tokenizer=tokenizer,
            text_tokens=text_tokens,
            segment_start=segment["start"],
            segment_end=segment["end"],
            num_frames=num_frames
        )
        
        # 计算单词持续时间统计
        word_durations = np.array([t.end - t.start for t in alignment])
        word_durations = word_durations[word_durations > 0]
        median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.5
        max_duration = median_duration * 2
        
        # 处理过长的单词
        sentence_end_marks = ".!?"
        for i in range(len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks:
                    alignment[i] = WordTiming(
                        alignment[i].word,
                        alignment[i].tokens,
                        alignment[i].start,
                        alignment[i].start + max_duration,
                        alignment[i].probability
                    )
                elif i > 0 and alignment[i - 1].word in sentence_end_marks:
                    alignment[i] = WordTiming(
                        alignment[i].word,
                        alignment[i].tokens,
                        alignment[i].end - max_duration,
                        alignment[i].end,
                        alignment[i].probability
                    )

        # 合并标点符号
        merge_punctuations(alignment, prepend_punctuations, append_punctuations)

        # 转换为输出格式
        words = []
        for timing in alignment:
            if timing.word:
                words.append(
                    dict(
                        word=timing.word,
                        start=round(timing.start, 2),
                        end=round(timing.end, 2),
                        probability=timing.probability,
                    )
                )

        # 处理segment边界
        if len(words) > 0:
            # 确保首尾单词与segment时间戳对齐
            if words[0]["start"] > segment["start"]:
                words[0]["start"] = segment["start"]
            if words[-1]["end"] < segment["end"]:
                words[-1]["end"] = segment["end"]
                
            # 更新segment时间戳
            segment["start"] = words[0]["start"]
            segment["end"] = words[-1]["end"]

        segment["words"] = words
