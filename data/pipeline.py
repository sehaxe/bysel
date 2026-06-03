"""
📚 busel PIPELINE v5.1 - Parallel Stream Interleaving Loader
Поддерживает динамическую сборку и параллельное перемешивание данных на лету в памяти.
"""
import torch
import os
import json
import random
import platform
from torch.utils.data import IterableDataset, DataLoader

try:
    import busel
    HAS_RUST_IO = True
except ImportError:
    HAS_RUST_IO = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

class PythonByteStreamer:
    def __init__(self, file_path, chunk_size, start_offset=0):
        with open(file_path, "rb") as f:
            self.data = f.read()
        self.position = start_offset
        self.chunk_size = chunk_size

    def next_chunk(self):
        if self.position >= len(self.data):
            return None
        start = self.position
        end = min(self.position + self.chunk_size, len(self.data))
        chunk = list(self.data[start:end])
        self.position = end
        if len(chunk) < self.chunk_size:
            chunk = chunk + [0] * (self.chunk_size - len(chunk))
        return chunk

    def get_position(self):
        return self.position

class buselOmnivoreTextExtractor:
    def __init__(self, file_path, chunk_size, start_offset=0, img_size=(32, 32)):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.position = start_offset
        self.img_size = img_size
        self.raw_bytes = bytearray()

        if file_path.endswith('.parquet'):
            if not HAS_PANDAS:
                raise ImportError("❌ Для чтения .parquet установите: 'uv add pandas pyarrow'")
            df = pd.read_parquet(file_path)
            text_col = self._detect_text_column(df)
            full_text = "\n".join(text_col.astype(str).tolist())
            self.raw_bytes = bytearray(full_text.encode('utf-8'))
        elif file_path.endswith('.jsonl'):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        if "image" in data and HAS_PIL:
                            img_path = data["image"]
                            if not os.path.isabs(img_path):
                                img_path = os.path.join(os.path.dirname(file_path), img_path)
                            if os.path.exists(img_path):
                                img = Image.open(img_path).convert("RGB")
                                img = img.resize(self.img_size)
                                img_bytes = img.tobytes()
                                self.raw_bytes.append(256) 
                                self.raw_bytes.extend(img_bytes)
                                self.raw_bytes.append(257) 
                                text_val = self._recursive_extract_excluding_image(data)
                                if text_val.strip():
                                    self.raw_bytes.extend(text_val.strip().encode('utf-8'))
                                self.raw_bytes.extend(b"\n")
                        else:
                            text_val = self._recursive_extract_excluding_image(data)
                            if text_val.strip():
                                self.raw_bytes.extend(text_val.strip().encode('utf-8'))
                                self.raw_bytes.extend(b"\n")
                    except Exception:
                        continue
        else:
            with open(file_path, "rb") as f:
                self.raw_bytes = bytearray(f.read())

    def _recursive_extract_excluding_image(self, obj):
        if isinstance(obj, str):
            if obj.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')): return ""
            return obj
        elif isinstance(obj, dict):
            return "\n".join([self._recursive_extract_excluding_image(v) for k, v in obj.items() if k != "image" and v])
        elif isinstance(obj, list):
            return "\n".join([self._recursive_extract_excluding_image(item) for item in obj if item])
        else:
            return ""

    def _detect_text_column(self, df):
        for col in ["text", "content", "body", "code", "markdown", "raw_text"]:
            if col in df.columns: return df[col]
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "string": return df[col]
        raise ValueError("Не удалось найти текстовую колонку в Parquet файле.")

    def next_chunk(self):
        if self.position >= len(self.raw_bytes):
            return None
        start = self.position
        end = min(self.position + self.chunk_size, len(self.raw_bytes))
        chunk = list(self.raw_bytes[start:end])
        self.position = end
        if len(chunk) < self.chunk_size:
            chunk = chunk + [0] * (self.chunk_size - len(chunk))
        return chunk

    def get_position(self):
        return self.position

class RustByteStreamDataset(IterableDataset):
    def __init__(self, data_path, chunk_size=8192, start_file_idx=0, start_byte_offset=0):
        super().__init__()
        self.chunk_size = chunk_size
        self.start_file_idx = start_file_idx
        self.start_byte_offset = start_byte_offset
        self.files = []

        if os.path.isdir(data_path):
            for root, _, filenames in os.walk(data_path):
                for filename in filenames:
                    if filename.endswith(('.txt', '.py', '.rs', '.go', '.be', '.json', '.cpp', '.h', '.jsonl', '.parquet')):
                        self.files.append(os.path.join(root, filename))
            self.files.sort()
        elif os.path.isfile(data_path):
            self.files.append(data_path)

        if not self.files:
            raise ValueError(f"❌ [ОШИБКА ДАННЫХ]: В папке '{data_path}' не найдено подходящих файлов для обучения!\n")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_to_process = [f for i, f in enumerate(self.files) if i % num_workers == worker_id]
        else:
            files_to_process = self.files

        if not files_to_process:
            return

        random.shuffle(files_to_process)
        active_streamers = []
        for file_path in files_to_process:
            offset = self.start_byte_offset if (self.start_file_idx < len(self.files) and file_path == self.files[self.start_file_idx]) else 0
            use_rust_streamer = (not file_path.endswith(('.parquet', '.jsonl')) and HAS_RUST_IO)
            
            if use_rust_streamer:
                streamer = busel.ByteStreamer(file_path, self.chunk_size, offset)
            else:
                streamer = buselOmnivoreTextExtractor(file_path, self.chunk_size, offset)
            active_streamers.append((streamer, file_path))

        shuffle_buffer = []
        buffer_size = 50

        while active_streamers:
            idx = random.randint(0, len(active_streamers) - 1)
            streamer, file_path = active_streamers[idx]
            chunk = streamer.next_chunk()
            if chunk is None:
                active_streamers.pop(idx)
                continue

            pseudo_file_idx = self.files.index(file_path) if file_path in self.files else 0
            byte_offset = streamer.get_position()
            shuffle_buffer.append((chunk, pseudo_file_idx, byte_offset))

            if len(shuffle_buffer) >= buffer_size:
                random.shuffle(shuffle_buffer)
                yield shuffle_buffer.pop(0)

        random.shuffle(shuffle_buffer)
        for item in shuffle_buffer:
            yield item

def collate_busel_batch(batch):
    chunks = [item[0] for item in batch]
    file_indices = [item[1] for item in batch]
    byte_offsets = [item[2] for item in batch]
    
    tensors = []
    for c in chunks:
        # 🎯 ИСПРАВЛЕНИЕ ОШИБКИ: PyO3 0.28 возвращает bytes. 
        # Используем torch.frombuffer для мгновенного zero-copy парсинга.
        if isinstance(c, (bytes, bytearray)):
            tensors.append(torch.frombuffer(bytearray(c), dtype=torch.uint8).to(torch.int32))
        else:
            tensors.append(torch.tensor(list(c), dtype=torch.int32))
            
    batch_tensors = torch.stack(tensors)
    return batch_tensors, file_indices[-1], byte_offsets[-1]

def get_busel_dataloader(data_path, chunk_size, batch_size, start_file_idx=0, start_byte_offset=0, num_workers=None):
    dataset = RustByteStreamDataset(data_path, chunk_size, start_file_idx, start_byte_offset)
    use_pin = torch.cuda.is_available()
    if num_workers is None:
        if platform.system() == "Linux" and torch.cuda.is_available():
            num_workers = min(4, os.cpu_count() or 1)
        else:
            num_workers = 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin,
        collate_fn=collate_busel_batch
    )