"""本机配置、DeepSeek聊天与SiliconFlow向量接口；不记录请求正文或密钥。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import requests


LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parents[1]
TIMEOUT = (10, 120)
EMBEDDING_BATCH_SIZE = 16
DEFAULTS = {
    "deepseek_api_key": "",
    "deepseek_base_url": "https://api.deepseek.com",
    "deepseek_model": "deepseek-flash",
    "siliconflow_api_key": "",
    "siliconflow_base_url": "https://api.siliconflow.cn/v1",
    "siliconflow_embedding_model": "BAAI/bge-m3",
}
ENV_NAMES = {
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "deepseek_base_url": "DEEPSEEK_BASE_URL",
    "deepseek_model": "DEEPSEEK_MODEL",
    "siliconflow_api_key": "SILICONFLOW_API_KEY",
    "siliconflow_base_url": "SILICONFLOW_BASE_URL",
    "siliconflow_embedding_model": "SILICONFLOW_EMBEDDING_MODEL",
}
LEGACY_ENV_NAMES = {
    "siliconflow_base_url": "RAG_LAB_BASE_URL",
    "siliconflow_embedding_model": "RAG_LAB_EMBEDDING_MODEL",
}
SECRET_FIELDS = ("deepseek_api_key", "siliconflow_api_key")
ROOT_ENV_NAMES = {
    "SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL", "SILICONFLOW_EMBEDDING_MODEL",
    "RAG_LAB_BASE_URL", "RAG_LAB_EMBEDDING_MODEL",
}
USAGE_FIELDS = {
    "prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens",
    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "cached_tokens", "reasoning_tokens",
    "audio_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens",
    "prompt_tokens_details", "completion_tokens_details", "input_tokens_details", "output_tokens_details",
}


class ProviderError(RuntimeError):
    """可展示的安全错误；不包含远端错误正文或原始requests异常。"""

    def __init__(self, code, message, status=None):
        super().__init__(message)
        self.code = code
        self.status = status


def _error(code, message, status=None):
    return ProviderError(code, message, status)


def _safe_text(value, settings):
    text = str(value)
    for field in SECRET_FIELDS:
        secret = settings.get(field, "")
        if secret:
            text = text.replace(secret, "<redacted>")
    return re.sub(r"(?i)Bearer\s+[^\s\"'<>]+", "Bearer <redacted>", text)


def _validated_settings(values):
    if not isinstance(values, dict):
        raise _error("invalid_config", "模型配置必须为一个对象。")
    settings = dict(DEFAULTS)
    for field in DEFAULTS:
        if field in values:
            value = values[field]
            if not isinstance(value, str) or any(ord(c) < 32 for c in value):
                raise _error("invalid_config", "模型配置项必须为单行文字。")
            settings[field] = value.strip()
    for field in ("deepseek_base_url", "siliconflow_base_url"):
        value = settings[field].rstrip("/")
        try:
            url = urlsplit(value)
            _ = url.port
            valid = (url.scheme == "https" and bool(url.hostname) and not url.username
                     and not url.password and not url.query and not url.fragment
                     and not any(c.isspace() for c in value))
        except ValueError:
            valid = False
        if not valid:
            raise _error("invalid_config", "Base URL 必须为不含账号、查询参数或片段的HTTPS地址。")
        settings[field] = value
    for field in ("deepseek_model", "siliconflow_embedding_model"):
        if not settings[field] or len(settings[field]) > 300 or any(c.isspace() for c in settings[field]):
            raise _error("invalid_config", "请填写有效的模型名称。")
    return settings


def _parse_env(text, allowed):
    """只取明确允许的键；不执行shell、变量替换或任意Python表达式。"""
    result = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if not match or match[1] not in allowed:
            continue
        name, value = match.groups()
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                continue
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if isinstance(value, str):
            result[name] = value
    return result


def _read_env(path, allowed):
    try:
        if not path.exists():
            return {}
        return _parse_env(path.read_text(encoding="utf-8"), allowed)
    except (OSError, UnicodeError):
        raise _error("config_read", "本机模型配置读取失败，请检查.env文件及权限。") from None


class ConfigStore:
    """非空进程变量 > 本实验.env > 仓库.env的SF兼容项；只保存本实验.env。"""

    def __init__(self, lab_dir=LAB_DIR, root_dir=None):
        candidate = Path(lab_dir)
        self.path = candidate if candidate.name == ".env" else candidate / ".env"
        self.lab_dir = self.path.parent
        root = Path(root_dir) if root_dir is not None else PROJECT_DIR
        self.root_path = root if root.name == ".env" else root / ".env"

    def get(self):
        allowed = set(ENV_NAMES.values()) | set(LEGACY_ENV_NAMES.values())
        local = _read_env(self.path, allowed)
        fallback = _read_env(self.root_path, ROOT_ENV_NAMES) if self.root_path != self.path else {}
        settings = dict(DEFAULTS)
        for field, name in ENV_NAMES.items():
            names = (name, LEGACY_ENV_NAMES[field]) if field in LEGACY_ENV_NAMES else (name,)
            for source in (os.environ, local, fallback):
                value = next((source[n] for n in names if n in source and str(source[n]).strip()), None)
                if value is not None:
                    settings[field] = value
                    break
        return _validated_settings(settings)

    def public(self):
        settings = self.get()
        public = {field: _safe_text(value, settings) for field, value in settings.items() if field not in SECRET_FIELDS}
        public["deepseek_configured"] = bool(settings["deepseek_api_key"])
        public["siliconflow_configured"] = bool(settings["siliconflow_api_key"])
        public["configured"] = {
            "deepseek": public["deepseek_configured"],
            "siliconflow": public["siliconflow_configured"],
        }
        return public

    def save(self, updates):
        if not isinstance(updates, dict):
            raise _error("invalid_config", "待保存的模型配置必须为对象。")
        if set(updates) - set(ENV_NAMES):
            raise _error("invalid_config", "待保存配置包含无法识别的字段。")
        merged = self.get()
        for field, value in updates.items():
            if field in SECRET_FIELDS and isinstance(value, str) and not value.strip():
                continue
            merged[field] = value
        merged = _validated_settings(merged)
        try:
            old_lines = self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []
        except (OSError, UnicodeError):
            raise _error("config_read", "本机模型配置读取失败，未保存修改。") from None
        managed = {name: json.dumps(merged[field], ensure_ascii=False) for field, name in ENV_NAMES.items()}
        lines, written = [], set()
        for line in old_lines:
            match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            name = match[1] if match else None
            if name in managed:
                if name not in written:
                    lines.append(f"{name}={managed[name]}")
                    written.add(name)
            else:
                lines.append(line)
        lines.extend(f"{name}={value}" for name, value in managed.items() if name not in written)
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=".env-write-", delete=False) as handle:
                temporary = Path(handle.name)
                descriptor_chmod = getattr(os, "fchmod", None)
                if callable(descriptor_chmod):
                    descriptor_chmod(handle.fileno(), 0o600)
                else:
                    # Windows inherits directory ACLs and may lack fchmod;
                    # preserve atomic writes without assuming POSIX mode bits.
                    os.chmod(temporary, 0o600)
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError:
            raise _error("config_write", "本机模型配置保存失败，请检查实验目录权限。") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return self.public()


def _usage(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if key not in USAGE_FIELDS:
            continue
        if isinstance(item, dict) and key.endswith("_details"):
            result[key] = _usage(item)
        elif not key.endswith("_details") and isinstance(item, (int, float)) and not isinstance(item, bool):
            try:
                if math.isfinite(item) and item >= 0:
                    result[key] = item
            except OverflowError:
                pass
    return result


def _merge_usage(total, incoming):
    for key, value in incoming.items():
        if isinstance(value, dict):
            if not isinstance(total.get(key), dict):
                total[key] = {}
            _merge_usage(total.setdefault(key, {}), value)
        else:
            previous = total.get(key, 0)
            total[key] = (previous if isinstance(previous, (int, float)) else 0) + value


def _post(settings, provider, endpoint, body):
    key = settings[f"{provider}_api_key"]
    label = "DeepSeek" if provider == "deepseek" else "SiliconFlow"
    if not key:
        raise _error("missing_key", f"尚未配置{label} API Key。")
    try:
        response = requests.post(
            settings[f"{provider}_base_url"] + endpoint,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body, timeout=TIMEOUT, allow_redirects=False,
        )
    except requests.Timeout:
        raise _error("timeout", f"{label}请求超时，请稍后重试。") from None
    except requests.RequestException:
        raise _error("connection", f"{label}连接失败，请检查网络及服务地址。") from None
    status = response.status_code
    if not isinstance(status, int) or not 200 <= status < 300:
        messages = {
            401: ("authentication", "API Key无效或已失效"),
            403: ("permission", "账户没有模型访问权限"),
            429: ("rate_limit", "请求受限，请稍后重试或检查账户限额"),
        }
        code, message = messages.get(status, ("http_error", "模型服务请求失败"))
        safe_status = status if type(status) is int else None
        suffix = f"（HTTP {safe_status}）" if safe_status is not None else ""
        raise _error(code, f"{label}{message}{suffix}。", safe_status)
    try:
        payload = response.json()
    except (ValueError, TypeError):
        raise _error("invalid_response", f"{label}未返回有效JSON。") from None
    if not isinstance(payload, dict) or payload.get("error"):
        raise _error("invalid_response", f"{label}返回了错误或不符合约定的结果。")
    return payload


def _vector(value):
    if not isinstance(value, list) or not value:
        raise _error("invalid_response", "Embedding未返回非空向量。")
    result = []
    for number in value:
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            raise _error("invalid_response", "Embedding包含非数值或无效向量。")
        try:
            number = float(number)
        except (OverflowError, ValueError):
            raise _error("invalid_response", "Embedding包含非有限向量数值。") from None
        if not math.isfinite(number):
            raise _error("invalid_response", "Embedding包含非有限向量数值。")
        result.append(number)
    return result


class SiliconFlowEmbedder:
    def __init__(self, settings, cache_dir=None):
        self.settings = _validated_settings(settings)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else LAB_DIR / "runtime" / "cache"
        self.cache_path = self.cache_dir / "embeddings.sqlite3"
        self.dimension = None
        self.last_usage = {}
        self.last_stats = {"requests": 0, "cache_hits": 0, "input_count": 0, "dimension": None, "usage": {}}

    def _cache_key(self, text, role):
        identity = {
            "version": 1, "provider": "siliconflow", "url": self.settings["siliconflow_base_url"],
            "model": self.settings["siliconflow_embedding_model"], "text": text, "role": role,
        }
        encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _connection(self):
        connection = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.cache_path, timeout=15)
            connection.execute("CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
            return connection
        except (OSError, sqlite3.Error):
            if connection is not None:
                connection.close()
            raise _error("cache_error", "本机向量缓存无法打开，请检查实验缓存目录。") from None

    def _read_cache(self, keys):
        cached = {}
        connection = self._connection()
        try:
            for key in keys:
                row = connection.execute("SELECT vector FROM embeddings WHERE cache_key = ?", (key,)).fetchone()
                if row:
                    try:
                        cached[key] = _vector(json.loads(row[0]))
                    except (ValueError, TypeError, ProviderError):
                        continue
        except sqlite3.Error:
            raise _error("cache_error", "本机向量缓存读取失败。") from None
        finally:
            connection.close()
        return cached

    def _write_cache(self, rows):
        connection = self._connection()
        try:
            with connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO embeddings (cache_key, vector) VALUES (?, ?)",
                    [(key, json.dumps(vector, allow_nan=False, separators=(",", ":"))) for key, vector in rows],
                )
        except (sqlite3.Error, ValueError):
            raise _error("cache_error", "本机向量缓存写入失败。") from None
        finally:
            connection.close()

    def embed(self, texts, role="document"):
        if not isinstance(texts, (list, tuple)) or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise _error("invalid_input", "Embedding输入须为非空文本组成的列表。")
        if not isinstance(role, str) or role not in {"document", "query"}:
            raise _error("invalid_input", "Embedding角色只能为document或query。")
        self.last_usage = {}
        self.last_stats = {"requests": 0, "cache_hits": 0, "input_count": len(texts), "dimension": None, "usage": {}}
        if not texts:
            return []
        unique = list(dict.fromkeys(texts))
        keys = {text: self._cache_key(text, role) for text in unique}
        cached = self._read_cache(keys.values())
        vectors = {text: cached[keys[text]] for text in unique if keys[text] in cached}
        dimensions = {len(vector) for vector in vectors.values()}
        if len(dimensions) > 1:
            raise _error("invalid_cache", "同一Embedding模型的缓存维度不一致，请重建缓存。")
        if self.dimension is not None and dimensions and self.dimension not in dimensions:
            raise _error("invalid_cache", "Embedding缓存维度与当前模型调用不一致。")
        expected_dimension = self.dimension or next(iter(dimensions), None)
        self.last_stats["cache_hits"] = sum(text in vectors for text in texts)
        missing = [text for text in unique if text not in vectors]
        for start in range(0, len(missing), EMBEDDING_BATCH_SIZE):
            batch = missing[start:start + EMBEDDING_BATCH_SIZE]
            payload = _post(self.settings, "siliconflow", "/embeddings", {
                "model": self.settings["siliconflow_embedding_model"], "input": batch, "encoding_format": "float",
            })
            self.last_stats["requests"] += 1
            rows = payload.get("data")
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise _error("invalid_response", "Embedding返回条数与输入不一致。")
            indexed = {}
            for row in rows:
                index = row.get("index") if isinstance(row, dict) else None
                if type(index) is not int or not 0 <= index < len(batch) or index in indexed:
                    raise _error("invalid_response", "Embedding索引缺失、重复或越界。")
                indexed[index] = _vector(row.get("embedding"))
            batch_dimensions = {len(vector) for vector in indexed.values()}
            if len(batch_dimensions) != 1 or (expected_dimension is not None and expected_dimension not in batch_dimensions):
                raise _error("invalid_response", "Embedding返回维度不一致，已停止计算。")
            expected_dimension = next(iter(batch_dimensions))
            self._write_cache([(keys[text], indexed[index]) for index, text in enumerate(batch)])
            for index, text in enumerate(batch):
                vectors[text] = indexed[index]
            _merge_usage(self.last_usage, _usage(payload.get("usage")))
        self.last_stats["dimension"] = expected_dimension
        self.dimension = expected_dimension
        self.last_stats["usage"] = dict(self.last_usage)
        return [list(vectors[text]) for text in texts]


class DeepSeekChat:
    def __init__(self, settings):
        self.settings = _validated_settings(settings)

    def chat(self, messages, max_tokens=1000):
        if not isinstance(messages, list) or not messages:
            raise _error("invalid_input", "聊天需要非空消息列表。")
        normalized = []
        for message in messages:
            if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str) or not message["content"].strip()):
                raise _error("invalid_input", "消息须包含有效role及非空文字content。")
            normalized.append({"role": message["role"], "content": message["content"]})
        if type(max_tokens) is not int or max_tokens < 1:
            raise _error("invalid_input", "max_tokens必须为正整数。")
        payload = _post(self.settings, "deepseek", "/chat/completions", {
            "model": self.settings["deepseek_model"], "messages": normalized,
            "max_tokens": max_tokens, "temperature": 0, "stream": False, "thinking": {"type": "disabled"},
        })
        choices = payload.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise _error("invalid_response", "DeepSeek未返回回答正文，请检查输出长度或模型设置。")
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            model = self.settings["deepseek_model"]
        return {"content": _safe_text(content, self.settings), "usage": _usage(payload.get("usage")),
                "model": _safe_text(model, self.settings)}
