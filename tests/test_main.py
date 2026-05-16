import sys
from unittest.mock import Mock

import pytest

import wyoming_openai.__main__ as main_module
from wyoming_openai.__main__ import main


@pytest.mark.asyncio
async def test_main_rejects_non_object_stt_extra_body_env(monkeypatch, capsys):
    monkeypatch.setenv("STT_EXTRA_BODY", '["not-an-object"]')
    monkeypatch.setattr(sys, "argv", ["wyoming_openai"])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "Invalid STT extra body: expected a JSON object" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_invalid_tts_extra_body_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["wyoming_openai", "--tts-extra-body", '{"stream":'])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "Invalid TTS extra body" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_invalid_stt_response_format_before_server_start(monkeypatch, capsys):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-models",
            "whisper-1",
            "--stt-extra-body",
            '{"response_format":"text"}',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "STT extra_body response_format must be one of 'json'" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_non_boolean_stt_stream_override_before_server_start(monkeypatch, capsys):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-models",
            "whisper-1",
            "--stt-extra-body",
            '{"stream":"yes"}',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "STT extra_body stream must be a boolean" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_rejects_tts_transport_override_before_server_start(monkeypatch, capsys):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-models",
            "tts-1",
            "--tts-voices",
            "alloy",
            "--tts-extra-body",
            '{"stream":true}',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "TTS extra_body does not support overriding 'stream'" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_validates_tts_extra_body_before_client_creation(monkeypatch, capsys):
    def unexpected_factory():
        async def should_not_be_called(*args, **kwargs):
            raise AssertionError("client factory should not be created for invalid extra_body")

        return should_not_be_called

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(unexpected_factory),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-models",
            "tts-1",
            "--tts-voices",
            "alloy",
            "--tts-extra-body",
            '{"stream":true}',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "TTS extra_body does not support overriding 'stream'" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_allows_invalid_unused_tts_extra_body_when_voice_discovery_returns_none(monkeypatch):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-models",
            "whisper-1",
            "--tts-models",
            "tts-1",
            "--tts-extra-body",
            '{"stream":true}',
        ],
    )

    await main()


class _FakeClient:
    def __init__(self):
        self.backend = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def list_supported_voices(self, *args, **kwargs):
        return []


class _CapturingServer:
    async def run(self, handler_factory):
        handler_factory(Mock(name="reader"), Mock(name="writer"))


@pytest.mark.asyncio
async def test_main_allows_unused_tts_response_format_in_stt_only_mode(monkeypatch):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    for env_var in ("TTS_MODELS", "TTS_STREAMING_MODELS", "TTS_VOICES"):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--stt-models",
            "whisper-1",
            "--tts-extra-body",
            '{"response_format":"mp3"}',
        ],
    )

    await main()


@pytest.mark.asyncio
async def test_main_passes_cooldown_buffer_ms_from_env(monkeypatch):
    captured_kwargs: dict = {}

    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    class _CapturingKwargsServer:
        async def run(self, handler_factory):
            captured_kwargs.update(handler_factory.keywords)

    monkeypatch.setenv("TTS_COOLDOWN_BUFFER_MS", "300")
    monkeypatch.setenv("TTS_TRAILING_SILENCE_MS", "150")
    for env_var in ("STT_MODELS", "STT_STREAMING_MODELS"):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingKwargsServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-models",
            "tts-1",
            "--tts-voices",
            "alloy",
        ],
    )

    await main()

    assert captured_kwargs["tts_cooldown_buffer_ms"] == 300
    assert captured_kwargs["tts_trailing_silence_ms"] == 150


@pytest.mark.asyncio
async def test_main_cooldown_buffer_ms_defaults_to_none(monkeypatch):
    captured_kwargs: dict = {}

    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    class _CapturingKwargsServer:
        async def run(self, handler_factory):
            captured_kwargs.update(handler_factory.keywords)

    for env_var in (
        "STT_MODELS",
        "STT_STREAMING_MODELS",
        "TTS_COOLDOWN_BUFFER_MS",
        "TTS_TRAILING_SILENCE_MS",
    ):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingKwargsServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-models",
            "tts-1",
            "--tts-voices",
            "alloy",
        ],
    )

    await main()

    assert captured_kwargs["tts_cooldown_buffer_ms"] is None
    assert captured_kwargs["tts_trailing_silence_ms"] is None


@pytest.mark.asyncio
async def test_main_allows_unused_stt_response_format_in_tts_only_mode(monkeypatch):
    async def fake_factory(*args, **kwargs):
        return _FakeClient()

    for env_var in ("STT_MODELS", "STT_STREAMING_MODELS"):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(
        main_module.CustomAsyncOpenAI,
        "create_autodetected_factory",
        staticmethod(lambda: fake_factory),
    )
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        staticmethod(lambda uri: _CapturingServer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming_openai",
            "--tts-models",
            "tts-1",
            "--tts-voices",
            "alloy",
            "--stt-extra-body",
            '{"response_format":"text"}',
        ],
    )

    await main()
