import unittest

from context_compiler.providers import ModelClient, ProviderConfig, ProviderError


class ProviderTest(unittest.TestCase):
    def test_config_selects_provider_specific_key(self):
        config = ProviderConfig.from_env({
            "AI_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "ds-key",
            "AI_MODEL": "deepseek-reasoner",
        })
        self.assertEqual(config.api_key, "ds-key")
        self.assertEqual(config.model, "deepseek-reasoner")
        self.assertIn("deepseek.com", config.base_url)

    def test_openai_compatible_normalizes_usage_and_cache_reads(self):
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update({"url": url, "headers": headers, "payload": payload})
            return {
                "model": "test-model",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 4,
                    "prompt_cache_hit_tokens": 20,
                },
            }

        client = ModelClient(
            ProviderConfig("openai-compatible", "test-model", api_key="secret", base_url="http://test"),
            transport=transport,
        )
        response = client.complete([{"role": "user", "content": "hello"}], max_tokens=20)
        self.assertEqual(response.text, "ok")
        self.assertEqual(response.usage.input_tokens, 10)
        self.assertEqual(response.usage.cache_read_tokens, 20)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["payload"]["max_tokens"], 20)

    def test_openrouter_adds_required_attribution_headers(self):
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update({"url": url, "headers": headers})
            return {"choices": [{"message": {"content": "ok"}}]}

        client = ModelClient(
            ProviderConfig("openrouter", "openai/test", api_key="router-key", base_url="http://router"),
            transport=transport,
        )
        client.complete([{"role": "user", "content": "hello"}])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer router-key")
        self.assertEqual(captured["headers"]["HTTP-Referer"], "graph-context-compiler")
        self.assertEqual(captured["headers"]["X-Title"], "Graph Context Compiler")

    def test_anthropic_moves_system_message_to_system_field(self):
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update({"url": url, "headers": headers, "payload": payload})
            return {
                "model": "claude-test",
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 3},
            }

        client = ModelClient(
            ProviderConfig("anthropic", "claude-test", api_key="anthropic-key", base_url="http://anthropic"),
            transport=transport,
        )
        response = client.complete([{"role": "user", "content": "question"}], system="repository context")
        self.assertEqual(response.text, "done")
        self.assertEqual(captured["payload"]["system"], "repository context")
        self.assertNotIn("system", [message["role"] for message in captured["payload"]["messages"]])
        self.assertEqual(captured["headers"]["x-api-key"], "anthropic-key")

    def test_gemini_translates_roles_and_usage(self):
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update({"url": url, "payload": payload})
            return {
                "candidates": [{"content": {"parts": [{"text": "gemini"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 2, "cachedContentTokenCount": 5},
            }

        client = ModelClient(
            ProviderConfig("gemini", "gemini-test", api_key="gemini-key", base_url="http://gemini/models"),
            transport=transport,
        )
        response = client.complete([{"role": "assistant", "content": "prior"}, {"role": "user", "content": "next"}])
        self.assertEqual(response.text, "gemini")
        self.assertEqual(response.usage.cache_read_tokens, 5)
        self.assertEqual(captured["payload"]["contents"][0]["role"], "model")
        self.assertIn("generateContent", captured["url"])

    def test_unknown_provider_fails_loudly(self):
        with self.assertRaises(ProviderError):
            ProviderConfig.from_env({"AI_PROVIDER": "unknown"})


if __name__ == "__main__":
    unittest.main()


class RetryTest(unittest.TestCase):
    """Transient failures must not discard a run that is hours of paid calls.

    A 192-call live sweep died on its first budget point when one read timed out,
    losing every call already made. These pin both halves of the fix: retry what
    is worth retrying, and never retry what is not.
    """

    def _client(self, transport, **overrides):
        config = ProviderConfig(provider="openrouter", model="m", api_key="k",
                                base_url="https://example.invalid/v1/chat/completions",
                                **overrides)
        return ModelClient(config, transport=transport, sleep=lambda _seconds: None)

    @staticmethod
    def _ok(url, headers, payload, timeout):
        return {"choices": [{"message": {"content": "done"}}], "model": "m"}

    def test_a_transient_failure_is_retried_and_then_succeeds(self):
        attempts = []

        def flaky(url, headers, payload, timeout):
            attempts.append(1)
            if len(attempts) < 3:
                raise ProviderError("read timed out", transient=True)
            return self._ok(url, headers, payload, timeout)

        client = self._client(flaky)
        self.assertEqual(client.complete([{"role": "user", "content": "hi"}]).text, "done")
        self.assertEqual(len(attempts), 3)
        # Reported so a run can say how much wall clock went to failures.
        self.assertEqual(client.retries, 2)

    def test_a_permanent_failure_is_not_retried(self):
        # Retrying a rejected key only spends money to fail four times.
        attempts = []

        def rejected(url, headers, payload, timeout):
            attempts.append(1)
            raise ProviderError("HTTP 401")

        client = self._client(rejected)
        with self.assertRaises(ProviderError):
            client.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(client.retries, 0)

    def test_retries_are_bounded_and_the_last_error_propagates(self):
        attempts = []

        def always_timing_out(url, headers, payload, timeout):
            attempts.append(1)
            raise ProviderError("read timed out", transient=True)

        client = self._client(always_timing_out, max_attempts=3)
        with self.assertRaises(ProviderError) as caught:
            client.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(len(attempts), 3)
        self.assertIn("timed out", str(caught.exception))

    def test_backoff_is_exponential_in_the_configured_step(self):
        waits = []
        attempts = []

        def flaky(url, headers, payload, timeout):
            attempts.append(1)
            if len(attempts) < 4:
                raise ProviderError("boom", transient=True)
            return self._ok(url, headers, payload, timeout)

        config = ProviderConfig(provider="openrouter", model="m", api_key="k",
                                base_url="u", retry_backoff_seconds=1.5)
        client = ModelClient(config, transport=flaky, sleep=waits.append)
        client.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(waits, [1.5, 3.0, 6.0])

    def test_the_read_timeout_is_wide_enough_for_this_benchmark(self):
        # Live mean latency is ~50s with a 27s sd, so a 120s ceiling sat inside
        # the observed tail. Pinned so it is not quietly lowered again.
        self.assertGreaterEqual(ProviderConfig(provider="p", model="m").timeout_seconds, 300)
        self.assertGreaterEqual(
            ProviderConfig.from_env({"AI_PROVIDER": "openrouter", "AI_MODEL": "m",
                                     "OPENROUTER_API_KEY": "k"}).timeout_seconds, 300)

    def test_environment_can_override_the_retry_policy(self):
        config = ProviderConfig.from_env({
            "AI_PROVIDER": "openrouter", "AI_MODEL": "m", "OPENROUTER_API_KEY": "k",
            "AI_MAX_ATTEMPTS": "7", "AI_RETRY_BACKOFF_SECONDS": "0.5",
            "AI_TIMEOUT_SECONDS": "600"})
        self.assertEqual((config.max_attempts, config.retry_backoff_seconds,
                          config.timeout_seconds), (7, 0.5, 600.0))
