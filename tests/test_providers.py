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
