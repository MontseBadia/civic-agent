import tempfile
import unittest
from pathlib import Path

from agent.config import LOCAL_API_KEY_PLACEHOLDER, load_config, parse_env_file


class EnvFileTestCase(unittest.TestCase):
    """Each test writes into its own temp dir, removed after the test."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_dir = Path(tmp.name)

    def write_env(self, text: str) -> Path:
        path = self.tmp_dir / ".env"
        path.write_text(text)
        return path


class ParseEnvFileTest(EnvFileTestCase):
    def test_skips_comments_and_blank_lines(self):
        path = self.write_env("# comment\n\nA=1\n  B = two  \n")
        self.assertEqual(parse_env_file(path), {"A": "1", "B": "two"})

    def test_strips_matching_quotes_only(self):
        path = self.write_env("A=\"x y\"\nB='z'\nC=\"mixed'\n")
        self.assertEqual(parse_env_file(path), {"A": "x y", "B": "z", "C": "\"mixed'"})

    def test_line_without_equals_names_the_line(self):
        path = self.write_env("A=1\nbroken\n")
        with self.assertRaisesRegex(ValueError, ":2: expected KEY=VALUE"):
            parse_env_file(path)


class LoadConfigTest(EnvFileTestCase):
    def test_reads_file(self):
        path = self.write_env("LLM_BASE_URL=http://h/v1\nLLM_API_KEY=k\nLLM_MODEL=m\n")
        config = load_config(path, environ={})
        self.assertEqual((config.base_url, config.api_key, config.model), ("http://h/v1", "k", "m"))

    def test_environment_wins_over_file(self):
        path = self.write_env("LLM_BASE_URL=http://h/v1\nLLM_MODEL=from-file\n")
        config = load_config(path, environ={"LLM_MODEL": "from-env"})
        self.assertEqual(config.model, "from-env")

    def test_missing_file_uses_environment(self):
        config = load_config(Path("/nonexistent/.env"), environ={"LLM_BASE_URL": "http://h", "LLM_MODEL": "m"})
        self.assertEqual(config.model, "m")

    def test_missing_keys_are_listed(self):
        path = self.write_env("LLM_API_KEY=k\n")
        with self.assertRaisesRegex(ValueError, "LLM_BASE_URL, LLM_MODEL"):
            load_config(path, environ={})

    def test_empty_api_key_gets_placeholder(self):
        path = self.write_env("LLM_BASE_URL=http://h\nLLM_API_KEY=\nLLM_MODEL=m\n")
        self.assertEqual(load_config(path, environ={}).api_key, LOCAL_API_KEY_PLACEHOLDER)


if __name__ == "__main__":
    unittest.main()
