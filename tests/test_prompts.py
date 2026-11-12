import unittest
from unittest.mock import patch

from agent import prompts


class SystemPromptTest(unittest.TestCase):
    def test_every_condition_starts_with_the_same_core(self):
        for condition in ("A", "B", "C"):
            self.assertTrue(prompts.system_prompt(condition).startswith(prompts.CORE))

    def test_condition_part_is_appended(self):
        self.assertTrue(prompts.system_prompt("A").endswith(prompts.CONDITION_PARTS["A"]))

    def test_unknown_condition_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown condition"):
            prompts.system_prompt("D")


class PromptHashTest(unittest.TestCase):
    def test_stable(self):
        self.assertEqual(prompts.prompt_hash(), prompts.prompt_hash())

    def test_changes_with_core(self):
        before = prompts.prompt_hash()
        with patch.object(prompts, "CORE", prompts.CORE + " "):
            self.assertNotEqual(prompts.prompt_hash(), before)

    def test_changes_with_any_condition_part(self):
        before = prompts.prompt_hash()
        changed = {**prompts.CONDITION_PARTS, "C": "different"}
        with patch.object(prompts, "CONDITION_PARTS", changed):
            self.assertNotEqual(prompts.prompt_hash(), before)

    def test_changes_with_c_final_instruction(self):
        before = prompts.prompt_hash()
        with patch.object(prompts, "C_FINAL_INSTRUCTION", "Answer."):
            self.assertNotEqual(prompts.prompt_hash(), before)

    def test_changes_with_b_extraction_prompt(self):
        before = prompts.prompt_hash()
        with patch.object(prompts, "B_EXTRACTION_PROMPT", "Extract."):
            self.assertNotEqual(prompts.prompt_hash(), before)


if __name__ == "__main__":
    unittest.main()
