from pathlib import Path
import unittest

from scripts.validate_skill import parse_frontmatter, validate


ROOT = Path(__file__).resolve().parents[1]


class SkillMetadataTests(unittest.TestCase):
    def test_skill_package_validates(self):
        self.assertEqual(validate(ROOT), [])

    def test_skill_frontmatter_has_only_runtime_fields(self):
        text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        fields = parse_frontmatter(text)
        self.assertEqual(set(fields), {"name", "description"})
        self.assertEqual(fields["name"], "agentic-rag")
        self.assertIn("Sufficient Context", fields["description"])

    def test_skill_uses_progressive_disclosure(self):
        text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("references/agentic-rag-behavior.md", text)
        self.assertIn("references/prompts-and-schemas.md", text)
        self.assertLess(len(text.splitlines()), 200)

    def test_openai_metadata_matches_skill(self):
        text = (ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
        self.assertIn('display_name: "Agentic RAG"', text)
        self.assertIn("Sufficient Context", text)
        self.assertIn("agentic-rag skill", text)

    def test_pyproject_uses_main_repo_name(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('name = "agentic-rag"', text)
        self.assertIn("Agent Skill", text)

    def test_readme_cites_google_agentic_rag_and_codex_implementation(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("implemented with Codex", text)
        self.assertIn(
            "https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/",
            text,
        )
        self.assertIn(
            "https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/cross-corpus-retrieval",
            text,
        )


if __name__ == "__main__":
    unittest.main()
