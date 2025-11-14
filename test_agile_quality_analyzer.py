import unittest
from aval_mds import AgileQualityAnalyzer, extract_co_authors_from_commit

class TestAgileQualityAnalyzer(unittest.TestCase):

    def test_extract_coauthors(self):
        msg = """Fix bug X
        Co-authored-by: Maria <maria123@gmail.com>
        Co-authored-by: joao <joao@github.com>"""
        
        result = extract_co_authors_from_commit(msg)
        self.assertIn("maria123", result)
        self.assertIn("joao", result)

    def test_commit_quality_no_ai(self):
        analyzer = AgileQualityAnalyzer(api_key=None)
        analyzer.enabled = False  # simula "AI off"
        score = analyzer.analyze_commit_quality(["fix bug", "update docs"])
        self.assertEqual(score, 0.5)

    def test_issue_quality_no_ai(self):
        analyzer = AgileQualityAnalyzer(api_key=None)
        analyzer.enabled = False
        score = analyzer.analyze_issue_quality(["Issue body"])
        self.assertEqual(score, 0.5)

    def test_review_quality_no_ai(self):
        analyzer = AgileQualityAnalyzer(api_key=None)
        analyzer.enabled = False
        score = analyzer.analyze_review_quality(["Great fix! Consider X"])
        self.assertEqual(score, 0.5)

    def test_openai_disabled_initialization(self):
        analyzer = AgileQualityAnalyzer(api_key=None)
        # se OPENAI_AVAILABLE for False no ambiente, testará isso
        self.assertFalse(analyzer.enabled)
        self.assertIsNone(analyzer.client)

if __name__ == "__main__":
    unittest.main()