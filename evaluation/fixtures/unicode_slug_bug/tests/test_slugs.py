import unittest

from slugs import make_slug


class SlugTest(unittest.TestCase):
    def test_accented_title_is_ascii(self):
        self.assertEqual(make_slug("Crème Brûlée"), "creme-brulee")
