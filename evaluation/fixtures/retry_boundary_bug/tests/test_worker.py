import unittest

from worker import execute_job


class WorkerTest(unittest.TestCase):
    def test_third_attempt_is_the_final_attempt(self):
        self.assertFalse(execute_job(3))

    def test_second_attempt_can_retry(self):
        self.assertTrue(execute_job(2))
