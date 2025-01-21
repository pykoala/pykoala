import unittest
import os

from pykoala.instruments.mock import mock_rss
from pykoala.corrections.throughput import Throughput, ThroughputCorrection

class TestThroughput(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting up RSS objects for testing")
        self.rss_1 = mock_rss()
        self.rss_2 = mock_rss()
        self.rss_3 = mock_rss()
        self.rss_list = [self.rss_1, self.rss_2, self.rss_3]
    
    def test_correction(self):
        self.correction = ThroughputCorrection.from_rss(
            self.rss_list, clear_nan=True, medfilt=10)
        self.correction.throughput.to_fits("./test_throughput.fits")
        throughput = Throughput.from_fits("./test_throughput.fits")
        os.unlink("./test_throughput.fits")
        self.assertTrue(
            (self.correction.throughput.throughput_data == throughput.throughput_data).all())
        self.assertTrue((self.correction.throughput.throughput_error == throughput.throughput_error).all())

if __name__ == "__main__":
    unittest.main()
