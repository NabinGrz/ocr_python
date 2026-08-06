"""HTTP contract regressions for scan input validation."""

import unittest

from fastapi.testclient import TestClient

import api


class TestAPIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)

    def test_invalid_image_payload_returns_400(self):
        response = self.client.post(
            "/api/v1/scan",
            files={"file": ("invalid.png", b"not an image", "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Could not decode image file.")

    def test_non_image_content_type_returns_400(self):
        response = self.client.post(
            "/api/v1/scan",
            files={"file": ("invalid.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
