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


    def test_root_get_and_head(self):
        get_res = self.client.get("/")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["status"], "ok")
        head_res = self.client.head("/")
        self.assertEqual(head_res.status_code, 200)

    def test_health_get_and_head(self):
        get_res = self.client.get("/health")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["service"], "pokemon-tcg-ocr")
        head_res = self.client.head("/health")
        self.assertEqual(head_res.status_code, 200)

    def test_favicon_endpoint(self):
        res = self.client.get("/favicon.ico")
        self.assertEqual(res.status_code, 204)

    def test_scan_get_info_and_head(self):
        for path in ("/scan", "/api/v1/scan"):
            get_res = self.client.get(path)
            self.assertEqual(get_res.status_code, 200)
            self.assertEqual(get_res.json()["status"], "ok")
            head_res = self.client.head(path)
            self.assertEqual(head_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
