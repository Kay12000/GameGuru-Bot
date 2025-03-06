import unittest
import io
from flask import Flask
from app.routes import app_routes

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(app_routes)
        self.client = self.app.test_client()

    def test_index(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_upload_file(self):
        data = {
            'file': (io.BytesIO(b"some initial text data"), 'test.txt')
        }
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)

    def test_get_response(self):
        data = {
            'question': '¿Cuál es mi anime favorito?',
            'user_id': '12345'
        }
        response = self.client.post('/get_response', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('response', response.json)

if __name__ == '__main__':
    unittest.main()