import requests
import unittest

class TestApi(unittest.TestCase):

    def setUp(self):
        self.adressToMakeRequest = 'http://127.0.0.1:5000'
        self.InvalidInputMessage = "The message must be JSON in the form json={\'comment\': string_to_predict}."
        self.badRequestSatusCode = 400
        self.okStatusCode = 200

    #test a successfull response
    def testComment(self):
        payload = {
            'comment': "Subway is the smartest option unless you have tones of stuffs to carry !"
        }
        response = requests.post(self.adressToMakeRequest, json=payload)
        responseJSON = response.json()
        self.assertEqual(responseJSON["success"], True)
        self.assertIn(responseJSON["prediction"], ["admiration", "amusement", "anger", "annoyance", "curiosity", "disapproval", "gratitude", "joy", "love", "optimism", "remorse", "sadness", "surprise", "neutral"])
        self.assertEqual(len(response.json().keys()), 2)
        self.assertEqual(response.status_code, self.okStatusCode)

    #test an invalid format. The payload is a dict thatn has more than 1 key-value pairs which is invalid.
    def testWrongFormatMoreThanOneKey(self):
        payload = {
            'comment': "Subway is the smartest option unless you have tones of stuffs to carry !",
            "wrong": "wrong comment"
        }
        response = requests.post(self.adressToMakeRequest, json=payload)
        responseJSON = response.json()
        self.assertEqual(responseJSON["success"], False)
        self.assertEqual(responseJSON["error"], self.InvalidInputMessage)
        self.assertEqual(len(response.json().keys()), 2)
        self.assertEqual(response.status_code, self.badRequestSatusCode)

        #test an invalid format. The payload is a dict thatn has more than 1 key-value pairs which is invalid.
    def testWrongFormatZeroKeys(self):
        payload = {}
        response = requests.post(self.adressToMakeRequest, json=payload)
        responseJSON = response.json()
        self.assertEqual(responseJSON["success"], False)
        self.assertEqual(responseJSON["error"], self.InvalidInputMessage)
        self.assertEqual(len(response.json().keys()), 2)
        self.assertEqual(response.status_code, self.badRequestSatusCode)
    
    #test an invalid format. The payload is a dict that has one key-value pair but the key of the dict is invalid.
    def testWrongFormatStringKey(self):
        payload = {
            'commmentadfsf': "Subway is the smartest option unless you have tones of stuffs to carry !"
        }
        response = requests.post(self.adressToMakeRequest, json=payload)
        responseJSON = response.json()
        self.assertEqual(responseJSON["success"], False)
        self.assertEqual(responseJSON["error"], self.InvalidInputMessage)
        self.assertEqual(len(response.json().keys()), 2)
        self.assertEqual(response.status_code, self.badRequestSatusCode)

    #test an invalid format. The payload is a dict that has one key-value pair but the value of the comment key is an integer which is invalid.
    def testWrongFormatIntegerValue(self):
        payload = {
            'commment': 8
        }
        response = requests.post(self.adressToMakeRequest, json=payload)
        responseJSON = response.json()
        self.assertEqual(responseJSON["success"], False)
        self.assertEqual(responseJSON["error"], self.InvalidInputMessage)
        self.assertEqual(len(response.json().keys()), 2)
        self.assertEqual(response.status_code, self.badRequestSatusCode)

    #test an invalid format. The payload is a string which is invalid.
    def testWrongFormatStringPayload(self):
        payload = "wrong"
        response = requests.post(self.adressToMakeRequest, json=payload)
        responseJSON = response.json()
        self.assertEqual(responseJSON["success"], False)
        self.assertEqual(responseJSON["error"], self.InvalidInputMessage)
        self.assertEqual(len(response.json().keys()), 2)
        self.assertEqual(response.status_code, self.badRequestSatusCode)

    #test an invalid format. The payload is an integer which is invalid
    def testWrongFormatIntegerPayload(self):
        payload = 5
        response = requests.post(self.adressToMakeRequest, json=payload)
        responseJSON = response.json()
        self.assertEqual(responseJSON["success"], False)
        self.assertEqual(responseJSON["error"], self.InvalidInputMessage)
        self.assertEqual(len(response.json().keys()), 2)
        self.assertEqual(response.status_code, self.badRequestSatusCode)

    #test an invalid format. The payload is a dict that has one key-value pair but the value of the comment key is an empty string which is invalid
    def testWrongFormatEmptyStringValue(self):
        payload = {
            'comment': ""
        }
        response = requests.post(self.adressToMakeRequest, json=payload)
        responseJSON = response.json()
        self.assertEqual(responseJSON["success"], False)
        self.assertEqual(responseJSON["error"], self.InvalidInputMessage)
        self.assertEqual(len(response.json().keys()), 2)
        self.assertEqual(response.status_code, self.badRequestSatusCode)