from locust import HttpUser, task
from dataset import EmotionsDataset
from random import randint


class StressTest(HttpUser):
    dataset = EmotionsDataset(split='test')

    @task
    def get_prediction(self):
        random_comment = randint(0, len(self.dataset)-1)
        self.client.post(url='/', json={'comment':self.dataset[random_comment][0]})
