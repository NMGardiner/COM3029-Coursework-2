from locust import HttpUser, task
from dataset import EmotionsDataset
from random import randint


class StressTest(HttpUser):
    host = 'http://localhost:5000'
    dataset = EmotionsDataset(split='test')
    longest_comment = max(dataset.x, key=len)

    @task(1)
    def load_test(self):
        random_comment = randint(0, len(self.dataset)-1)
        self.client.post(url='/', json={'comment':self.dataset[random_comment][0]})

    @task(2)
    def volume_test(self):
        self.client.post(url='/', json={'comment':self.longest_comment})
