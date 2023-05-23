# Torch is the main library being used for machine learning
import torch
from torch import nn
import torchmetrics

# Used to display confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

# tqdm is used for progress bar
import tqdm

# Dataset.py for class reduction
import dataset

from random import randint

from matplotlib import pyplot

from nltk import FreqDist

import datasets

# Transformers lets us easily load transformer models from huggingface
from transformers import AutoTokenizer, BertForSequenceClassification, GPT2ForSequenceClassification, DistilBertForSequenceClassification, BartForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

#Prevent the dataset changes from being cached. This requires the user to rerun the data processing each time they run this notebook but helps fix any confusion.
from datasets import disable_caching
disable_caching()

# Hide warnings to reduce clutter from output
import transformers
transformers.logging.set_verbosity_error()

import datasets
datasets.logging.set_verbosity_error()


#
# Hyperparameters
#

epochs = 4
batch_size = 128
padding_length = 40
learning_rate = 0.00003


#
# Text processing
#

def tokenize_text(dataset, tokenizer):
    return dataset.map(lambda entry: tokenizer(entry["text"], return_tensors="pt", padding='max_length', truncation=True, max_length=padding_length), batched=True)

def modify_entry_label(entry):
    label = entry["labels"]
    # Turn to an integer tensor so that the model can use it. The one hot encoding function wants integers so make sure that they are integer tensors
    label = torch.Tensor([label]).to(torch.long)
    
    # Apply one hot encoding, then flatten classes into 
    label = torch.nn.functional.one_hot(label, num_classes=14)
    
    # Flatten the multiple label encodings into one and convert to floats which the models use
    entry["labels"] = label.sum(0).float()
    
    return entry

def process_text(dataset, labels, tokenizer):
    # First we apply the tokenizer
    dataset = dataset.map(lambda entry: tokenizer(entry["text"], return_tensors="pt", padding='max_length', truncation=True, max_length=padding_length), batched=True)
    
    dataset = dataset.rename_column("labels", "labels_old")
    dataset = dataset.add_column("labels", labels.tolist())
    
    # Apply the label modification to all of the dataset entries
    dataset = dataset.map(modify_entry_label)
    
    # Remove columns we don't want
    dataset.set_format(type="torch", columns=["labels", "input_ids", "attention_mask"])
    
    return dataset


#
# Dataloader that automatically emits batches on gpu
#

class GPUDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __iter__(self):
        batches = super().__iter__()
        
        for batch in batches:
            yield {k: v.cuda() for k, v in batch.items()}


#
# The trainer class that trains the model and tests it
#

class Trainer:
    def __init__(self, train_data, valid_data, batch_size, optimizer, scheduler, print_progress):
        self.batch_size = batch_size
        self.train_data = GPUDataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.valid_data = GPUDataLoader(valid_data, batch_size=batch_size, shuffle=True)
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.history = []
        self.print_progress = print_progress
        
    # Train a single epoch 
    def train_one_epoch(self, model):
        running_loss = 0.
        
        index = 0
        
        for item in tqdm.tqdm(self.train_data):
            # Float fix
            item["labels"] = item["labels"].float()
            
            self.optimizer.zero_grad()
            
            outputs = model(**item)
            loss = outputs.loss # Do this on next line else it doesn't work
            
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss = loss.detach().item()
            
            # Record accuracy 5 times per epoch
            if index % (len(self.train_data) / 5) == 0:
                self.valid_one_epoch(model)
            
            index += 1
        
        return running_loss
    
    def valid_one_epoch(self, model):
        total_loss = 0.0
        correct = 0
        printed_first_batch = False
        
        for item in self.valid_data:
            item["labels"] = item["labels"].float()
            
            # Turn off grad mode else we can run out of GPU memory when we don't need to
            with torch.no_grad():
                outputs = model(**item)
                loss = outputs.loss
            
            total_loss += loss
            
            predicted = torch.argmax(outputs.logits, dim=1)
            truth = torch.argmax(item["labels"], dim=1)
            
            # Print out the first batch predictions to visualise the progress of the training
            if self.print_progress and not printed_first_batch:
                print("predicted:", predicted[:20])
                print("truth:", truth[:20])
                printed_first_batch = True
            
            # Calculate comparison for how close ouputs are to correct values
            correct += (predicted == truth).sum().item()
        
        val_loss = total_loss / (len(self.valid_data.dataset) + 1)
                
        accuracy = 100 * (correct / len(self.valid_data.dataset))

        # Update history
        self.history.append(accuracy)
        
        return val_loss, correct, accuracy
        
    def train(self, model, epochs):
        if self.print_progress:
            print("TRAINING")
        
        for epoch in range(epochs):
            if self.print_progress:
                print(f"EPOCH: {epoch + 1} lr: {self.optimizer.param_groups[0]['lr']}")
            
            model.train(True)
            train_loss = self.train_one_epoch(model)
            model.train(False)
            
            val_loss, correct, accuracy = self.valid_one_epoch(model)
            
            if self.print_progress:
                print(f"Train: {train_loss} Valid: {val_loss} Accuracy: {(accuracy):>0.2f}%")
                print("correct", correct, "out of", len(self.valid_data.dataset))
                print()
            
            self.scheduler.step(train_loss)
            
            # Help prevent gpu memory fragmentation
            torch.cuda.empty_cache()
            
    def test(self, model, test_data):
        data_loader = GPUDataLoader(test_data, batch_size=batch_size, shuffle=True)
        
        # The predicted and truth tensors that we will add to for the confusion matrix
        all_predicted = torch.Tensor()
        all_truth = torch.Tensor()
        
        correct = 0
        
        for item in tqdm.tqdm(data_loader):
            item["labels"] = item["labels"].float()
            
            with torch.no_grad():
                outputs = model(**item)

            predicted = torch.argmax(outputs.logits, dim=1)
            all_predicted = torch.cat((all_predicted, predicted.cpu()))
            
            truth = torch.argmax(item["labels"], dim=1)
            all_truth = torch.cat((all_truth, truth.cpu()))
            
            correct += (predicted == truth).sum().item()
            
        accuracy = 100 * (correct / len(data_loader.dataset))
            
        print(f"correct: {correct}/{len(data_loader.dataset)} accuracy: {accuracy}")
        
        confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=label_id_count)(all_predicted.long(), all_truth.long())
        
        # Display the accuracy for each class
        per_class_accuracy = confusion_matrix.diag()/confusion_matrix.sum(1)
        per_class_accuracy = dict(zip(new_labels, per_class_accuracy.tolist())) # Turn into dict of strings to accuracy
        
        # Display the confusion matrix
        display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix.numpy(), display_labels=new_labels)
        display.plot(xticks_rotation='vertical')
        pyplot.show()

        print(f"per class accuracy: {per_class_accuracy}")


#
# Load the dataset
#

train_data = datasets.load_dataset('go_emotions', split="train")
valid_data = datasets.load_dataset('go_emotions', split="validation")
test_data = datasets.load_dataset('go_emotions', split="test")

train_labels, valid_labels, test_labels = dataset.load_labels(
    dataset.labels_to_list(train_data),
    dataset.labels_to_list(valid_data),
    dataset.labels_to_list(test_data)
)


#
# Load the model
#

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=14, problem_type="multi_label_classification").cuda()


#
# Train the model on the dataset
#

train_data = process_text(train_data, train_labels, tokenizer)
valid_data = process_text(valid_data, valid_labels, tokenizer)
test_data = process_text(test_data, test_labels, tokenizer)

# Optimizer and scheduler used in https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613.
# They seem to work very well for bert fine tuning
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(len(train_data)/batch_size)*epochs)

trainer = Trainer(train_data, valid_data, batch_size, optimizer, scheduler, False)

trainer.train(model, epochs)


#
# Save the model and tokenizer
#

model.save_pretrained("model")
tokenizer.save_pretrained("model")