import os
import pprint
import argparse
from tqdm import tqdm
import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize
import copy
import torch.nn as nn
import torch.nn.functional as F

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import train_main, load_data, load_clip, preprocess_text
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor

from transformers import AutoTokenizer, AutoModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=512)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="cxrbert")
    parser.add_argument('--model_path', type=str, default=None, help="Path to the model checkpoint for loading pre-trained weights.")
    parser.add_argument('--moco_weight', type=float, default=1.0)
    # rest of your arguments
    args = parser.parse_args()
    return args

class CLIPMoCoWrapper(nn.Module):
    def __init__(self, clip_model, cxr_bert_model, cxr_bert_tokenizer, embed_dim, queue_size=65536, momentum=0.999, temperature=0.07):
        super().__init__()
        self.query_encoder = clip_model  # Pretrained CLIP model as query encoder (use visual encoder only)
        self.cxr_bert_model = cxr_bert_model
        self.cxr_bert_tokenizer = cxr_bert_tokenizer
        self.momentum_encoder = copy.deepcopy(clip_model)  # Momentum encoder
        self.embed_dim = embed_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Add a linear projection layer to match dimensions
        self.text_projection = nn.Linear(cxr_bert_model.config.hidden_size, embed_dim)

        self.register_buffer("queue", torch.randn(embed_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Freeze the momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False

    @property
    def context_length(self):
        # Assuming the query_encoder has an attribute `context_length`
        return self.query_encoder.context_length

    def _momentum_update(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, images):
        # Compute query features
        query_features = self.query_encoder.encode_image(images)

        # Compute key features with the momentum encoder
        with torch.no_grad():
            self._momentum_update()  # Update the momentum encoder
            key_features = self.momentum_encoder.encode_image(images).detach()

        return query_features, key_features

    def compute_contrastive_loss(self, query_features, key_features):
        query_features = query_features.float()  # Ensure query_features is Float
        key_features = key_features.float()  # Ensure key_features is Float
        self.queue = self.queue.float()  # Ensure the queue is Float and normalized
        self.queue = F.normalize(self.queue, dim=0)
    
        positive = torch.matmul(query_features, key_features.T).unsqueeze(-1)  # [batch_size, 1, 1]
        negatives = torch.matmul(query_features, self.queue)  # [batch_size, queue_size]
    
        positive = positive.squeeze(-1)  # Adjust positive to [batch_size, 1] if necessary
    
        logits = torch.cat([positive, negatives], dim=1)  # Ensure compatible shapes for concatenation
        logits /= self.temperature
    
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query_features.device)
        loss = F.cross_entropy(logits, labels)
    
        return loss

    def encode_image(self, images):
        """
        Encode image inputs using the query encoder's encode_image method.

        Parameters:
        - images: a batch of image inputs.

        Returns:
        - Image features encoded by the query encoder.
        """
        return self.query_encoder.encode_image(images)    

    def encode_text(self, text):
        """
        Encode text inputs using the CXRBERT model.

        Parameters:
        - text: a batch of text inputs.

        Returns:
        - Text features encoded by the CXRBERT model.
        """
        inputs = self.cxr_bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.context_length).to(next(self.cxr_bert_model.parameters()).device)
        outputs = self.cxr_bert_model(**inputs)
        text_features = outputs.last_hidden_state[:, 0, :]  # Use the representation of the [CLS] token
        text_features = self.text_projection(text_features)  # Project to the same dimension as image features
        return text_features

def model_pipeline(config, verbose=0): 
    # make the model, data, and optimization problem
    model, data_loader, device, criterion, optimizer = make(config)

    # and use them to train the model
    train(model, data_loader, device, criterion, optimizer, config)

    # save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint_cxrBERT.pt')
    save(model, model_path)

    if verbose: 
        print(model)
    return model

def make(config):
    pretrained = not config.random_init
    data_loader, device = load_data(config.cxr_filepath, config.txt_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression")
    model = load_clip(model_path=None, pretrained=pretrained, context_length=config.context_length)
    #from checkpoint
    config.model_path = "/home/woody/iwi5/iwi5190h/CheXzero/checkpoints/chexzero_weights/best_64_5e-05_original_22000_0.864.pt"
    model.load_state_dict(torch.load(config.model_path))
    print("Loaded model from checkpoint:", config.model_path)
    

    # Load saved BioBERT model and tokenizer
    #cxr_bert_model = BertModel.from_pretrained("/home/woody/iwi5/iwi5190h/CheXzero/saved_models/text_model")
    #cxr_bert_tokenizer = BertTokenizer.from_pretrained("/home/woody/iwi5/iwi5190h/CheXzero/saved_models/text_tokenizer")
    # Load CXRBERT model and tokenizer
    cxr_bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
    cxr_bert_model = AutoModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)

    # Initialize MoCo wrapper with CXRBERT
    moco_wrapper = CLIPMoCoWrapper(clip_model=model, cxr_bert_model=cxr_bert_model, cxr_bert_tokenizer=cxr_bert_tokenizer, embed_dim=512)
    moco_wrapper.to(device)
    print('moco_bert_wrapper loaded')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(moco_wrapper.parameters(), lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(moco_wrapper.parameters(), lr=config.lr, momentum=config.momentum)
    return moco_wrapper, data_loader, device, criterion, optimizer

def train(model, loader, device, criterion, optimizer, config): 
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir): 
        # Create a new folder if not exists
        os.makedirs(model_save_dir)
    
    # Run training
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = config.log_interval
    highest_val_auc = 0 # save highest mean auc
    
    for epoch in range(config.epochs):
        running_loss = 0.0 # running loss over batch
        for data in tqdm(loader):
            # get the images
            images = data['img']

            texts = data['txt']
            texts = preprocess_text(texts, model) 
            
            # perform step for a single batch
            loss = train_batch(images, texts, model, device, criterion, optimizer)
            example_ct +=  len(images)
            batch_ct += 1
            running_loss += loss.item()

            # Report metrics every `report_freq` batch
            if (batch_ct % report_freq) == 0:
                train_log(running_loss / report_freq, example_ct, epoch)
                running_loss = 0.0
            
            if (batch_ct % config.save_interval) == 0: 
                model_path = os.path.join(model_save_dir, "checkpoint_{batch_ct}.pt".format(
                    batch_ct=str(batch_ct), 
                ))
                print("Saved checkpoint to: ", model_path)
                save(model, model_path)
                
def train_batch(images, texts, model, device, criterion, optimizer):
    images = images.to(device)

    # Ensure texts are in the correct format
    if isinstance(texts, list):
        texts = [str(text) for text in texts]
    elif isinstance(texts, torch.Tensor):
        texts = texts.tolist()
        texts = [str(text) for text in texts]
    else:
        raise ValueError(f"Unexpected type for texts: {type(texts)}. Expected list or torch.Tensor.")

    # Forward pass ➡
    # Forward pass to get MoCo features
    query_features, key_features = model(images)  # This should invoke the MoCo-specific forward method

    # Forward pass to get image-text features
    image_features, text_features = model.encode_image(images), model.encode_text(texts)
    # Convert image features to float precision to match text features
    if image_features.dtype != text_features.dtype:
        image_features = image_features.float()

    # Compute logits for image-text comparison
    logits_per_image = torch.matmul(image_features, text_features.T)
    logits_per_text = logits_per_image.T  # Transpose due to symmetry in cosine similarity
    
    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)
    
    # Compute loss (# output of momentum & text encoder)
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt)/2 # avg. img and txt loss
    weight = 0.1
    contrastive_loss = model.compute_contrastive_loss(query_features, key_features)
    total_loss = loss + weight * contrastive_loss

    # Backward pass ⬅
    optimizer.zero_grad()
    total_loss.backward()
    
    # Step with optimizer
    optimizer.step()
        
    return total_loss

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
def save(model, path): 
    torch.save(model.state_dict(), path)
    
if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)
    

