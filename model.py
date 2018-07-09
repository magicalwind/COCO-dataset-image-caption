import torch
import torch.nn as nn
import torchvision.models as models
import random


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        captions = captions[:,:-1]
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #print (embeddings.shape)
        #embeddings = embeddings.permute(1,0,2)
        #print (embeddings.shape)
        out,_ = self.lstm(embeddings)
        #out = out.transpose(0,1)
        #print (out.shape)
        out = self.linear(out)
        
        return out
    
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        created_wordid = []
        
        A = random.randint(10,max_len)
        for i in range(A):
            hiddens,states= self.lstm(inputs,states)
            output =self.linear(hiddens.squeeze(1))
            #output = self.linear(hiddens)
            #output = output.squeeze(1)
            _,wordid = output.max(1)
            prediction = wordid.item()
            created_wordid.append(prediction)
            inputs = self.embed(wordid)
            inputs = inputs.unsqueeze(1)
        return created_wordid
            
        
        
        
        
        
        
        
        
        
        
        