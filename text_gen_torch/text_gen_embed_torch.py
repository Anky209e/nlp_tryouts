import torch
import torch.nn as nn
import string
import numpy
import random
import os
import sys
import unidecode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_chars = string.printable
n_characters = len(all_chars)

file = unidecode.unidecode(open("wiki_data.txt").read())


class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(RNN,self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Input_size is max no of characters hidden size is dimension of embedding
        self.embed = nn.Embedding(input_size,hidden_size) 
        self.lstm = nn.LSTM(hidden_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)
    
    def forward(self,x,hidden,cell):

        out = self.embed(x)
        out,(hidden,cell) = self.lstm(out.unsqueeze(1),(hidden,cell))
        out = self.fc(out.reshape(out.shape[0],-1))

        return out,(hidden,cell)
    
    def init_hidden(self,batch_size):

        hidden = torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)

        return hidden,cell


class Generator():
    def __init__(self):
        self.chunk_len = 500
        self.num_epoch = 2000
        self.batch_size = 1
        self.print_every = 2
        self.hidden_size = 120
        self.num_layers = 2
        self.lr = 0.003

    def char_tensor(self,string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            # Replacing zero with character index
            tensor[c] = all_chars.index(string[c])
        return tensor
    
    def get_random_batch(self):
        # start index of our random batch
        start_idx = random.randint(0,len(file)-self.chunk_len)
        # end index of our random batch
        end_idx = start_idx + self.chunk_len + 1
        # picking text string from file using start and end index
        text_str = file[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size,self.chunk_len)
        text_target = torch.zeros(self.batch_size,self.chunk_len)

        # Targets will be next character in sequence since we will be predicting next characters
        for i in range(self.batch_size):
            text_input[i,:] = self.char_tensor(text_str[:-1])
            text_target[i,:] = self.char_tensor(text_str[1:])
        
        return text_input.long(),text_target.long()

    def generate(self,initial_str='A',prediction_len=1000,temprature=0.85):
        hidden,cell = self.rnn.init_hidden(self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str
        
        # Handeling LSTM hidden and cell if initial string is longer than 1
        for p in range(len(initial_str)-1):
            _,(hidden,cell) = self.rnn(initial_input[p].view(1).to(device),hidden,cell)

        last_char = initial_input[-1]

        for p in range(prediction_len):
            output,(hidden,cell) = self.rnn(last_char.view(1).to(device),hidden,cell)
            # Handeling temperature for predictions
            output_dist = output.data.view(-1).div(temprature).exp()
            # Finding characters with highest probs but with a little randomisation too
            top_char = torch.multinomial(output_dist,1)[0]
            predicted_char = all_chars[top_char]
            predicted+=predicted_char

            last_char = self.char_tensor(predicted_char)
        
        return predicted
    
    def save_weights(self,filename):
        torch.save(self.rnn.state_dict(),filename)


    def train(self):
        self.rnn = RNN(n_characters,self.hidden_size,self.num_layers,n_characters).to(device)

        optimizer = torch.optim.Adam(self.rnn.parameters(),lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print("==> Starting Training")

        for epoch in range(1,self.num_epoch+1):
            input,target = self.get_random_batch()

            hidden,cell = self.rnn.init_hidden(self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            input = input.to(device)
            target = target.to(device)

            # providing characters 1 by 1
            for c in range(self.chunk_len):
                output,(hidden,cell) = self.rnn(input[:,c],hidden,cell)
                loss += criterion(output,target[:,c])
            
            loss.backward()
            optimizer.step()
            loss = loss.item()/self.chunk_len

            if epoch % self.print_every == 0:
                print(f"===> Epoch:{epoch} ==> Loss:{loss}")
                f_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                initial_letter = f_chars[random.randint(0,len(f_chars)-1)]
                print(f"-----\n{self.generate(initial_str=initial_letter)}\n-----")
        self.save_weights(f"wiki_data_weights_{epoch}.pth")



name_gen_1 = Generator()
name_gen_1.train()
