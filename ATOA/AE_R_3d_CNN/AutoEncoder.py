import torch
import torch.nn as nn
import torch.optim as optim
import CNN_2d as AE

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = AE.Encoder_CNN_2D(mask_size=160, dropout_p=0.0)
        self.decoder = AE.Decoder_CNN_2D(dropout_p=0.0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#Initialize the model
autoencoder = AutoEncoder()

#Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

#Training Autoencoder
def train(model, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs in dataloader:
            #Assuming that the inputs are already the correct batch size x 1 x 160 x 160 tensor, and the elements are 0 or 1
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs) 

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()  

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")


