import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina' # to make graphs sharper
from pprint import pprint # pretty print
# Title of the app
st.title("Parameter Selection and Text Input App")
class Nextword(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, number_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(block_size * emb_dim, hidden_size))
        for i in range(number_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, vocab_size))

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.size(0), -1)
        for l in self.layers:
            x = l(x)
            x = nn.ReLU()(x)
        return x

num_layers=10
def load_model(Emb_dim,cotext,af,rsd):
    model = Nextword(Context, len(wtoi), Emb_dim, 1024, num_layers)
    path=f"'/Users/pranavthakkar/Desktop/Ml_3/model_{Emb_dim}_{Context}_{af}_{rsd}.pth'"
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


#text box for user to input the text
x_test = st.text_area("Enter the input text:", value="", height=150)

# Dropdown (Selectbox) for Embedding Dimension selection
Emb_dim = st.selectbox(
    "Select a Embedding Dimension:",
    ("32", "64")
)
# Dropdown (Selectbox) for Context Length selection
Context = st.selectbox(
    "Select a Context Length:",
    ("5", "7")
)
# Dropdown (Selectbox) for Activation Function selection
af = st.selectbox(
    "Select a Activation Function:",
    ("ReLU", "Sin")
)
# Dropdown (Selectbox) for Context Length selection
rsd = st.selectbox(
    "Select a Random Seed",
    ("96", "42")
)

k_words = st.number_input("Number of words to predict:", min_value=1, value=1000)
