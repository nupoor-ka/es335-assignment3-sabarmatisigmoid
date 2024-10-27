import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import pickle

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


with open('streamlit_app/int2word.pkl', 'rb') as f:
    int2word = pickle.load(f)

with open('streamlit_app/word2int.pkl', 'rb') as f:
    word2int = pickle.load(f)

num_layers=10
def load_model(Emb_dim,context,af,rsd):
    model = Nextword(context, len(word2int), Emb_dim, 1024, num_layers)
    path=f"streamlit_app/model_{Emb_dim}_{context}_{af}_{rsd}.pth"
    model.load_state_dict(torch.load(path), map_location = torch.device("cpu"))
    model.eval()
    return model

def set_context(x_text,Context):
    x_l = x_text.split()
    inp_con = x_l[-Context:]
    inp_num=[]
    for i in  inp_con:
        if i not in word2int.keys():
            num=18335
        else:
            num=word2int[i]
        inp_num.append(num)
    return inp_num

def generate_text(model, int2word, num_words , context):
    gen_text = ''
    text_len = 0
    while text_len < num_words:
      x = torch.tensor(context).view(1, -1)
      y_pred = model(x)
      ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
      wor = int2word[ix]
      gen_text += wor + ' '
      context = context[1:] + [ix]
      if wor!='.':
        text_len += 1

    return gen_text



#text box for user to input the text
x_test = st.text_area("Enter the input text:", value="", height=150)

# Dropdown (Selectbox) for Embedding Dimension selection
Emb_dim = st.selectbox(
    "Select an Embedding Dimension:",
    [32, 128]
)
# Dropdown (Selectbox) for Context Length selection
Context = st.selectbox(
    "Select a Context Length:",
    [4, 8]
)
# Dropdown (Selectbox) for Activation Function selection
af = st.selectbox(
    "Select an Activation Function:",
    ("ReLU", "Sin")
)
# Dropdown (Selectbox) for Context Length selection
rsd = st.selectbox(
    "Select a Random Seed",
    [96, 42]
)

num_words = st.number_input("Number of words to predict:", min_value=1, value=1000)

model=load_model(Emb_dim,Context,af,rsd)
input_test=set_context(x_test,Context)
output=generate_text(model, int2word, num_words , input_test)

if st.button('Predict'):
    st.write(f"Predicted next word(s): {output}")

