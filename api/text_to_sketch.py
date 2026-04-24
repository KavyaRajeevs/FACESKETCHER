import torch
import torch.nn as nn
from torch.autograd import Variable
from nltk.tokenize import RegexpTokenizer
import numpy as np
from PIL import Image
import os

from model import RNN_ENCODER, G_NET
from config import cfg, cfg_from_file

class TextToSketchGenerator:
    def __init__(self, text_encoder_path, generator_path, cfg_path=None):
        """Initialize the generator with model paths"""
        # Force CPU usage
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        # Load configuration if provided
        if cfg_path:
            cfg_from_file(cfg_path)

        # Initialize tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')

        # Load vocabulary
        self.load_vocabulary()

        # Initialize and load models
        self.text_encoder = self.load_text_encoder(text_encoder_path)
        self.generator = self.load_generator(generator_path)

        # Set models to evaluation mode
        self.text_encoder.eval()
        self.generator.eval()

    def load_vocabulary(self):
        """Load vocabulary from captions.pickle"""
        import pickle

        # Use vocabulary file bundled in this api directory
        vocab_path = os.path.join(os.path.dirname(__file__), "captions_org.pickle")

        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                self.ixtoword = ixtoword
                self.wordtoix = wordtoix
                self.n_words = n_words
        else:
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")

    def load_text_encoder(self, path):
        """Load the text encoder model"""
        print(f"Expected vocabulary size: {self.n_words}")  # Debugging

        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)

        state_dict = torch.load(path, map_location=self.device)  # Load to CPU
        print(f"Checkpoint vocabulary size: {state_dict['encoder.weight'].shape}")  # Debugging

        # If the vocab sizes don't match, manually resize
        if text_encoder.encoder.weight.shape[0] != state_dict['encoder.weight'].shape[0]:
            print("Mismatch detected! Adjusting model size...")

            new_n_words = state_dict['encoder.weight'].shape[0]
            text_encoder = RNN_ENCODER(new_n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)

        text_encoder.load_state_dict(state_dict)
        text_encoder = text_encoder.to(self.device)  # Move to CPU
        return text_encoder

    def load_generator(self, path):
        """Load the generator model"""
        netG = G_NET()
        state_dict = torch.load(path, map_location=self.device)  # Load to CPU
        netG.load_state_dict(state_dict)
        netG = netG.to(self.device)  # Move to CPU
        return netG

    def preprocess_text(self, text):
        """Preprocess input text similar to training data"""
        tokens = self.tokenizer.tokenize(text.lower())
        print(f"Tokenized words: {tokens}")  # Debugging step

        cap = []
        for token in tokens:
            if token in self.wordtoix:
                cap.append(self.wordtoix[token])

        if len(cap) == 0:
            raise ValueError("No words found in vocabulary. Please check captions.pickle.")

        cap = torch.LongTensor(cap).unsqueeze(0).to(self.device)  # Move to CPU
        cap_len = torch.LongTensor([len(cap[0])]).to(self.device)  # Move to CPU

        return cap, cap_len

    def generate_sketch(self, text, save_path=None):
        """Generate sketch from input text"""
        # Preprocess text
        captions, cap_lens = self.preprocess_text(text)

        # Generate embeddings
        batch_size = captions.size(0)
        hidden = self.text_encoder.init_hidden(batch_size)
        # Move hidden states to CPU
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(self.device) for h in hidden)
        else:
            hidden = hidden.to(self.device)
            
        words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)

        # Create mask
        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]

        # Generate noise on CPU
        noise = torch.randn(batch_size, cfg.GAN.Z_DIM).to(self.device)

        # Generate fake images
        with torch.no_grad():
            fake_imgs, _, _, _ = self.generator(noise, sent_emb, words_embs, mask)

        # Process and save/return the generated image
        im = fake_imgs[-1][0].data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)

        if save_path:
            im.save(save_path)

        return im

def main():
    # Force CPU usage
    print("Using CPU for computation")

    # Paths to your saved models
    text_encoder_path = 'C:\\Users\\Suraij PC\\FaceFinder\\Face_Finder\\my-app\\api\\models\\text_encoder600.pth'
    generator_path = 'C:\\Users\\Suraij PC\\FaceFinder\\Face_Finder\\my-app\\api\\models\\netG_epoch_600.pth'
    cfg_path = "C:\\Users\\Suraij PC\\FaceFinder\\Face_Finder\\my-app\\api\\config\\train_sketch_18_4.yml"

    # Initialize generator
    generator = TextToSketchGenerator(text_encoder_path, generator_path, cfg_path)

    # Get input from user
    text = input("Enter facial description: ")

    # Generate timestamp for filename
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"generated_sketch_{timestamp}.png"
    
    # Create output directory if it doesn't exist
    output_dir = "D:/Facefinder/LAGAN/testing_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Full path with timestamp
    full_output_path = os.path.join(output_dir, output_path)

    # Generate sketch
    sketch = generator.generate_sketch(text, save_path=full_output_path)
    print(f"Sketch generated and saved to {full_output_path}")

    # Display sketch if running in notebook environment
    try:
        from IPython.display import display
        display(sketch)
    except ImportError:
        sketch.show()

if __name__ == "__main__":
    main()