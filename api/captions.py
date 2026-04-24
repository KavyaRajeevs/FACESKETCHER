import pickle

# Replace 'vocab.pkl' with your actual pickle file path
pickle_file_path = 'captions_org.pickle'

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    vocabulary = pickle.load(file)

# Print the vocabulary
print("Vocabulary Loaded:")
if isinstance(vocabulary, dict):  # If it's a dictionary
    for key, value in vocabulary.items():
        print(f"{key}: {value}")
elif isinstance(vocabulary, list):  # If it's a list
    for word in vocabulary:
        print(word)
else:  # If it's another data structure
    print(vocabulary)
