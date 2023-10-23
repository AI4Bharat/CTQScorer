from torch.utils.data import Dataset
from utils.constants import BATCH_SIZE

"""
Datastructure to run the input samples in a batch
"""
class MTDataset(Dataset):
    
    def __init__(self):
        self.prompts = []
        self.inputs = []
    
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

    def addprompt(self, prompt):
        self.prompts.append(prompt)

    def addinput(self, input):
        self.inputs.append(input)

    def getNumTokens(self, start_index):
        inputs_in_batch = self.inputs[start_index : start_index + BATCH_SIZE]
        inputs_in_batch = list(map(lambda x: x.split(), inputs_in_batch))
        tokens_in_input_batch = list(map(lambda x: len(x), inputs_in_batch))
        max_tokens = max(tokens_in_input_batch)

        return int(max_tokens * 1.5)