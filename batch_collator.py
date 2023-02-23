import torch
import numpy

class BatchCollator:
    def __init__(self, sample_types=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.initialised = False
        self.sample_types = sample_types # Override the initialise method and just use this 

    def initialise(self, samples):
        sample_keys = set()
        for sample in samples:
            sample_keys.update(sample.keys())

        sample_types = {}
        for key in sample_keys:
            types = [type(sample[key]) if key in sample else None for sample in samples]
            data_type = types[0] if len(set(types)) == 1 else 'unknown' # In the event of unknown the collator will just return a list of the values
            sample_types[key] = data_type

        return sample_types

    def __call__(self, samples):
        # Still want to return a dictionary indicating each value from the dataset
        batched_samples = {}

        # The collator needs to calculate how to treat each value given the first time it is called 
        # this is so that the data collator can handle different data input types
        # Only do this if a mapping of key -> data type wasn't provided
        sample_types = self.sample_types
        if sample_types is None:
            sample_types = self.initialise(samples)

        for key in sample_types:
            batched_samples[key] = self.batch_values([sample[key] if key in sample else None for sample in samples], sample_types[key])

        return batched_samples

    def batch_values(self, value_list, value_type):
        # Strings and unknown types should just be returned as list of values
        if value_type == str or value_type == 'unknown':
            return value_list

        # Tensor and Numpy arrays should be grouped together as one single torch tensor 
        # this value should also be on the correct device
        # Numpy arrays need to be converted to tensors first 
        if value_type == numpy.ndarray:
            value_list = [torch.from_numpy(value) for value in value_list]
            value_type = type(value_list[0])
        if value_type == torch.Tensor:
            # First check if this value needs padding
            lengths = [len(value) for value in value_list]
            padding_required = len(set(lengths)) > 1

            if padding_required:
                # TODO: This may not work for all types that need padding. pad_packed_sequence is worth looking into 
                max_len = max(lengths)
                value_tensor = torch.zeros(len(value_list), max_len, device=self.device)
                for i, value in enumerate(value_list):
                    value_tensor[i,:lengths[i]] = value
                return value_tensor
            else:
                value_tensor = torch.zeros(len(value_list), lengths[0], device=self.device)
                for i, value in enumerate(value_list):
                    value_tensor[i,:] = value
                return value_tensor

        # Float values should be turned to torch tensors as well
        # these are labels and only need the one dimension
        if value_type == float:
            value_tensor = torch.zeros(len(value_list), dtype=torch.float32, device=self.device)
            for i, value in enumerate(value_list):
                value_tensor[i] = value
            return value_tensor

        # Int values should be turned to torch tensors as well (but as long tensors)
        # these are labels and only need the one dimension
        if value_type == int or value_type == numpy.int64:
            value_tensor = torch.zeros(len(value_list), dtype=torch.long, device=self.device)
            for i, value in enumerate(value_list):
                value_tensor[i] = value
            return value_tensor
