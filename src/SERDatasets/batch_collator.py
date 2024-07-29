import torch
import numpy
import traceback

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
            def get_type(sample):
                if key not in sample:
                    return None
                if type(sample[key]) == torch.Tensor:
                    return sample[key].dtype
                return type(sample[key])
            types = [get_type(sample) for sample in samples]
            set_of_types = set(types)
            if len(set_of_types) > 1:
                # Some samples may be types that are mostly shared and should be combined
                int_types = [torch.long, torch.int64, int, numpy.int64]
                float_types = [torch.float32, float, numpy.float32]
                all_float = True
                all_int = True
                for type_var in set_of_types:
                    all_int &= type_var in int_types
                    all_float &= type_var in float_types
                assert not all_int or not all_float
                if all_int:
                    types = [int]
                if all_float:
                    types = [float]
            set_of_types = set(types)
            data_type = types[0] if len(set_of_types) == 1 else 'unknown' # In the event of unknown the collator will just return a list of the values
            if data_type == torch.float32: # If all values have these types then this is already a tensor and needs to be treated specially
                data_type = torch.Tensor
            if data_type == torch.long:
                data_type = torch.Tensor
            sample_types[key] = data_type

        return sample_types

    def __call__(self, samples):
        # Still want to return a dictionary indicating each value from the dataset
        batched_samples = {}

        # All the numpy.ndarrays need to be treated as torch arrays
        for i, sample in enumerate(samples):
            for key in sample:
                # Want to keep kde 2d probability as a numpy array as it is only used in f-sim comparisons 
                if type(sample[key]) == numpy.ndarray and key != 'kde_2d_probability':
                    samples[i][key] = torch.from_numpy(sample[key])

        # The collator needs to calculate how to treat each value given the first time it is called 
        # this is so that the data collator can handle different data input types
        # Only do this if a mapping of key -> data type wasn't provided
        sample_types = self.sample_types
        if self.sample_types is None:
            sample_types = self.initialise(samples)

        for key in sample_types:
            # print('batching', key, 'as', sample_types[key])
            try:
                batched_samples[key] = self.batch_values([sample[key] if key in sample else None for sample in samples], list if key == 'kde_2d_probability' else sample_types[key]) # Manually force kde_2d_probability to be returned as a list
                # print('became', batched_samples[key])
            except TypeError as e:
                # Something went wrong with the presumed type, store this item as a normal list 
                print(f'Error batching {key} -- returning as a normal python list in batch')
                print('error:')
                print(e)
                print(traceback.print_exc())
                batched_samples[key] = [sample[key] if key in sample else None for sample in samples]

        return batched_samples

    def batch_values(self, value_list, value_type):
        # Strings and unknown types should just be returned as list of values
        if value_type == dict or value_type == list or value_type == numpy.str_ or value_type == str or value_type == 'unknown' or value_type == bool:
            return value_list

        # Tensor and Numpy arrays should be grouped together as one single torch tensor 
        # this value should also be on the correct device
        # Numpy arrays need to be converted to tensors first 
        if value_type == numpy.ndarray:
            value_list = [torch.from_numpy(value) for value in value_list]
            value_type = type(value_list[0])
        if value_type == torch.Tensor:
            # First check if this value needs padding
            lengths = []
            
            lengths = [len(value) for value in value_list]
            padding_required = len(set(lengths)) > 1

            if padding_required:
                # TODO: This may not work for all types that need padding. consider pad_packed_sequence.
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
        if value_type == float or value_type == numpy.float64:
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

        return 'ERROR: No definition for this data type for batching'

# A collator that merges self-report-X and X data with a mask where the mask is true for values that are from self-report and false for values from perception of other
# useful for things like ExpeR where both types are merged and values may be missing in each preventing the default collator from creating torch tensors correctly
class SelfReportBatchCollator(BatchCollator):
    def __call__(self, samples):
        batched_samples = super().__call__(samples)
        # Merge 'X' and 'self-report-X', for act, act_bin, val, and val_bin
        # first find the mask 
        self_report_mask = batched_samples['dataset_id'] == -1
        batched_samples['self_report_mask'] = self_report_mask

        # Now create merged tensors 
        for value in ['act', 'act_bin', 'val', 'val_bin']:
            dtype = torch.long if '_bin' in value else torch.float32
            merge_tensor = torch.zeros(len(self_report_mask), dtype=dtype, device=self.device)
            for i in range(len(self_report_mask)):
                merge_tensor[i] = batched_samples[f'self-report-{value}'][i] if self_report_mask[i] else batched_samples[value][i]
            batched_samples[f'merged_{value}'] = merge_tensor

        return batched_samples
