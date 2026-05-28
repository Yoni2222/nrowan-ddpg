import torch

class NROWANParameterNoise:
    def __init__(self, initial_std=0.1, min_std=0.01, decay_rate=0.995):
        """
        NROWAN Parameter Noise Mechanism.
        Adds noise directly to the network's weights to encourage consistent exploration.
        """
        self.std = initial_std
        self.min_std = min_std
        self.decay_rate = decay_rate
        self.clean_weights = {}

    def apply_noise(self, model):
        """
        Saves the current clean weights and applies Gaussian noise 
        directly to the network's parameters.
        """
        self.clean_weights = {}
        # Disable gradient tracking for noise injection
        with torch.no_grad():
            for name, param in model.named_parameters():
                # 1. Save the clean, unaltered weight
                self.clean_weights[name] = param.data.clone()
                
                # 2. Generate Gaussian noise with the current standard deviation
                noise = torch.randn_like(param) * self.std
                
                # 3. Add noise directly to the weight tensor
                param.add_(noise)

    def revert_noise(self, model):
        """
        Restores the network to its clean, un-noised state.
        This must be called immediately after selecting an action!
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.clean_weights:
                    # Restore the saved clean weight
                    param.data.copy_(self.clean_weights[name])

    def decay_noise(self):
        """
        Reduces the standard deviation of the noise. 
        In NROWAN, as the agent gets safer and collects rewards, the noise decays.
        """
        self.std = max(self.min_std, self.std * self.decay_rate)