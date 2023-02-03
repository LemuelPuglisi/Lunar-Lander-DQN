from torch import nn


class QFunc(nn.Module):
    """
    """
    
    def __init__(self, 
                 action_space=4, 
                 observation_space=8,
                 hidden_space_1=128, 
                 hidden_space_2=128):
        """
        """
        super(QFunc, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(observation_space, hidden_space_1), 
            nn.ReLU(),
            nn.Linear(hidden_space_1, hidden_space_2), 
            nn.ReLU(), 
            nn.Linear(hidden_space_2, action_space)
        )
        
    def forward(self, observations):
        """
        Given the observations, it returns the Q-values of each action.
        """
        return self.ffn(observations)
    
    @property
    def device(self):
        return next(self.parameters()).device