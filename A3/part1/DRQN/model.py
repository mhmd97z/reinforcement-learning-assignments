import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import gamma, device, batch_size

class DRQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DRQN, self).__init__()
        # The inputs are two integers giving the dimensions of the inputs and outputs respectively. 
        # The input dimension is the state dimention and the output dimension is the action dimension.
        # This constructor function initializes the network by creating the different layers. 
        # This function now only implements two fully connected layers. Modify this to include LSTM layer(s). 

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, num_outputs)
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=16, num_layers=1, batch_first=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)


    def forward(self, x, hidden=None):
        # The variable x denotes the input to the network. 
        # The hidden variable denotes the hidden state and cell state inputs to the LSTM based network. 
        # The function returns the q value and the output hidden variable information (new cell state and new hidden state) for the given input. 
        # This function now only uses the fully connected layers. Modify this to use the LSTM layer(s).          

        out, hidden = self.lstm(x, hidden)
        out = F.relu(self.fc1(out))
        qvalue = self.fc2(out)

        return qvalue, hidden


    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        # The online_net is the variable that represents the first (current) Q network.
        # The target_net is the variable that represents the second (target) Q network.
        # The optimizer is Adam. 
        # Batch represents a mini-batch of memory. Note that the minibatch also includes the rnn state (hidden state) for the DRQN. 

        # This function takes in a mini-batch of memory, calculates the loss and trains the online network. Target network is not trained using back prop. 
        # The loss function is the mean squared TD error that takes the difference between the current q and the target q. 
        # Return the value of loss for logging purposes (optional).

        # Implement this function. Currently, temporary values to ensure that the program compiles. 

        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.stack(batch.action)
        rewards = torch.stack(batch.reward)
        masks = torch.stack(batch.mask)
        rnn_state = torch.stack(batch.rnn_state).detach()

        h_ = rnn_state[:, :, 0, :]
        c_ = rnn_state[:, :, 1, :]

        h0 = h_.permute(1, 0, 2).reshape((4, 32, 1, 16))[0].permute(1, 0, 2)
        c0 = c_.permute(1, 0, 2).reshape((4, 32, 1, 16))[0].permute(1, 0, 2)
        hidden0 = (h0, c0)

        h1 = h_.permute(1, 0, 2).reshape((4, 32, 1, 16))[1].permute(1, 0, 2)
        c1 = c_.permute(1, 0, 2).reshape((4, 32, 1, 16))[1].permute(1, 0, 2)
        hidden1 = (h1, c1)

        pred, _ = online_net(states, hidden0)
        next_pred, _ = target_net(next_states, hidden1)
        
        actions = actions.reshape((32, 4, 1)).long()
        pred = torch.gather(pred, 2, actions).reshape((32, 4))

        target = rewards + masks * gamma * next_pred.max(2)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    
    def get_action(self, state, hidden):
        # state represents the state variable. 
        # hidden represents the hidden state and cell state for the LSTM.
        # This function obtains the action from the DRQN. The q value needs to be obtained from the forward function and then a max needs to be computed to obtain the action from the Q values. 
        # Implement this function. 
        # Template code just returning a random action.

        state = state.reshape(1, 1, self.num_inputs)
        qvalue, hidden = self.forward(state, hidden)
        _, action = torch.max(qvalue, dim=2)
        action = action.cpu().numpy()[0][0]

        return action, hidden

