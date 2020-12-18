import torch
import numpy as np

from torch import nn

class MultilayerPerceptron(nn.Module):
    def __init__(self, dims, non_linearity):
        """
        Args:
            dims: list of ints
            non_linearity: differentiable function

        Returns: nn.Module which represents an MLP with architecture

            x -> Linear(dims[0], dims[1]) -> non_linearity ->
            ...
            Linear(dims[-3], dims[-2]) -> non_linearity ->
            Linear(dims[-2], dims[-1]) -> y
            last layer is linear layer"""

        super(MultilayerPerceptron, self).__init__()
        self.dims = dims
        self.non_linearity = non_linearity
        self.linear_modules = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.linear_modules.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        """
        Args:
            x: input data into this NN module, Tensor
        
        Returns: Feedforward Output
        """
        temp = x
        for linear_module in self.linear_modules[:-1]:
            temp = self.non_linearity(linear_module(temp))
        return self.linear_modules[-1](temp)
    
def init_mlp(in_dim, out_dim, hidden_dim, num_hid_layers=1, non_linearity=nn.ReLU()):
    """Initializes a MultilayerPerceptron.

    Args:
        in_dim: int, intput dimension
        out_dim: int, output dimension
        hidden_dim: int, hidden layer width
        num_hid_layers: int, hidden layer number
        non_linearity: differentiable function

    Returns: a MultilayerPerceptron with the architecture

        x -> Linear(in_dim, hidden_dim) -> non_linearity ->
        ...
        Linear(hidden_dim, hidden_dim) -> non_linearity ->
        Linear(hidden_dim, out_dim) -> y

        where num_layers = 0 corresponds to

        x -> Linear(in_dim, out_dim) -> y"""
    dims = [in_dim] + [hidden_dim for _ in range(num_hid_layers)] + [out_dim]
    return MultilayerPerceptron(dims, non_linearity)

def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                                 exp(values[i_1, ..., i_N])
            log( ------------------------------------------------------------ )
                    sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator

def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))

def get_thermo_loss_from_log_weight_log_p_log_q(log_weight, log_p, log_q, partition, num_particles=1,
                                                integration='left'):
    """Args:
        log_weight: tensor of shape [batch_size, num_particles]
        log_p: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
        partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
            see https://en.wikipedia.org/wiki/Partition_of_an_interval
        num_particles: int
        integration: left, right or trapz

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    heated_log_weight = log_weight.unsqueeze(-1) * partition
    heated_normalized_weight = exponentiate_and_normalize(
        heated_log_weight, dim=1)
    thermo_logp = partition * log_p.unsqueeze(-1) + \
        (1 - partition) * log_q.unsqueeze(-1)

    wf = heated_normalized_weight * log_weight.unsqueeze(-1)
    w_detached = heated_normalized_weight.detach()
    wf_detached = wf.detach()
    if num_particles == 1:
        correction = 1
    else:
        correction = num_particles / (num_particles - 1)

    cov = correction * torch.sum(
        w_detached * (log_weight.unsqueeze(-1) - torch.sum(wf, dim=1, keepdim=True)).detach() *
        (thermo_logp - torch.sum(thermo_logp * w_detached, dim=1, keepdim=True)),
        dim=1)

    multiplier = torch.zeros_like(partition)
    if integration == 'trapz':
        multiplier[0] = 0.5 * (partition[1] - partition[0])
        multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
        multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
    elif integration == 'left':
        multiplier[:-1] = partition[1:] - partition[:-1]
    elif integration == 'right':
        multiplier[1:] = partition[1:] - partition[:-1]

    loss = -torch.mean(torch.sum(
        multiplier * (cov + torch.sum(
            w_detached * log_weight.unsqueeze(-1), dim=1)),
        dim=1))

    return loss


class VAE:
    def __init__(self, x_dim, z_dim, x_var, 
                encoder_hid_layer_num, encoder_hid_width,
                decoder_hid_layer_num, decoder_hid_width,
                encoder_nonlinearity = nn.ReLU(),
                decoder_nonlinearity = nn.ReLU(),
                device = 'cpu'):
        """VAE framework constructor
        Args:
            x_dim: Dimension of observations, int
            z_dim: Dimension of latent space, int
            x_var: varaince of generative model
            en/decoder_hid_layer_num: hidden layer number of en/decoder net, int
            en/decoder_hid_width: dim of en/decoder hidden layer, int
            en/decoder_nonlinearity: torch differentiable function"""
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.x_var = x_var

        self.encoder = init_mlp(self.x_dim, self.z_dim*2, 
                                encoder_hid_width, encoder_hid_layer_num,
                                encoder_nonlinearity)
        self.decoder = init_mlp(self.z_dim, self.x_dim,
                                decoder_hid_width, decoder_hid_layer_num,
                                decoder_nonlinearity)
        if device == 'cuda': 
            self.encoder=self.encoder.cuda()
            self.decoder=self.decoder.cuda()
        self.objective_trace = []
        self.llkhd_KL_trace = []
    
    def generate(self, N=300):
        """On CPU, use the generative model to generate x given zs sampled from the prior
        Returns: Synthetic observation points Pytorch Tensor with shape (N, x_dim)"""
        z_samples = torch.normal(0, 1, size=(N,self.z_dim))
        decodercpu = self.decoder.cpu()
        return decodercpu.forward(z_samples)
    
    def infer(self, x):
        """Use the encoder to infer mean and sigma of q(z|x)
        Args:
            x: observation points, torch tensor, shape (N, x_dim)
        Returns: 
            mean: mean of q(z|x), torch tensor, shape (N, z_dim)
            std: std of q(z|x), positive, torch tensor, shape (N, z_dim)"""
        z_params = self.encoder.forward(x)
        mean = z_params[:,:self.z_dim]
        parameterized_std = z_params[:,self.z_dim:]
        std = torch.exp(parameterized_std) # To keep the std positive
        return mean, std

    def llkhd_KL(self, x_train, S, device = 'cpu'):
        '''Output Log Likelihood and KL divergence, fro trace plot'''
        assert len(x_train.shape) == 2
        assert x_train.shape[1] == self.x_dim
        if S is not None:
            self.S = S
        N = x_train.shape[0] #sample numbers
        
        #infer zs with encoder 
        mean, std = self.infer(x_train)
        assert std.shape == (N, self.z_dim)
        assert mean.shape == (N, self.z_dim)
        
        #sample zs with the parameters
        if device == 'cuda': z_samples = torch.normal(0,1,size=(self.S, N, self.z_dim)).cuda() * std + mean
        if device == 'cpu': z_samples = torch.normal(0,1,size=(self.S, N, self.z_dim)) * std + mean
        assert z_samples.shape == (self.S, N, self.z_dim)
        
        #predict xs
        x = self.decoder.forward(z_samples)
        assert x.shape == (self.S, N, self.x_dim)
        
        #evaluate log_likelihood p(y_n)
        norm1 = torch.distributions.Normal(x, self.x_var**0.5)
        log_likelihood = torch.sum(norm1.log_prob(x_train), axis=-1)
        assert log_likelihood.shape == (self.S, N)
        
        #evaluate sampled zs under prior 
        norm2 = torch.distributions.Normal(0.0, 1.0)
        log_pz = torch.sum(norm2.log_prob(z_samples), axis=-1)
        assert log_pz.shape == (self.S, N)
        
        #evaluate sampled z's under variational distribution
        norm3 = torch.distributions.Normal(mean, std)
        log_qz_given_x = torch.sum(norm3.log_prob(z_samples), axis=-1)

        return [torch.mean(log_likelihood).item(), torch.mean(log_qz_given_x - log_pz).item()]

    
    def make_elbo_objective(self, x_train, S, device = 'cpu'):
        '''Make ELBO objective function'''
        assert len(x_train.shape) == 2
        assert x_train.shape[1] == self.x_dim
        if S is not None:
            self.S = S
        N = x_train.shape[0] #sample numbers
        
        #infer zs with encoder 
        mean, std = self.infer(x_train)
        assert std.shape == (N, self.z_dim)
        assert mean.shape == (N, self.z_dim)
        
        #sample zs with the parameters
        if device == 'cuda': z_samples = torch.normal(0,1,size=(self.S, N, self.z_dim)).cuda() * std + mean
        if device == 'cpu': z_samples = torch.normal(0,1,size=(self.S, N, self.z_dim)) * std + mean
        assert z_samples.shape == (self.S, N, self.z_dim)
        
        #predict xs
        x = self.decoder.forward(z_samples)
        assert x.shape == (self.S, N, self.x_dim)
        
        #evaluate log_likelihood p(y_n)
        norm1 = torch.distributions.Normal(x, self.x_var**0.5)
        log_likelihood = torch.sum(norm1.log_prob(x_train), axis=-1)
        assert log_likelihood.shape == (self.S, N)
        
        #evaluate sampled zs under prior 
        norm2 = torch.distributions.Normal(0.0, 1.0)
        log_pz = torch.sum(norm2.log_prob(z_samples), axis=-1)
        assert log_pz.shape == (self.S, N)
        
        #evaluate sampled z's under variational distribution
        norm3 = torch.distributions.Normal(mean, std)
        log_qz_given_x = torch.sum(norm3.log_prob(z_samples), axis=-1)
        
        elbo = torch.mean(log_likelihood - log_qz_given_x + log_pz)
        
        return -elbo
    
    def make_tvo_objective(self, x_train, S, partition, num_particles=10, integration='left', device='cpu'):
        '''Make TVO objective function'''
        assert len(x_train.shape) == 2
        assert x_train.shape[1] == self.x_dim
        if S is not None:
            self.S = S
        N = x_train.shape[0] #sample numbers
        
        #infer zs with encoder 
        mean, std = self.infer(x_train)
        assert std.shape == (N, self.z_dim)
        assert mean.shape == (N, self.z_dim)
        
        #sample zs with the parameters
        if device == 'cuda': z_samples = torch.normal(0,1,size=(self.S, N, self.z_dim)).cuda() * std + mean
        if device == 'cpu': z_samples = torch.normal(0,1,size=(self.S, N, self.z_dim)) * std + mean
        assert z_samples.shape == (self.S, N, self.z_dim)
        
        #predict xs
        x = self.decoder.forward(z_samples)
        assert x.shape == (self.S, N, self.x_dim)
        
        #evaluate log_likelihood log p_w(y_n)
        norm1 = torch.distributions.Normal(x, self.x_var**0.5)
        log_likelihood = torch.sum(norm1.log_prob(x_train), axis=-1)
        assert log_likelihood.shape == (self.S, N)
        
        #evaluate sampled zs under prior log q_v(z_n)
        norm2 = torch.distributions.Normal(0.0, 1.0)
        log_pz = torch.sum(norm2.log_prob(z_samples), axis=-1)
        assert log_pz.shape == (self.S, N)
        
        #evaluate sampled z's under variational distribution, p_w( zn | yn )
        norm3 = torch.distributions.Normal(mean, std)
        log_qz_given_x = torch.sum(norm3.log_prob(z_samples), axis=-1)
        
        log_weight = log_likelihood - log_qz_given_x + log_pz
        log_p = log_likelihood
        log_q = log_qz_given_x - log_pz
        
        return get_thermo_loss_from_log_weight_log_p_log_q(log_weight, 
                                                           log_p, 
                                                           log_q, 
                                                           partition, 
                                                           num_particles,
                                                           integration)