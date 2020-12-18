
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from vae_base import *
from scipy.stats import norm, gaussian_kde
from tqdm import tqdm

def calc_mi(vae, x_validation, device='cpu', S=1):
    '''Approximate the mutual information between x and z
    I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
    Modified from the implementation by Author of the paper "LAGGING INFERENCE NETWORKS 
    AND POSTERIOR COLLAPSE IN VARIATIONAL AUTOENCODERS"
    see https://github.com/jxhe/vae-lagging-encoder/blob/master/modules/encoders/encoder.py
    
    This function will calculate the mutual information during the training with LIN trick,
    as a criterion wehther we should stop the agressive training. 
    
    Parameters:
    -----------
    vae: A vae instance, with .infer() method
    
    x_validation: Validation X data set
    
    Returns: Float
    '''
    N_batch = x_validation.shape[0]
    
    #infer zs with encoder 
    # 2D Tensor, shape [N_batch, z_dim]
    mean, std = vae.infer(x_validation)
    assert std.shape == (N_batch, vae.z_dim)
    assert mean.shape == (N_batch, vae.z_dim)
    
    ## Term 1: calculate Negative Entropy, E_{q(z|x)}log(q(z|x))
    # E_{q(z|x)}log(q(z|x)) = -0.5* z_dim *log(2*\pi) - 0.5*(1+log(std**2)).sum(-1)
    # 1D Tensor, shape [N_batch]
    neg_entropy = (-0.5 * vae.z_dim * math.log(2. * math.pi)- 0.5 * (1 + torch.log(std**2)).sum(-1))
    
    
    ## Term 2: calculate E_{q(z|x)}log(q(z))
    #sample zs with the parameters
    if device == 'cuda': z_samples = torch.normal(0,1,size=(S, N_batch, vae.z_dim)).cuda() * std + mean
    if device == 'cpu': z_samples = torch.normal(0,1,size=(S, N_batch, vae.z_dim)) * std + mean
    assert z_samples.shape == (S, N_batch, vae.z_dim)
    
    #evaluate sampled z's under variational distribution
    # 2D Tensor, shape [S, N_batch]
    norm1 = torch.distributions.Normal(mean, std)
    log_qz= torch.sum(norm1.log_prob(z_samples), axis=-1)
    
    return (torch.mean(neg_entropy) - torch.mean(log_qz)).item()

def train_ELBO_VAE_batched(x_train, 
                   x_var = 0.01,
                   z_dim = 1,
                   width = 50,
                   hidden_layers = 1, 
                   learning_rate = 0.01,
                   S = 10,
                   n_epochs = 5000, 
                   report_iter = 50, 
                   batch_size = 256,
                   device = 'cpu'):
    x_dim = x_train.shape[1]
    if device == 'cuda': x_trainT = torch.tensor(x_train).float().cuda()
    if device == 'cpu': x_trainT = torch.tensor(x_train).float()
    batch_num = int(x_train.shape[0]/batch_size)
    vae_instance = VAE(x_dim, z_dim, x_var, hidden_layers, width, hidden_layers, width)
    parameters = list(vae_instance.encoder.parameters())+list(vae_instance.decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    counter=0
    for epoch in tqdm(range(n_epochs)):
        x_trainT=x_trainT[torch.randperm(x_trainT.size()[0])]
        for i in range(batch_num):
            x_batch = x_trainT[i*batch_size:(i+1)*batch_size,:]
            optimizer.zero_grad()
            loss = vae_instance.make_elbo_objective(x_trainT, S)
            loss.backward()
            optimizer.step()
            counter=counter+1
            if counter % report_iter == 0:
                vae_instance.objective_trace.append(loss.item())
                vae_instance.llkhd_KL_trace.append(vae_instance.llkhd_KL())
    return vae_instance

def train_ELBO_LIN_VAE_batched(x_train,
                   x_val,
                   x_var = 0.01,
                   z_dim = 1,
                   width = 50,
                   hidden_layers = 1, 
                   learning_rate = 0.01,
                   S = 10,
                   n_epochs = 5000, 
                   report_iter = 50, 
                   batch_size = 256,
                   device = 'cpu'):
    x_dim = x_train.shape[1]
    if device == 'cuda': 
        x_trainT = torch.tensor(x_train).float().cuda()
        x_valT = torch.tensor(x_val).float().cuda()
    if device == 'cpu': 
        x_trainT = torch.tensor(x_train).float()
        x_valT = torch.tensor(x_val).float()
    batch_num = int(x_train.shape[0]/batch_size)
    vae_instance = VAE(x_dim, z_dim, x_var, hidden_layers, width, hidden_layers, width)
    enc_optimizer = torch.optim.Adam(vae_instance.encoder.parameters(), lr=learning_rate)
    dec_optimizer = torch.optim.Adam(vae_instance.decoder.parameters(), lr=learning_rate)
    counter = 0
    pre_mi = 0
    best_mi = 0
    mi_not_improved = 0
    aggressive_flag = True
    for epoch in tqdm(range(n_epochs)):
        x_trainT=x_trainT[torch.randperm(x_trainT.size()[0])]
        for i in range(batch_num):
            x_batch = x_trainT[i*batch_size:(i+1)*batch_size,:]
            sub_iter=0
            while aggressive_flag and sub_iter < 100:
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                loss = vae_instance.make_elbo_objective(x_batch, S)
                loss.backward()
                enc_optimizer.step()
                sub_iter += 1
                
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            loss = vae_instance.make_elbo_objective(x_batch, S)
            loss.backward()
            if not aggressive_flag:
                enc_optimizer.step()
            dec_optimizer.step()
            counter=counter+1
            if counter % report_iter == 0:
                vae_instance.objective_trace.append(loss.item())
                vae_instance.llkhd_KL_trace.append(vae_instance.llkhd_KL())
            if aggressive_flag and counter % batch_num == 0:
                cur_mi = calc_mi(vae_instance, x_valT, device=device)
                if cur_mi - best_mi < 0:
                    mi_not_improved += 1
                    if mi_not_improved == 5:
                        print("At iteration {} aggresive_flag is set to False".format(counter))
                        aggressive_flag = False
                else:
                    best_mi = cur_mi
                pre_mi = cur_mi

    return vae_instance

def train_TVO_VAE_batched(x_train, 
                  x_var = 0.01,
                  z_dim = 1,
                  width = 50,
                  hidden_layers = 1, 
                  learning_rate = 0.01, 
                  partition = torch.tensor([0,0.25,0.5,0.75,1.]),
                  S = 10,
                  n_epochs = 5000, 
                  report_iter = 50, 
                  batch_size = 256,
                  device = 'cpu'):
    x_dim = x_train.shape[1]
    if device == 'cuda': 
        x_trainT = torch.tensor(x_train).float().cuda()
        partition = partition.cuda()
    if device == 'cpu': x_trainT = torch.tensor(x_train).float()
    batch_num = int(x_train.shape[0]/batch_size)
    vae_instance = VAE(x_dim, z_dim, x_var, hidden_layers, width, hidden_layers, width)
    parameters = list(vae_instance.encoder.parameters())+list(vae_instance.decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    counter=0
    for epoch in tqdm(range(n_epochs)):
        x_trainT=x_trainT[torch.randperm(x_trainT.size()[0])]
        for i in range(batch_num):
            x_batch = x_trainT[i*batch_size:(i+1)*batch_size,:]
            optimizer.zero_grad()
            loss = vae_instance.make_tvo_objective(x_batch, S, partition)
            loss.backward()
            optimizer.step()
            counter=counter+1
            if counter % report_iter == 0:
                vae_instance.objective_trace.append(loss.item())
                vae_instance.llkhd_KL_trace.append(vae_instance.llkhd_KL())
    return vae_instance

def train_TVO_LIN_VAE_batched(x_train,
                   x_val,
                   x_var = 0.01,
                   z_dim = 1,
                   width = 50,
                   hidden_layers = 1, 
                   learning_rate = 0.01,
                   partition = torch.tensor([0,0.25,0.5,0.75,1.]),
                   S = 10,
                   n_epochs = 5000, 
                   report_iter = 50, 
                   batch_size = 256,
                   device = 'cpu'):
    x_dim = x_train.shape[1]
    if device == 'cuda': 
        x_trainT = torch.tensor(x_train).float().cuda()
        x_valT = torch.tensor(x_val).float().cuda()
    if device == 'cpu': 
        x_trainT = torch.tensor(x_train).float()
        x_valT = torch.tensor(x_val).float()
    batch_num = int(x_train.shape[0]/batch_size)
    vae_instance = VAE(x_dim, z_dim, x_var, hidden_layers, width, hidden_layers, width)
    enc_optimizer = torch.optim.Adam(vae_instance.encoder.parameters(), lr=learning_rate)
    dec_optimizer = torch.optim.Adam(vae_instance.decoder.parameters(), lr=learning_rate)
    counter = 0
    pre_mi = 0
    best_mi = 0
    mi_not_improved = 0
    aggressive_flag = True
    for epoch in tqdm(range(n_epochs)):
        x_trainT=x_trainT[torch.randperm(x_trainT.size()[0])]
        for i in range(batch_num):
            x_batch = x_trainT[i*batch_size:(i+1)*batch_size,:]
            sub_iter=0
            while aggressive_flag and sub_iter < 100:
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                loss = vae_instance.make_tvo_objective(x_batch, S, partition)
                loss.backward()
                enc_optimizer.step()
                sub_iter += 1
                
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            loss = vae_instance.make_tvo_objective(x_batch, S, partition)
            loss.backward()
            if not aggressive_flag:
                enc_optimizer.step()
            dec_optimizer.step()
            counter=counter+1
            if counter % report_iter == 0:
                vae_instance.objective_trace.append(loss.item())
                vae_instance.llkhd_KL_trace.append(vae_instance.llkhd_KL())
            if aggressive_flag and counter % batch_num == 0:
                cur_mi = calc_mi(vae_instance, x_valT, device=device)
                if cur_mi - best_mi < 0:
                    mi_not_improved += 1
                    if mi_not_improved == 5:
                        print("At iteration {} aggresive_flag is set to False".format(counter))
                        aggressive_flag = False
                else:
                    best_mi = cur_mi
                pre_mi = cur_mi

    return vae_instance

def random_start_ELBO_VAE(x_train, 
                  x_var = 0.01,
                  z_dim = 1,
                  width = 50,
                  hidden_layers = 1, 
                  learning_rate = 0.01,
                  S = 10,
                  n_epochs = 5000, 
                  report_iter = 50, 
                  batch_size = 256,
                  restart = 5):
    best_vae = train_ELBO_VAE_batched(x_train, 
                  x_var = x_var,
                  z_dim = z_dim,
                  width = width,
                  hidden_layers = hidden_layers, 
                  learning_rate = learning_rate,
                  S = S,
                  n_epochs = n_epochs, 
                  report_iter = report_iter, 
                  batch_size = batch_size)
    loss_min = best_vae.objective_trace[-1]
    for i in range(1, restart):
        cur_vae = train_ELBO_VAE_batched(x_train, 
                  x_var = x_var,
                  z_dim = z_dim,
                  width = width,
                  hidden_layers = hidden_layers, 
                  learning_rate = learning_rate, 
                  S = S,
                  n_epochs = n_epochs, 
                  report_iter = report_iter, 
                  batch_size = batch_size)
        cur_loss = cur_vae.objective_trace[-1]
        if cur_loss < loss_min:
            loss_min = cur_loss
            best_vae = cur_vae
    return best_vae

def random_start_TVO_VAE(x_train, 
                  x_var = 0.01,
                  z_dim = 1,
                  width = 50,
                  hidden_layers = 1, 
                  learning_rate = 0.01, 
                  partition = torch.tensor([0,0.25,0.5,0.75,1.]),
                  S = 10,
                  n_epochs = 5000, 
                  report_iter = 50, 
                  batch_size = 256,
                  restart = 5):
    best_vae = train_TVO_VAE_batched(x_train, 
                  x_var = x_var,
                  z_dim = z_dim,
                  width = width,
                  hidden_layers = hidden_layers, 
                  learning_rate = learning_rate, 
                  partition = partition,
                  S = S,
                  n_epochs = n_epochs, 
                  report_iter = report_iter, 
                  batch_size = batch_size)
    loss_min = best_vae.objective_trace[-1]
    for i in range(1, restart):
        cur_vae = train_TVO_VAE_batched(x_train, 
                  x_var = x_var,
                  z_dim = z_dim,
                  width = width,
                  hidden_layers = hidden_layers, 
                  learning_rate = learning_rate, 
                  partition = partition,
                  S = S,
                  n_epochs = n_epochs, 
                  report_iter = report_iter, 
                  batch_size = batch_size)
        cur_loss = cur_vae.objective_trace[-1]
        if cur_loss < loss_min:
            loss_min = cur_loss
            best_vae = cur_vae
    return best_vae

def visualize_VAE(vae_instance,
                  x_train, 
                  mode,
                  x_var=0.01,
                  number_samples=300,
                  figsize=(10,10)):
    x_hat = vae_instance.generate(number_samples).detach().cpu()
    x_hat = x_hat + np.random.normal(0, x_var**0.5, size=x_hat.shape)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(x_train[:,0], x_train[:,1], color='black',s=20, label="Training Data")
    ax.scatter(x_hat[:, 0], x_hat[:,1], color='blue', s=20, alpha=0.5, label="Generative Data")
    ax.set_title('VAE Generative data with {}'.format(mode), fontsize=20)
    plt.show()

def visualize_pdf(data, lim=1, c_map="magma_r"):
    kde = gaussian_kde(data.T)

    # evaluate on a regular grid
    xgrid = np.linspace(-lim, lim, 60)
    ygrid = np.linspace(-lim, lim, 60)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    plt.figure(figsize=(5,4))
    plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[-lim, lim, -lim, lim],
               cmap=c_map)
    plt.xlabel('x1')
    plt.ylabel('x2')
    cb = plt.colorbar()
    cb.set_label("density")

def compare_VAE(elbo_vae, tvo_vae, x_train, number_samples=2000, x_var=0.01, lim=2, c_map="magma_r"):

    kde = gaussian_kde(x_train.T)
    
    x_hat_elbo = elbo_vae.generate(number_samples).detach().cpu()
    x_hat_elbo = x_hat_elbo + np.random.normal(0, x_var**0.5, size=x_hat_elbo.shape)
    kde_elbo = gaussian_kde(x_hat_elbo.T)
    
    x_hat_tvo = tvo_vae.generate(number_samples).detach().cpu()
    x_hat_tvo = x_hat_tvo + np.random.normal(0, x_var**0.5, size=x_hat_tvo.shape)
    kde_tvo = gaussian_kde(x_hat_tvo.T)
    
    # evaluate on a regular grid
    xgrid = np.linspace(-lim, lim, 60)
    ygrid = np.linspace(-lim, lim, 60)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    Z_elbo = kde_elbo(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    Z_tvo = kde_tvo(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    im=ax[0].imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[-lim, lim, -lim, lim],
               cmap=c_map)
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_title('data')
    fig.colorbar(im, ax=ax[0])
    
    im=ax[1].imshow(Z_elbo.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[-lim, lim, -lim, lim],
               cmap=c_map)
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    ax[1].set_title('elbo')
    fig.colorbar(im, ax=ax[1])
    
    im=ax[2].imshow(Z_tvo.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[-lim, lim, -lim, lim],
               cmap=c_map)
    ax[2].set_xlabel('x1')
    ax[2].set_ylabel('x2')
    ax[2].set_title('tvo')
    fig.colorbar(im, ax=ax[2])
    plt.show()
