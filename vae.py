from importlib.abc import ResourceLoader
import os, torch,  torch.nn as nn,  torch.nn.functional as F,  numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, in_channels = 3, latent_dim=128):
        
        super(Encoder, self).__init__()
        self.in_channels, self.latent_dim = in_channels, latent_dim

        self.submodules = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, \
                kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),     # 16*16 feature map
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),    # 8*8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),   # 4*4
            nn.Flatten(),
            nn.Linear(4*4*256, 2*self.latent_dim)   # FC layer for mean and variance
        )
    
    def forward(self, input):
        mean_and_var = self.submodules(input)
        return mean_and_var


# Reference: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, kl_weight=3e-4) -> None:
        super(VAE,self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim    # dimension of encoded embeddings
        self.kl_weight = kl_weight

        self.encoder = Encoder(in_channels, latent_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 128),
            nn.ReLU(),
        )            
            
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def encode(self, input):
        ''' Encode input tensor (B,C,H,W) '''
        mean_and_var = self.encoder(input)
        mean, var = mean_and_var[:,:self.latent_dim], mean_and_var[:,self.latent_dim:]
        return mean, var

    def decode(self, z):
        ''' Decode latent feature z (B, latent_dim)
        Return (B,C,H,W)
        '''
        _z = self.decoder_input(z)
        return self.decoder(_z.reshape(_z.shape[0], -1, 4, 4)) # (B, 128, 4, 4)

    def reparameterize(self, mean, var):
        ''' The reparameterization trick that makes probabilistic decoding differentiable 
        Input
            - mean: (B, latent_dim)
            - var: (B, latent_dim)
        Return
            - (B, latent_dim)
        '''
        std = torch.exp(0.5 * var)  # var is the log of variance
        eps = torch.randn_like(var)
        z = mean + eps * std
        return z
    
    def forward(self, input):
        ''' Return [input, latent feature, mean, log variance] '''
        mean, var = self.encode(input)
        z = self.reparameterize(mean, var)      # (B, 128)
        return input, self.decode(z), mean, var 

    def elbo_loss(self, input, recon, mean, var):
        '''
        Return:
            - Reconstruction loss  (avg. over batch dim, sum over feature dim)
            - KL loss (avg. over batch dim, sum over feature dim)
            - Negative ELBO
        '''
        loss_recon = F.mse_loss(recon, input, reduction='none')
        loss_recon = torch.mean(torch.sum(loss_recon, dim=(1,2,3)))

        loss_kl = torch.mean(
            -0.5 * torch.sum((1 + var - (mean ** 2) - var.exp()), dim=1), dim=0
        )
        loss = loss_recon + self.kl_weight * loss_kl

        return loss, loss_recon, loss_kl

    def sample(self, N):
        ''' Produce N samples from a trained model '''
        z = torch.randn(N, self.latent_dim)
        return self.decode(z)

    def interpolate(self, imgs1, imgs2):
        interpolations = []
        for i in range(imgs1.shape[0]):
            z1 = self.reparameterize(*self.encode(imgs1[i:i+1]))    
            z2 = self.reparameterize(*self.encode(imgs2[i:i+1]))
            z_interp = torch.lerp(z1, z2, torch.linspace(0.1,1,10).reshape(-1,1))
            decoded = self.decode(z_interp)
            interpolations.append(decoded)
        interpolations = torch.cat(interpolations, dim=0)
        return interpolations
        

''' Training '''

def train(model, train_loader, test_loader, config):
    epochs, remark = config['epochs'], config['remark']
    num_samples = len(train_loader)*config['bz']
    print(f'Training model [{remark}] on device [{device}] with [{num_samples}] samples')
    print(f'\tTotal epochs: [{epochs}], total steps: {epochs*len(train_loader)}')
    

    train_record, test_record = np.zeros((epochs * len(train_loader), 3)), np.zeros((epochs, 3))
    optim = torch.optim.Adam(model.parameters(), config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs, eta_min=1e-4)

    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        
        pbar.set_description_str(f"Epoch {epoch+1}")
        for step, imgs in enumerate(train_loader):
            input, recon, mean, var =  model(imgs)
            loss, loss_recon, loss_kl = model.elbo_loss(input, recon, mean, var)

            optim.zero_grad(); loss.backward(); optim.step(); scheduler.step()

            train_record[epoch+step, 0] = loss_recon.item() + loss_kl.item()
            train_record[epoch+step, 1] = loss_recon.item()
            train_record[epoch+step, 2] = loss_kl.item()

            pbar.set_postfix({
                'loss'      : round(train_record[epoch+step, 0], 2),
                'loss recon': round(train_record[epoch+step, 1], 2),
                'loss kl'   : round(train_record[epoch+step, 2], 2)
            })

        # Evaluation on Test set
        loss, loss_recon, loss_kl = test(model, test_loader)
        test_record[epoch, 0] = loss
        test_record[epoch, 1] = loss_recon
        test_record[epoch, 2] = loss_kl
    
    stats_record = (train_record, test_record)
    save_model(model, stats_record, remark)
    return stats_record


def test(model, test_loader):
    model.eval()
    
    loss_per_epoch, loss_recon_epoch, loss_kl_epoch = 0,0,0
    with torch.no_grad():
        for imgs in test_loader:
            input, recon, mean, var =  model(imgs)
            loss, loss_recon, loss_kl = model.elbo_loss(input, recon, mean, var)
            loss_per_epoch      += loss_recon.item() + loss_kl.item()
            loss_recon_epoch    += loss_recon.item()
            loss_kl_epoch       += loss_kl.item()
    loss_per_epoch      /= len(test_loader)
    loss_recon_epoch    /= len(test_loader)
    loss_kl_epoch       /= len(test_loader)
            
    model.train()
    return loss_per_epoch, loss_recon_epoch, loss_kl_epoch



''' Inspection '''
mean, std = np.array([111.608, 113.161, 120.565]), np.array([50.497, 51.258, 50.244])
denormalize = transforms.Normalize(-mean/std, 1/std)

def sample_and_visualize(model, view=True):
    samples = model.sample(100)
    samples = denormalize(samples).permute([0,2,3,1]).detach().numpy()
    if view: plot_images(samples, cols=20, title='Sampled images')
    return samples

def decode_and_visualize(imgs, model, view=True):   # decode 50 images and visualize
    imgs = imgs[:50]
    _, decoded, _, _ = model(imgs)
    imgs = denormalize(imgs).permute([0,2,3,1]).detach().numpy()
    decoded = denormalize(decoded).permute([0,2,3,1]).detach().numpy()
    real_recon_pairs = np.stack([imgs, decoded]).reshape((-1,*imgs.shape[1:]), order='F')

    if view:
        plot_images(real_recon_pairs, cols=20, title='Real/recon pairs')
    return real_recon_pairs

def interpolate_and_visualize(imgs, model, view=True):
    ''' Pick 10 random paris from images, then interpolate each pair; 
    Input
        - imgs: Tensor (B,C,H,W)
        - model: VAE Model
    Output
        - np.array. (B,H,W,3)
    '''
    indices = torch.randint(0, imgs.shape[0], size=(20,))   # randomly pick 20 images 
    interpolations = model.interpolate(imgs[indices[:10]], imgs[indices[10:]])
    interpolations = denormalize(interpolations).permute([0,2,3,1]).detach().numpy()
    if view: plot_images(interpolations,cols=20, title='Interpolations')
    return interpolations


''' Deliverables '''

def plot_vae_training_plot(train_losses, test_losses, title, fname):
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    savefig(fname)

# def vae_save_results(dset_id, fn):
#     assert dset_id in [1, 2]
#     if dset_id == 1:
#         train_data, test_data = load_pickled_data('svhn.pkl')
#     else:
#         train_data, test_data = load_pickled_data('cifar10.pkl')

#     train_losses, test_losses, samples, reconstructions, interpolations = fn(train_data, test_data, dset_id)
#     samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype('float32'), interpolations.astype('float32')
#     print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
#           f'KL Loss: {test_losses[-1, 2]:.4f}')
#     plot_vae_training_plot(train_losses, test_losses, f'VAE Dataset {dset_id} Train Plot',
#                            f'results/VAE_dset{dset_id}_train_plot.png')
#     show_samples(samples, title=f'VAE Dataset {dset_id} Samples',
#                  fname=f'results/VAE_dset{dset_id}_samples.png')
#     show_samples(reconstructions, title=f'VAE Dataset {dset_id} Reconstructions',
#                  fname=f'results/VAE_dset{dset_id}_reconstructions.png')
#     show_samples(interpolations, title=f'VAE Dataset {dset_id} Interpolations',
#                  fname=f'results/VAE_dset{dset_id}_interpolations.png')

def vae_save_results(args, dset_id=1):
    train_losses, test_losses, samples, reconstructions, interpolations = args
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype('float32'), interpolations.astype('float32')
    
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'VAE Dataset {dset_id} Train Plot',
                           f'results/VAE_dset{dset_id}_train_plot.png')
    show_samples(samples, title=f'VAE Dataset {dset_id} Samples',
                 fname=f'results/VAE_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'VAE Dataset {dset_id} Reconstructions',
                 fname=f'results/VAE_dset{dset_id}_reconstructions.png')
    show_samples(interpolations, title=f'VAE Dataset {dset_id} Interpolations',
                 fname=f'results/VAE_dset{dset_id}_interpolations.png')


def problem_vae(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns: 
    Loss(per batch, train) | Loss(per epoch, test) | Samples | Samples(real/recon pair) | Interpolations

    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255}
    """

    """ YOUR CODE HERE """
    return np.zeros([1000,3]), np.zeros([20,3]), np.zeros([100,32,32,3]), \
            np.zeros([100,32,32,3]), np.zeros([100,32,32,3])
