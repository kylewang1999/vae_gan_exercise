import os, pickle, torch
from os.path import join, dirname, exists
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from scipy.io import loadmat
from datetime import datetime

''' SVHN Dataset Helper'''

def load_svhn(train=True):
    ''' Load image and labels. Note: images are channels-last, i.e. (N, 32, 32, 3) '''
    dataset = loadmat('data/train.mat') if train else loadmat('data/test.mat')
    X, y = np.moveaxis(dataset['X'], -1, 0), dataset['y']
    return (X, y)

def one_hot_encoding(labels, num_classes=10):
    if np.min(labels!=0): labels = labels-1
    return np.eye(num_classes, dtype=int)[labels.flatten()]

def normalize_data(X):
    """ Normalizes inputs (on per channel basis of every image) here to have 0 mean and unit variance.
    This will require reshaping to separate the channels and then undoing it while returning

    args:
        X : np.array (N,32,32,3)

    returns:
        normalized X. np.array (N, 32,32,3)
        u: Mean per img per channel (N,3)
        ds: Std per imm per channel (N,3)
    """
    assert X.shape[-1] == 3, f'X should be of shape (N,H,W,3), but got {X.shape}'

    X = X.reshape(X.shape[0], -1, X.shape[-1]) # (N,1024,3)
    u = np.mean(X, axis=1)  # (N,3)
    X = X - np.moveaxis(np.repeat(u[:,:, np.newaxis], 32*32, axis=-1) , 1,-1)

    sd = np.std(X, axis=1)
    X = X / np.moveaxis(np.repeat(sd[:,:, np.newaxis], 32*32, axis=-1) , 1,-1)
    X = X.reshape(X.shape[0], 32, 32, 3)
    return X, u, sd

def split_and_preprocess(dataset, split=60000):
    ''' Shuffle, normalize, 1hot encode, split training set '''
    X, y = dataset
    assert X.shape[0] == y.shape[0], f"Number of images {X.shape[-1]}, disagree with number of labels {len(y)}"
    
    X,_,_ = normalize_data(X)                   # normalize
    y = one_hot_encoding(y)                     # 1hot encode

    p = np.random.permutation(X.shape[0])       # shuffle
    p1, p2 = p[:split], p[split:]

    img_trn, lab_trn = X[p1], y[p1]
    img_val, lab_val = X[p2], y[p2]

    return (img_trn, lab_trn), (img_val, lab_val)



''' Plotting '''

def plot_rgb_img(img_arr, path='foo.png'):
    ''' Plot one input RGB image (32,32,3)'''
    # if  img_arr.shape[0]==3072:
    #     img_arr = unflatten_img(img_arr)
    plt.imshow(img_arr)


def plot_images(imgs, annotations=None, cols=4, path=None, title=None):
    '''
    Input:
        - imgs: numpy array of shape (n, 32,32,3), representing n images to plot
        - cols: number of columns for subplots
        - annotations: A list of annotations containing titles for each img subplot
    Output:
        - Doesn't return. Plots and saves the image to ./imgs folder.
    '''
    if annotations is None:
        annotations = [None] * imgs.shape[0]


    rows = imgs.shape[0] // cols if imgs.shape[0] % cols == 0 else imgs.shape[0] // cols+1
    fig, axs = plt.subplots(rows,cols)
    if title:
        fig.suptitle(title, fontsize=2*cols, fontweight="bold")
    
    for idx, (img, ann) in enumerate(zip(imgs, annotations)):
        if np.min(img) < 0:
            img = np.uint8((img+1)*255/2)     # Convert [0,1] range to [0, 255] range
        else:
            img = np.uint8(img)
        j, i = idx % cols, idx // cols
        ax = axs[j] if rows == 1 else axs[i,j]
        ax.imshow(img, cmap='gray')
        
        ax.set_title(ann)
        ax.axis('off')
        
    fig.set_figheight(3*rows)
    fig.set_figwidth(3*cols)

    plt.subplots_adjust(left=0.05, right=0.95,
                      bottom=0.1, top=0.9,
                      wspace=0.4, hspace=0.3)

    if path is not None: plt.savefig(path)


def savefig(fname, show_figure=True):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def save_training_plot(train_losses, test_losses, title, fname):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    savefig(fname)


def save_scatter_2d(data, title, fname):
    plt.figure()
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1])
    savefig(fname)


def save_distribution_1d(data, distribution, title, fname):
    d = len(distribution)

    plt.figure()
    plt.hist(data, bins=np.arange(d) - 0.5, label='train data', density=True)

    x = np.linspace(-0.5, d - 0.5, 1000)
    y = distribution.repeat(1000 // d)
    plt.plot(x, y, label='learned distribution')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    savefig(fname)


def save_distribution_2d(true_dist, learned_dist, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(true_dist)
    ax1.set_title('True Distribution')
    ax1.axis('off')
    ax2.imshow(learned_dist)
    ax2.set_title('Learned Distribution')
    ax2.axis('off')
    savefig(fname)


def show_samples(samples, fname=None, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    if fname is not None:
        savefig(fname)
    else:
        plt.show()


def load_pickled_data(fname, include_labels=False):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    train_data, test_data = data['train'], data['test']
    if 'mnist.pkl' in fname or 'shapes.pkl' in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype('uint8')
        test_data = (test_data > 127.5).astype('uint8')
    if 'celeb.pkl' in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data['train_labels'], data['test_labels']
    return train_data, test_data


def quantize(images, n_bits):
    images = np.floor(images / 256. * 2 ** n_bits)
    return images.astype('uint8')

        
''' Save model '''
def get_time():
    return datetime.now().strftime("%m-%d_%H:%M")

def save_model(model, stats_record=None, remark=""):
    dir = './trained_model'
    if not os.path.exists(dir):
        os.makedirs(dir)

    torch.save(model.state_dict(), f'{dir}/{remark}_{get_time()}.pt')
    if stats_record is not None:
        with open(f'{dir}/{remark}_{get_time()}_record.pickle', 'wb') as f:
            pickle.dump(stats_record, f)

def load_model(model, path=""):
    model.load_state_dict(torch.load(path))
    return model
