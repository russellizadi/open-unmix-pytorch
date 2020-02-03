import shutil
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

def _sndfile_available():
    try:
        import soundfile
    except ImportError:
        return False

    return True


def _torchaudio_available():
    try:
        import torchaudio
    except ImportError:
        return False

    return True


def get_loading_backend():
    if _torchaudio_available():
        return torchaudio_loader

    if _sndfile_available():
        return soundfile_loader


def get_info_backend():
    if _torchaudio_available():
        return torchaudio_info

    if _sndfile_available():
        return soundfile_info


def soundfile_info(path):
    import soundfile
    info = {}
    sfi = soundfile.info(path)
    info['samplerate'] = sfi.samplerate
    info['samples'] = int(sfi.duration * sfi.samplerate)
    info['duration'] = sfi.duration
    return info


def soundfile_loader(path, start=0, dur=None):
    import soundfile
    # get metadata
    info = soundfile_info(path)
    start = int(start * info['samplerate'])
    # check if dur is none
    if dur:
        # stop in soundfile is calc in samples, not seconds
        stop = start + int(dur * info['samplerate'])
    else:
        # set to None for reading complete file
        stop = dur

    audio, _ = soundfile.read(
        path,
        always_2d=True,
        start=start,
        stop=stop
    )
    return torch.FloatTensor(audio.T)


def torchaudio_info(path):
    import torchaudio
    # get length of file in samples
    info = {}
    si, _ = torchaudio.info(str(path))
    info['samplerate'] = si.rate
    info['samples'] = si.length // si.channels
    info['duration'] = info['samples'] / si.rate
    return info


def torchaudio_loader(path, start=0, dur=None):
    import torchaudio
    info = torchaudio_info(path)
    # loads the full track duration
    if dur is None:
        sig, rate = torchaudio.load(path)
        return sig
        # otherwise loads a random excerpt
    else:
        num_frames = int(dur * info['samplerate'])
        offset = int(start * info['samplerate'])
        sig, rate = torchaudio.load(
            path, num_frames=num_frames, offset=offset
        )
        return sig


def load_info(path):
    loader = get_info_backend()
    return loader(path)


def load_audio(path, start=0, dur=None):
    loader = get_loading_backend()
    return loader(path, start=start, dur=dur)


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def save_checkpoint(
    state, is_best, path, target
):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, target + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

def normalize(x):
    return x/x.max()

def set_args():
    args = lambda: None
    args.fig_ratio = 1 # in latex
    args.clb_pad = .01
    args.clb_shrink = .85
    args.clb_fraction = .1 # change this
    args.font_size = 10
    args.samplerate = 44100
    args.path_fig = 'tmp/fig.pdf'
    args.fig_pad = .01
    args.fig_shrink = .5
    args.fig_fraction = 0.05
    return args

def plot_spc(x, args):
    len_x, dim_x = x.shape

    cfg_fig = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": args.font_size,
        "font.size": args.font_size,
        "font.serif": 'Times',
        "legend.fontsize": args.font_size,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size
    }
    mpl.rcParams.update(cfg_fig)

    len_fig = 345/72.27 # textwidth pt to inch
    len_fig *= (args.fig_ratio) # fraction in latex
    fig, axes = plt.subplots(1, 1, figsize=(len_fig*1.3, len_fig))

    im = axes.imshow(x, origin='lower', cmap=cm.Blues)

    cb = fig.colorbar(im, pad=args.fig_pad, shrink=args.fig_shrink, fraction=args.fig_fraction)
    cb.outline.set_visible(False)
    axb = cb.ax

    axb.set_xlabel("{0:.0f}".format(np.min(x)))
    axt = axb.twiny()

    axt.set_xlabel("{0:.0f}".format(np.max(x)))
    axt.xaxis.set_ticks([])
    axt.yaxis.set_ticks([])

    axt.spines['top'].set_visible(False)
    axt.spines['bottom'].set_visible(False)
    axt.spines['right'].set_visible(False)
    axt.spines['left'].set_visible(False)

    axes.tick_params(axis='both', which='both', length=0, direction='in')
    
    if hasattr(args, 'path_fig'):
        fig.savefig(args.path_fig, bbox_inches='tight')
    plt.close()
