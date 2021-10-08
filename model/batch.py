import torch


class Batch:
    def __init__(self, imgs, xs, ys):
        self.size = len(xs)
        self.imgs = torch.FloatTensor(imgs)
        self.xs = torch.LongTensor(xs)
        self.ys = torch.FloatTensor(ys)

    def to_device(self, device):
        self.imgs = self.imgs.to(device)
        self.xs = self.xs.to(device)
        self.ys = self.ys.to(device)


def generate_batch(batch):
    """
        This function is used as a 'collate_fn' of torch DataLoader
        Importance: do the padding here if needed
    """
    imgs = []
    xs = []
    ys = []
    for b in batch:
        imgs.append(b.get('img'))
        xs.append(b.get('x'))
        ys.append(b.get('y'))

    return Batch(imgs, xs, ys)
