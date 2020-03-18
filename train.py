from unet import UNet
from data import Dataset
import torch.utils.data
import torch.nn.functional as f


def loss_function(out, target):
    mask = target > 0

    segment_loss = f.binary_cross_entropy_with_logits(
        out,
        mask.astype(out.dtype)
    )

    height_loss = f.mse_loss(
        out[mask],
        torch.log(target[mask])
    )

    return segment_loss + height_loss


def main(num_epoch, batch_size, device):
    model = UNet().train()
    dataset = Dataset()
    optimizer = torch.optim.Adam(model.parameters())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True)

    for e in range(num_epoch):
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)

            y = model(x)
            loss = loss_function(y, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            print(loss)
            break

    torch.save(model.state_dict(), 'unet.pt')


if __name__ == '__main__':
    main(num_epoch=100, batch_size=1, device=torch.device('cpu'))
