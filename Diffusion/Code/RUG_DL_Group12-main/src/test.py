import torch
from piqa import SSIM


def test_diffusion(fold, model, dataloader, logger, device):
    metric_fn = SSIM()
    model.eval()
    total_loss = 0
    total_ssim = 0
    for i, x0 in enumerate(dataloader):
        t = torch.randint(0, int(model.forward_process.T/50), (1,)).to(device)
        t = ((t[0]+1) * 50) - 1
        t = t.repeat(x0.shape[0])

        x0 = x0.to(device)
        x0_noise, gt_noise = model.forward_process(x0, t)

        x0_noise = x0_noise.to(device)
        gt_noise = gt_noise.to(device)
        noise_pred = model(x0_noise, t)

        loss = model.get_loss(noise_pred, gt_noise, t)
        total_loss += loss.item()

        t = t.to("cpu")
        alphas = model.forward_process.alpha[t].to("cpu")
        pre_scale = 1 / torch.sqrt(alphas)

        pre_scale = pre_scale.to(device)
        temp = [8 * a**2 for a in range(int(x0.shape[0]))]
        idx = int((t[0].item() + 1) / 50) - 1
        pre_scale = ((pre_scale - 1) * torch.Tensor([temp[idx] for _ in range(int(x0.shape[0]))]).to(device)) + 1
        pred = pre_scale.view(-1, 1, 1, 1) * (x0_noise - noise_pred)

        pred = torch.nn.functional.normalize(pred, dim=1, p=2)
        pred = (pred + 1) / 2
        x0 = (x0 + 1) / 2
        pred = pred.to("cpu")
        x0 = x0.to("cpu")
        ssim_loss = metric_fn(pred, x0)
        total_ssim += ssim_loss.item()

        print(f"Fold: {fold}, t: {t[0].item()+1}, Loss: {loss.item()}, SSIM: {ssim_loss.item()}")
        logger.log_testing_loss(fold, i, t[0].item()+1, loss.item(), ssim_loss.item())

    total_loss /= len(dataloader)
    total_ssim /= len(dataloader)
    print(f"Testing Loss: {total_loss}")
    print(f"SSIM: {total_ssim}")
