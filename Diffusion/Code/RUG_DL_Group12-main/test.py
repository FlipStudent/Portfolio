import src.dataset as ds
import src.model as md
import src.logger as lg
import src.test as ts

import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms


def test(file="utils/test_fruits.csv"):
    dataset = ds.Fruits(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = lg.Logger(testing=True)

    for fold in range(5):
        print(f"output/models/model_f{fold}.pth")
        model = md.DiffusionUNet(in_size=100, t_range=1000, img_depth=3, device=device).to(device)
        model.load_state_dict(torch.load(f"output/models/model_f{fold}.pth"))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

        ts.test_diffusion(fold, model, dataloader, logger, device)


def generate_reconstructions(file="utils/test_fruits.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = md.DiffusionUNet(in_size=100, t_range=1000, img_depth=3, device=device).to(device)
    model.load_state_dict(torch.load("output/models/model_f0.pth"))
    dataset = ds.Fruits("utils/test_fruits.csv")

    sample = dataset[0]
    sample = sample.unsqueeze(0)
    sample = sample.repeat(20, 1, 1, 1)
    print(sample.shape)
    sample = sample.to(device)

    t = torch.arange(1, 21).to(device)
    t = (t * 50) - 1

    sample_noised, noise = model.forward_process(sample, t)
    sample_noised = sample_noised.to(device)
    pred_noise = model(sample_noised, t)

    t = t.to("cpu")
    alphas = model.forward_process.alpha[t].to("cpu")
    alpha_bars = model.forward_process.alpha_bar[t].to("cpu")
    pre_scale = 1 / torch.sqrt(alphas)
    e_scale = (1 - alphas) / torch.sqrt(1 - alpha_bars)

    pre_scale = pre_scale.to(device)
    e_scale = e_scale.to(device)
    pre_scale = ((pre_scale - 1) * torch.Tensor([8 * a**2 for a in range(20)]).to(device)) + 1
    pred = pre_scale.view(-1, 1, 1, 1) * (sample_noised - pred_noise)

    grid = torch.cat([sample, sample_noised, pred], dim=0)
    print(grid.shape)
    vutils.save_image(grid, "output/combined.png", nrow=20)


def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = md.DiffusionUNet(in_size=100, t_range=1000, img_depth=3, device=device).to(device)
    model.load_state_dict(torch.load("output/models/model_f0.pth"))

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    sample = torch.randint(0, 256, (3, 100, 100)) / 255
    sample = transform(sample)
    print(sample)
    print(torch.max(sample), torch.min(sample))
    vutils.save_image(sample, "output/sample.png")

    grid = torch.Tensor([])

    for i in range(1000, 0, -1):
        t = torch.tensor([i]).to(device)
        print(t)
        sample = sample.unsqueeze(0)
        sample = sample.to(device)
        sample, _ = model.forward_process(sample, i-1)
        model.eval()
        with torch.no_grad():
            # if t > 1:
            #     z = torch.randn(1)
            # else:
            #     z = 0
            e_hat = model(sample, t).to(device)
            e_hat = torch.squeeze(e_hat)
            sample = torch.squeeze(sample)
            sample = (sample - e_hat) * (1 + (i / 1000))
            # sample = torch.squeeze(sample)
            # e_hat = torch.squeeze(e_hat)
            # pre_scale = 1 / math.sqrt(model.forward_process.alpha[int(t.item()) - 1])
            # e_scale = (1 - model.forward_process.alpha[int(t.item()) - 1]) /
            # math.sqrt(1 - model.forward_process.alpha_bar[int(t.item()) - 1])
            # post_sigma = math.sqrt(model.forward_process.beta[int(t.item()) - 1]) * torch.tensor([z])
            # sample = sample.to(device)
            # e_scale = e_scale.to(device)
            # e_hat = e_hat.to(device)
            # post_sigma = post_sigma.to(device)
            # sample = pre_scale * (sample - e_scale * e_hat) + post_sigma
        # if i % 100 == 0 or i == 1:
        #     vutils.save_image(sample, f"output/pred_{i}.png")
        if i % 100 == 0:
            temp = sample.unsqueeze(0)
            temp = temp.to("cpu")
            grid = torch.cat([grid, temp], dim=0)
    vutils.save_image(grid, "output/grid_generated.png", nrow=5)


if __name__ == '__main__':
    generate_reconstructions()
    # generate_reconstructions(file="utils/dummy_fruits.csv")

    generate()

    test()
    # test(file="utils/dummy_fruits.csv")
