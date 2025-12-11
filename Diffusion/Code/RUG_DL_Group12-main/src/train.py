import torch


def train_diffusion(fold, model, optimizer, train_loader, val_loader, logger, device, epochs=10):
    model = model.to(device)
    current_smallest_val_loss = float("inf")
    early_stopping = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, x0 in enumerate(train_loader):
            t = torch.randint(0, model.forward_process.T, (x0.shape[0],)).to(device)
            x0 = x0.to(device)
            x0_noise, gt_noise = model.forward_process(x0, t)

            x0_noise = x0_noise.to(device)
            gt_noise = gt_noise.to(device)
            optimizer.zero_grad()
            noise_pred = model(x0_noise, t)
            loss = model.get_loss(noise_pred, gt_noise, t)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        total_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {total_loss}")
        logger.log_training_loss(fold, epoch, total_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, x0 in enumerate(val_loader):
                t = torch.randint(0, model.forward_process.T, (x0.shape[0],)).to(device)
                x0 = x0.to(device)
                x0_noise, gt_noise = model.forward_process(x0, t)

                x0_noise = x0_noise.to(device)
                gt_noise = gt_noise.to(device)
                optimizer.zero_grad()
                noise_pred = model(x0_noise, t)
                loss = model.get_loss(noise_pred, gt_noise, t)
                total_val_loss += loss.item()
            total_val_loss /= len(val_loader)
            print(f"Validation Loss: {total_val_loss}")
            logger.log_validation_loss(fold, epoch, total_val_loss)

        if total_val_loss < current_smallest_val_loss:
            current_smallest_val_loss = total_val_loss
            early_stopping = 0
            logger.save_model(model, fold)
        else:
            print("Validation loss did not decrease.")
            early_stopping += 1
            if early_stopping == 5:
                print("Early stopping.")
                break

    return model
