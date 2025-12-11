import os
import torch


class Logger:
    def __init__(self, testing=False, model=""):
        self.logs_path = "output/logs/"
        self.models_path = "output/models/"

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

        if testing:
            self.testing = open(f"{self.logs_path}testing_loss.csv", "w")
            self.testing.write("fold,batch,t,loss,ssim\n")
            return

        self.training = open(self.logs_path + "training_loss.csv", "w")
        self.training.write("fold,epoch,loss\n")

        self.validation = open(self.logs_path + "validation_loss.csv", "w")
        self.validation.write("fold,epoch,loss\n")

    def save_model(self, model: torch.nn.Module, fold):
        torch.save(model.state_dict(), "%smodel_f%s.pth" % (self.models_path,
                                                            str(fold),))

    def log_training_loss(self, fold, epoch, loss):
        if self.training.closed:
            raise ValueError("Logger is closed.")

        self.training.write("%s,%s,%s\n" % (str(fold),
                                            str(epoch),
                                            str(loss)))

    def log_validation_loss(self, fold, epoch, loss):
        if self.validation.closed:
            raise ValueError("Logger is closed.")

        self.validation.write("%s,%s,%s\n" % (str(fold),
                                              str(epoch),
                                              str(loss)))

    def log_testing_loss(self, fold, batch, t, loss, ssim):
        if self.testing.closed:
            raise ValueError("Logger is closed.")

        self.testing.write("%s,%s,%s,%s,%s\n" % (str(fold),
                                                 str(batch),
                                                 str(t),
                                                 str(loss),
                                                 str(ssim)))

    def close(self):
        self.training.close()
        self.validation.close()
