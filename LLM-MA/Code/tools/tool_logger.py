from transformers import TrainerCallback
import matplotlib.pyplot as plt
import pandas as pd

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_entry = {
                'step': state.global_step,
                'loss': logs.get("loss"),
                'eval_loss': logs.get("eval_loss"),
            }
            self.logs.append(log_entry)

    def to_csv(self, filename="loss_logs.csv"):
        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        print(f"Saved loss logs to {filename}")

    def get_dataframe(self):
        return pd.DataFrame(self.logs)
        
    def save_plot(self, filename="loss_plot.png"):
      df = self.get_dataframe()
      train_df = df[df["loss"].notna()]
      eval_df = df[df["eval_loss"].notna()]
      
      plt.figure(figsize=(10, 6))
      # Training loss: blue line with dots
      plt.plot(train_df["step"], train_df["loss"],
               label="Training Loss", color="blue", marker="o", linestyle='-')
      
      # Evaluation loss: orange line with dots
      plt.plot(eval_df["step"], eval_df["eval_loss"],
               label="Eval Loss", color="orange", marker="o", linestyle='-')
      
      plt.xlabel("Step")
      plt.ylabel("Loss")
      plt.title("QLoRA Training & Evaluation Loss")
      plt.legend()
      plt.grid(True)
      plt.savefig(filename)