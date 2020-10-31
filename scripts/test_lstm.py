from dataset.task_dataset import TaskHandler
from models.simple_lstm import SimpleLSTM

dataset = TaskHandler()
model = SimpleLSTM(embed_dim=128,
                   hidden_dim=128)

tr, ts, _ = dataset.sample_task()
out = model(tr)

print("Model output:", out)