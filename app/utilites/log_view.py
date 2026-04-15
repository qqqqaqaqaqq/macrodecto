import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def load_all_scalars(log_dir, tag):
    all_events = []

    # 하위 폴더까지 모든 event 파일 탐색
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if 'events.out.tfevents' in file:
                ea = EventAccumulator(os.path.join(root, file))
                ea.Reload()
                if tag in ea.Tags().get('scalars', []):
                    all_events.extend(ea.Scalars(tag))

    # step 기준으로 정렬 후 중복 제거
    all_events.sort(key=lambda e: e.step)
    seen = set()
    unique_events = []
    for e in all_events:
        if e.step not in seen:
            seen.add(e.step)
            unique_events.append(e)

    steps = [e.step for e in unique_events]
    values = [e.value for e in unique_events]
    return steps, values

def log_view(base_dir):
    train_steps, train_values = load_all_scalars(base_dir, 'Loss/train')
    val_steps, val_values = load_all_scalars(base_dir, 'Loss/val')

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_values, label='Train Loss')
    plt.plot(val_steps, val_values, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train / Val Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()