import time
from contextlib import contextmanager

@contextmanager
def timer(label):
    start = time.time()
    yield # 여기서 실제 코드가 실행됨
    end = time.time()
    print(f"⏱️ {label} 소요 시간: {end - start:.4f}초")