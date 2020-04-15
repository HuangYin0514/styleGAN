
from tqdm import tqdm
from datetime import datetime
import time

train_now = datetime.now().timestamp()
print(train_now)
for _ in tqdm(range(100000), mininterval=0.000000001, desc=f'lalala'):
    print(_)
    if _ % 10 == 0:
        if datetime.now().timestamp() - train_now > 10:
            break
