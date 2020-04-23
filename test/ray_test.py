import tqdm
import ml.dataloader
import time

measurements = []
WORKLOAD = 200000
for i in range(2, 8):
    r = ml.dataloader.RayManager(blowout=2**i)
    start_time = time.time()
    c = 0
    for x in iter(r):
        c+=1
        if c==WORKLOAD:
            break
            # t.update(x[1].num_graphs//3)
            # t.update(1)
    duration = time.time()-start_time
    print("NUM_WORKERS",i,"TIME", duration)
    measurements.append((2**i, duration, 1.0 * WORKLOAD / duration))
    print(measurements)
    del r