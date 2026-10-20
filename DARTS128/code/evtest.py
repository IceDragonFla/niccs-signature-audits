import sys
import numpy as np
N = 512
rawp, truthp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
raw = np.fromfile(rawp, dtype=np.float64)
off = 3 * N + 5
xz0 = raw[off:off+N]; xz1 = raw[off+N:off+2*N]
truth = np.fromfile(truthp, dtype=np.int8).astype(np.float64)
ev = np.concatenate([xz0 - truth[:N], xz1 - truth[N:2*N]])
np.concatenate([ev, raw[3*N:3*N+3]]).tofile(outp)
print('rms=%.4f a0=%.4f a1=%.4f ae=%.4f' % (np.sqrt((ev**2).mean()), *raw[3*N:3*N+3]))
