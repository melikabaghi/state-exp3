"""Direct numerical verification of Theorem 3.1 (lower bound) claims."""
import numpy as np

def run(K=5, d=50, T=20000, eps=0.10, seed=0, policy='uniform'):
    rng = np.random.default_rng(seed)
    B = T // d
    sigma = rng.choice([-1, 1], size=B).astype(float)   # block signs
    sigma[0] = 0.0                               # neutral first block: c_1 = (1/2)1, so the
                                                 # stale vector given over t <= d is uninformative
    astar = 0                                     # a* -> s2 ; others -> s1
    blk = np.arange(T) // d
    # true mean loss of each action at each round
    c = np.full((T, K), 0.5)
    c[:, astar] = 0.5 - eps * sigma[blk]
    # hint m_t = c_{t-d}  (block b-1)
    m = np.vstack([np.tile(c[0], (d, 1)), c[:-d]])   # m_t = c_1 for t <= d
    A = np.empty(T, dtype=int)
    for t in range(T):
        if policy == 'uniform':      A[t] = rng.integers(K)
        elif policy == 'greedy_hint':A[t] = m[t].argmin()
        elif policy == 'always_star':A[t] = astar
        elif policy == 'never_star': A[t] = rng.integers(1, K)
        elif policy == 'oracle':     A[t] = c[t].argmin()      # cheats: sees sigma_b
    loss = c[np.arange(T), A].sum()
    best = c.sum(0).min()
    return loss, best, loss - best, sigma, c, m

d, T, eps, K = 50, 20000, 0.10, 5
print(f"C(h=d,eps): K={K} d={d} T={T} eps={eps}   B={T//d}\n")
print(f"{'policy':<13}{'E[loss]':>12}{'T/2':>10}{'best fixed':>12}{'regret':>10}")
for pol in ['uniform','greedy_hint','always_star','never_star','oracle']:
    L=[];R=[]
    for s in range(200):
        loss,best,reg,_,_,_ = run(K,d,T,eps,seed=s,policy=pol)
        L.append(loss); R.append(reg)
    print(f"{pol:<13}{np.mean(L):>12.1f}{T/2:>10.1f}{np.mean(L)-np.mean(R):>12.1f}{np.mean(R):>10.1f}")

print("\nCLAIM (Lemma 4.2): every non-cheating policy has E[loss] = T/2 EXACTLY,")
print("so regret = comparator fluctuation = eps*d*E[(sum sigma)^+]")
_,_,_,sig,c,m = run(K,d,T,eps,seed=0)
B=T//d
S=np.array([np.random.default_rng(s).choice([-1,1],size=B).sum() for s in range(20000)])
pred = eps*d*np.maximum(S,0).mean()
regs=[run(K,d,T,eps,seed=s,policy='uniform')[2] for s in range(400)]
print(f"  predicted eps*d*E[(sum sigma)^+] = {pred:.2f}")
print(f"  measured  (uniform, 400 seeds)   = {np.mean(regs):.2f}  +/- {np.std(regs)/20:.2f}")
print(f"  Theta form eps*sqrt(d*T)         = {eps*np.sqrt(d*T):.2f}   (Theta hides E[(.)^+]~0.4*sqrt(B))")
