from sources import b_policy ,pi_policy,gamma,generate_episode

def ordinary_importance_sampling(episodes):
    numer , denom = 0.0,0.0
    for ep in episodes:
        G, rho = 0,1
        for t in reversed(range(len(ep))):
            s,a,r = ep[t]
            G = r+gamma*G
            rho *=pi_policy(s)[a]/b_policy(s)[a]
        numer+=rho*G
        denom +=1
    return numer /denom

def weighted_importance_sampling(episodes):
    numer,denom =0.0,0.0
    for ep in episodes:
        G,rho = 0,1
        for t in reversed(range(len(ep))):
            s,a,r = ep[t]
            G =r+gamma*G
            rho *=pi_policy(s)[a]/b_policy(s)[a]
        numer +=rho*G
        denom +=rho 
    return numer / denom if denom !=0 else 0

def discounting_aware_is(episodes):
    numer , denom = 0.0,0.0
    for ep in episodes:
        T = len(ep)
        for t in range(T):
            s,a,r = ep[t]
            G_partial , rho_partial = 0.0,1.0
            for h in range(t,T):
                sh,ah,rh = ep[h]
                G_partial +=rh 
                rho_partial *=pi_policy(sh)[ah]/b_policy(sh)[ah]
                weight = (1-gamma)*(gamma **(h-t))
                numer +=weight*rho_partial *G_partial
                denom +=weight*rho_partial
    return numer/denom if denom !=0 else 0
def  per_decision_is(episodes):
    numer , denom = 0.0,0.0
    for ep in episodes:
        T = len(ep)
        for t in range(T):
            s,a,_ = ep[t]
            G_t,rho = 0.0,1.0
            for k in range(t,T):
                sk,ak,rk = ep[k]
                if k>t:
                    rho *= pi_policy(sk)[ak]/b_policy(sk)[ak]
                G_t +=(gamma **(k-t)) *rho *rk
            numer +=G_t
            denom +=1
    return numer/denom
if __name__=="__main__":
    episodes = [generate_episode(b_policy,5) for _ in range(10000)]
    v_ois = ordinary_importance_sampling(episodes)
    v_wis = weighted_importance_sampling(episodes)
    v_dais = discounting_aware_is(episodes)
    v_pdis = per_decision_is(episodes)
    print(f"Ordinary IS :   {v_ois:.3f}")
    print(f"weighted IS :   {v_wis:.3f}")
    print(f"Discount-aware IS :   {v_dais:.3f}")
    print(f"Per-decision IS :   {v_pdis:.3f}")
