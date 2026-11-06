import numpy as np
def gwet_ac1(a,b,labels):
    a=list(a); b=list(b); n=len(a); q=len(labels)
    po=np.mean([x==y for x,y in zip(a,b)])
    pi=[(sum(x==l for x in a)+sum(y==l for y in b))/(2*n) for l in labels]
    pe=sum(p*(1-p) for p in pi)/(q-1)
    return (po-pe)/(1-pe)
def pabak(a,b,labels):
    po=np.mean([x==y for x,y in zip(a,b)]); q=len(labels)
    return (q*po-1)/(q-1)
