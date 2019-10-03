import numpy as np


def spots(spots, numframes=2000, avg_on_time = 20, on_fraction=0.1):
    numspots = len(spots)

    p_off = 1-on_fraction
    k_off = 1/avg_on_time
    k_on =( k_off - p_off*k_off)/p_off 
    
    print(f"p_off={p_off}, k_on={k_on}, k_off={k_off}")
    
    blinkstate = np.random.binomial(1, on_fraction, size=numspots)

    xyI = np.zeros((numframes,numspots,3))
    oncounts = np.zeros(numframes, dtype=int)
    for f in range(numframes):
        turning_on = (1 - blinkstate) * np.random.binomial(1, k_on, size=numspots)
        remain_on = blinkstate * np.random.binomial(1, 1 - k_off, size=numspots)
        blinkstate = remain_on + turning_on

        c = np.sum(blinkstate)
        oncounts[f] = c

        xyI[f] = spots
        xyI[f,blinkstate == 0, 2] = 0
        
#        if(f % 20 == 0):
 #           print(f"Fraction on : {c/numspots:.5f}")

    return xyI
