import numpy as np

def alpha_help(voltage):
    if np.isclose(voltage,10):
        return 0.1
    else:
        y = 10 - voltage
        x = np.divide(y,10)
        e = np.exp(x)
        x = e - 1
        y = 0.01 * y
        x = np.divide(y, x)
        return x
        
def alpha_n(V):
    alpha = np.vectorize(alpha_help)
    return alpha(V)

def beta_help(voltage):
    y = np.divide(voltage, -80)
    y = np.exp(y)
    y = 0.125 * y
    return y
    
def beta_n(voltage):
    beta = np.vectorize(beta_help)
    return beta(voltage)

def n(a, b):
    t = np.vectorize(n_help)
    return t(a, b)

def n_help(a, b):
    x = a + b
    x = np.divide(a, x)
    return x

def tau_help(a, b):
    x = a + b
    x = np.divide(1, x)
    return x
    
def tau(a, b):
    t = np.vectorize(tau_help)
    return t(a, b)

def alpha_m_help(V):
    if np.isclose(V,25):
        return 1
    else:
        x = 25 - V
        y = 0.1 * x
        x = np.divide(x, 10)
        x = np.exp(x)
        x = x - 1
        x = np.divide(y, x)
        return x
    
def beta_m_help(V):
    x = np.divide(V, 18)
    x = x * -1
    x = np.exp(x)
    return 4 * x

def beta_m(V):
    beta = np.vectorize(beta_m_help)
    return beta(V)
    
def alpha_m(V):
    alpha = np.vectorize(alpha_m_help)
    return alpha(V)

def alpha_h_help(V):
    x = np.divide(V, 20)
    x = x*-1
    x = np.exp(x)
    x = 0.07 * x
    return x 

def alpha_h(V):
    alpha = np.vectorize(alpha_h_help)
    return alpha(V)

def beta_h_help(V):
    x = 30 - V
    x = np.divide(x, 10)
    x = np.exp(x)
    x = x + 1
    x = np.divide(1, x)
    return x
    
def beta_h(V):
    beta = np.vectorize(beta_h_help)
    return beta(V)

def h_help(a, b):
    x = a + b
    x = np.divide(a, x)
    return x

def m_help(a, b):
    x = a + b
    x = np.divide(a, x)
    return x
    
def EulerHodkinHuxley(EL, DeltaT, EK, maxT, gNa, ENa, Cm, V0, Iext, gL, gK):
    T = np.arange(0, maxT, DeltaT)
    V = np.zeros(len(T))
    V[0] = V0
    n = np.zeros(len(T))
    n[0] = n_help(alpha_help(Iext(V[0])), beta_help(Iext(V[0])))
    h = np.zeros(len(T))
    h[0] = h_help(alpha_h_help(V0), beta_h_help(V0))
    m = np.zeros(len(T))
    m[0] = m_help(alpha_m_help(V0), beta_m_help(V0))
    GK = np.zeros(len(T))
    GNa = np.zeros(len(T))
    tauN = np.zeros(len(T))
    tauN[0] = tau_help(alpha_help(V0), beta_help(V0))
    tauH = np.zeros(len(T))
    tauH[0] = tau_help(alpha_h_help(V0), beta_h_help(V0))
    tauM = np.zeros(len(T))
    tauM[0] = tau_help(alpha_m(V0), beta_m(V0))
    x = gNa * np.power(m[0], 3)
    x = x * h[0]
    GNa[0] = x
    GK[0] = gK * np.power(n[0], 4)
    
    for i in range(1, len(T)):
        tauN[i] = tau_help(alpha_n(V[i - 1]), beta_n(V[i - 1]))
        x = np.divide(DeltaT, tauN[i])
        x = 1 - x
        x = x * n[i - 1]
        x = x + (np.divide(DeltaT, tauN[i]) * n_help(alpha_n(V[i - 1]), beta_n(V[i - 1])))
        n[i] = x
        tauH[i] = tau_help(alpha_h(V[i - 1]), beta_h(V[i - 1]))
        x = np.divide(DeltaT, tauH[i])
        x = 1 - x
        x = x*h[i - 1]
        x = x + (np.divide(DeltaT, tauH[i]) * h_help(alpha_h(V[i - 1]), beta_h(V[i - 1])))
        h[i] = x
        tauM[i] = tau_help(alpha_m(V[i - 1]), beta_m(V[i - 1]))
        x = np.divide(DeltaT, tauM[i])
        x = 1 - x
        x = x*m[i - 1]
        x = x + (np.divide(DeltaT, tauM[i]) * m_help(alpha_m(V[i - 1]), beta_m(V[i - 1])))
        m[i] = x
        GNa[i] = gNa * (np.power(m[i], 3)) * h[i]
        GK[i] = gK * (np.power(n[i], 4))
        x = [np.divide(GNa[i],Cm), np.divide(GK[i],Cm), np.divide(gL,Cm)]
        x = np.multiply(x, -DeltaT)
        x = np.multiply(x,[V[i - 1] - ENa, V[i - 1] - EK, V[i - 1] - EL])
        x = np.sum(x)
        x = x + np.divide(DeltaT,Cm)*Iext(T[i])
        V[i] = V[i - 1] + x
        
    return {'T': T, 'V': V, 'GK': GK, 'GNa': GNa, 'n': n,'h': h,'m': m, 'tauN': tauN, 'tauH': tauH, 'tauM': tauM}




