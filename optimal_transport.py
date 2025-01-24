def passthrough(x):
    return x

def sliced_wass(seed=None):
    """
    Generates a Wasserstein function that uses a given seed,
    for reproducibility.
    Seed can still be overwritten if that's what suits you.
    """
    import ot
    def inner_wass(x,y,seed=seed):
        weights1, weights2 = [np.ones(len(x))/len(x), np.ones(len(y))/len(y)]
        try:
            return ot.sliced_wasserstein_distance(x, y, a=w1, b=w2, seed=seed)
        except RuntimeError:
            warnings.warn("Wasserstein did not converge;"+\
                          "if this happens often,"+\
                          "increase Wass max iterations.",\
                          category=RuntimeWarning)
