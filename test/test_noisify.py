import numpy as np
import pandas as pd
from scipy import stats

from opti.problems import ZDT2, Cake
from opti.tools.noisify import _add_noise_to_data, noisify_problem_with_gaussian


def test_add_noise_to_data():
    cake = Cake()
    cake.f = lambda df: pd.DataFrame(
        data=np.zeros((len(df), cake.n_outputs)),
        columns=cake.outputs.names,
    )

    def no_noise(Y):
        return Y

    def to_zero(Y):
        return Y * 0

    Y = cake.get_data()[cake.outputs.names]
    Y_not_noisy = _add_noise_to_data(Y, [no_noise] * len(Y), cake.outputs)
    assert np.allclose(Y, Y_not_noisy)

    # outputs are clipped to their domain bounds after noise is applied, which is why the first output ("calories") is set to 300.
    Y_not_noisy = _add_noise_to_data(Y, [to_zero] * len(Y), cake.outputs)
    assert np.allclose(Y_not_noisy, [300, 0, 0])


def test_noisify_problem_with_gaussian():
    n_samples = 5000
    mu = 0.1
    sigma = 0.05

    zdt2 = ZDT2()
    zdt2.create_initial_data(n_samples)
    zdt2_gaussian = noisify_problem_with_gaussian(zdt2, mu=mu, sigma=sigma)

    reference = np.clip(stats.norm.rvs(loc=mu, scale=sigma, size=n_samples), 0, 1)
    Y_noise = zdt2_gaussian.get_Y() - zdt2.get_Y()
    for col in Y_noise.T:
        s, _ = stats.ks_2samp(col, reference)
        assert s < 0.1
