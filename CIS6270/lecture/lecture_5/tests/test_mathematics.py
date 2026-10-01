import itertools
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lecture_core import fisher_path, gumbel_path, proposal
from flows import reference_model


class Mathematics(unittest.TestCase):
    def test_gat_posterior_average_matches_full_master_equation(self):
        states = list(itertools.product(range(4), repeat=2))
        targets = [(a,a) for a in range(4)]
        def marginal(t):
            p=np.zeros(16); numerator=np.zeros((16,16))
            for a in states:
                for b in targets:
                    for xi,x in enumerate(states):
                        probability=np.prod([(1-t)*(x[j]==a[j])+t*(x[j]==b[j]) for j in range(2)])/64
                        p[xi]+=probability
                        for j in range(2):
                            if x[j]!=b[j]:
                                y=list(x);y[j]=b[j];yi=states.index(tuple(y))
                                numerator[xi,yi]+=probability/(1-t)
            rates=numerator/p[:,None]
            np.fill_diagonal(rates,-rates.sum(1))
            return p,rates
        t=.4;h=1e-5
        p,r=marginal(t)
        derivative=(marginal(t+h)[0]-marginal(t-h)[0])/(2*h)
        np.testing.assert_allclose(p@r,derivative,atol=1e-9)

    def test_fisher_midpoint_and_tangent(self):
        y,v=fisher_path(torch.full((1,1,4),.25),torch.tensor([[2]]),torch.tensor([.5]))
        torch.testing.assert_close(y.square(),torch.tensor([[[1/12,1/12,3/4,1/12]]]))
        self.assertLess(float((y*v).sum(-1).abs().max()),1e-6)

    def test_gumbel_velocity_keeps_the_realized_noise(self):
        target=torch.tensor([[0,1,2,3]])
        def path(t):
            torch.manual_seed(12)
            return gumbel_path(target,torch.tensor([t]))
        z,v=path(.3);h=1e-3
        numeric=(path(.3+h)[0]-path(.3-h)[0])/(2*h)
        torch.testing.assert_close(v,numeric,atol=2e-4,rtol=2e-3)
        self.assertLess(float(v.sum(-1).abs().max()),1e-6)

    def test_mh_detailed_balance_on_all_two_base_sequences(self):
        states=[''.join(x) for x in itertools.product('ACGT',repeat=2)]
        score=lambda s:(s.count('G')+.5*s.count('C'))/2
        weight=np.array([np.exp(3*score(s)) for s in states]);pi=weight/weight.sum()
        matrix=np.zeros((16,16))
        for i,x in enumerate(states):
            neighbors,q=proposal(x,score,3.)
            for y,forward in zip(neighbors,q):
                j=states.index(y);back,reverse=proposal(y,score,3.)
                accept=min(1,pi[j]*reverse[back.index(x)]/(pi[i]*forward))
                matrix[i,j]=forward*accept
            matrix[i,i]=1-matrix[i].sum()
        np.testing.assert_allclose(pi[:,None]*matrix,pi[None,:]*matrix.T,atol=1e-14)
        np.testing.assert_allclose(pi@matrix,pi,atol=1e-14)


if __name__ == '__main__': unittest.main()
