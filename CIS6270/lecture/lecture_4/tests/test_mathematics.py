import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from lecture_core import udlm_rates, geometric_cfg
from diffusion import guidance_changes, NoisyClassifier


class Mathematics(unittest.TestCase):
    def test_masked_kl_reduces_to_cross_entropy(self):
        r = .5
        d = torch.tensor([.1,.6,.2,.1], dtype=torch.float64)
        q = torch.tensor([0,r,0,0,1-r], dtype=torch.float64)
        p = torch.cat((r*d, torch.tensor([1-r])))
        actual = (q[q>0] * (q[q>0]/p[q>0]).log()).sum()
        self.assertAlmostEqual(float(actual), float(-r*d[1].log()), places=12)

    def test_udlm_small_step_kl_converges_to_rate_loss(self):
        a = torch.tensor([2.5,.5,.5], dtype=torch.float64)
        b = torch.tensor([17/18,7/18,7/18], dtype=torch.float64)
        target = (a*(a/b).log()+b-a).sum()
        errors = []
        for h in [1e-3,1e-4,1e-5]:
            q = torch.cat((h*a,(1-h*a.sum()).reshape(1)))
            p = torch.cat((h*b,(1-h*b.sum()).reshape(1)))
            local = (q*(q/p).log()).sum()/h
            errors.append(abs(float(local-target)))
        self.assertLess(errors[-1], errors[0]/50)
        self.assertAlmostEqual(float(target), .9071595148, places=8)

    def test_cfg_dna_example(self):
        p = geometric_cfg(torch.full((4,),.25),torch.tensor([.1,.2,.6,.1]),2.)
        torch.testing.assert_close(p, torch.tensor([1,4,36,1])/42.)

    def test_classifier_gradient_matches_directional_derivative(self):
        torch.manual_seed(2)
        model = NoisyClassifier(4).double()
        z = torch.tensor([[0,1,2,3]])
        soft = torch.nn.functional.one_hot(z,5).double().requires_grad_(True)
        t = torch.tensor([.5], dtype=torch.float64)
        f = lambda x: torch.nn.functional.logsigmoid(model(x,t)).sum()
        grad = torch.autograd.grad(f(soft),soft)[0]
        direction = torch.zeros_like(soft);direction[0,0,0]=-1;direction[0,0,1]=1
        h=1e-5
        numeric=(f(soft+h*direction)-f(soft-h*direction))/(2*h)
        self.assertAlmostEqual(float(numeric.detach()),float((grad*direction).sum()),places=8)


if __name__ == '__main__': unittest.main()
