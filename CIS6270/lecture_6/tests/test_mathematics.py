"""Independent mathematical checks, including failure cases from the lecture."""
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import (MapNet, SequenceNet, finite_map, exact_denoiser, exact_velocity,
                    mixture_posterior, seed_all, DATA_STD, CENTERS)
from continuous import lagrangian_residual, eulerian_residual, semigroup_loss
from categorical import (DecodingClock, categorical_map, composition_target,
                         corrected_logit_teacher, probability_kl)
from posterior import glass_denoiser, fine_tune_surrogate, weighted_diamond_samples
from expanding import (ExpandingNet, local_clock, local_map, gap_counts,
                       bounded_counts, insert_tokens, count_divergence)
from stochastic import sample_coefficients, chen_two, aggregate_tree


class ExactExponential(nn.Module):
    def forward(self,x,s,t,context=None):
        h=t-s
        # Stable at h=0 while keeping the exact off-diagonal expression.
        safe=h.clamp_min(1e-12)
        ratio=torch.where(h.abs()<1e-8,1+h/2+h.square()/6,torch.expm1(h)/safe)
        return x*ratio


class ConstantLogits(nn.Module):
    def forward(self,x,s,t):
        return torch.zeros_like(x)+torch.tensor([.2,-.1],dtype=x.dtype,device=x.device)


class MathematicsTests(unittest.TestCase):
    def setUp(self):seed_all(6270)

    def test_exact_identity_and_semigroup(self):
        model=ExactExponential();x=torch.tensor([[1.2],[-.4]],dtype=torch.float64)
        s=torch.zeros_like(x);u=s+.3;t=s+.9
        torch.testing.assert_close(finite_map(model,x,s,s),x)
        expected=x*math.exp(.9)
        torch.testing.assert_close(finite_map(model,x,s,t),expected)
        torch.testing.assert_close(finite_map(model,finite_map(model,x,s,u),u,t),expected)

    def test_lagrangian_and_eulerian_jvp(self):
        x=torch.tensor([[1.7]],dtype=torch.float64);s=x*0+.2;t=x*0+.8
        for residual in [lagrangian_residual,eulerian_residual]:
            value=residual(ExactExponential(),lambda z,a:z,x,s,t)
            self.assertLess(float(value.abs().max()),1e-10)

    def test_jvp_losses_reach_parameters(self):
        for residual in [lagrangian_residual,eulerian_residual]:
            m=MapNet(2,16);x=torch.randn(8,2);s=torch.rand(8,1)*.3;t=s+.5
            loss=residual(m,lambda z,a:z,x,s,t).square().mean();loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.parameters()),0)

    def test_composition_alone_does_not_identify_motion(self):
        m=MapNet(2,8)
        for p in m.parameters():p.data.zero_()
        x=torch.randn(8,2);s=torch.zeros(8,1);t=s+1
        self.assertEqual(float(semigroup_loss(m,m,x,s,t)),0.)
        self.assertGreater(float((m(x,s,s)-x).square().mean()),0.)

    def test_meanflow_backward_identity(self):
        def average(z,r,t):return z*(-torch.expm1(-(t-r)))/(t-r)
        z=torch.tensor([[1.7]],dtype=torch.float64);r=z*0+.2;t=z*0+.8
        u,du=torch.func.jvp(average,(z,r,t),(z,torch.zeros_like(r),torch.ones_like(t)))
        torch.testing.assert_close(u,z-(t-r)*du,atol=1e-11,rtol=1e-11)

    def test_analytic_mixture_velocity(self):
        x=torch.randn(20,2,dtype=torch.float64);t=torch.rand(20,1,dtype=torch.float64)*.9
        torch.testing.assert_close(exact_velocity(x,t),(exact_denoiser(x,t)-x)/(1-t),atol=1e-10,rtol=1e-10)

    def test_categorical_numerical_step(self):
        class Net(nn.Module):
            def forward(self,x,s,t):return torch.tensor([.1,.7,.2]).log().expand_as(x)
        x=torch.tensor([[[-.2,.6,1.1]]]);y,p=categorical_map(Net(),x,.25,.75)
        torch.testing.assert_close(y,torch.tensor([[[0.,2/3,.5]]]),atol=1e-7,rtol=1e-6)
        self.assertGreater(float(y.sum()),1.1)
        torch.testing.assert_close(p.sum(-1),torch.ones(1,1))

    def test_weighted_probability_composition_matches_maps(self):
        net=SequenceNet(3,4,12);x=torch.randn(6,3,4)
        s=torch.zeros(6,1);u=s+.5;t=s+.75
        q=composition_target(net,x,s,u,t)
        mid,_=categorical_map(net,x,s,u);split,_=categorical_map(net,mid,u,t)
        direct=(1-t[...,None])*x+t[...,None]*q
        torch.testing.assert_close(direct,split)
        torch.testing.assert_close(q.sum(-1),torch.ones(6,3))

    def test_detached_kl_gradient(self):
        logits=torch.tensor([.2,-.1],requires_grad=True);p=torch.tensor([.4,.6])
        probability_kl(logits,p).backward()
        torch.testing.assert_close(logits.grad,logits.softmax(-1)-p)

    def test_discrete_teachers_with_zero_derivative(self):
        x=torch.randn(5,3,2);s=torch.zeros(5,1)+.2;t=s+.5;net=ConstantLogits()
        for kind in ['discrete-lsd','discrete-esd']:
            logits,target,clipped=corrected_logit_teacher(net,net,x,s,t,kind)
            torch.testing.assert_close(target,logits.softmax(-1))
            self.assertEqual(float(clipped),0.)

    def test_endpoint_match_does_not_remove_temporal_residual(self):
        dt=torch.tensor([.2,-.2]);residual=.5*.5*dt
        self.assertAlmostEqual(float(residual.square().sum()),.005,places=7)

    def test_decoding_clock_inverse(self):
        clock=DecodingClock(13);tau=torch.linspace(0,1,101)
        time=clock.inverse(tau)
        self.assertTrue((time[1:]>=time[:-1]).all())
        self.assertEqual(float(time[0]),0.);self.assertEqual(float(time[-1]),1.)

    def test_glass_matches_two_observation_bayes_rule(self):
        inner=torch.tensor([[.3,-.2]],dtype=torch.float64);outer=inner+.4
        s=torch.tensor([[.35]],dtype=torch.float64);t=s+.2
        # Independent derivation by conditioning each Gaussian component twice.
        variance=1/(1/DATA_STD**2+(s/(1-s))**2+(t/(1-t))**2)
        means=variance[:,None]*(CENTERS.double()/DATA_STD**2+s[:,None]*inner[:,None]/(1-s[:,None])**2+t[:,None]*outer[:,None]/(1-t[:,None])**2)
        covariance=torch.tensor([[float((1-s)**2+s**2*DATA_STD**2),float(s*t*DATA_STD**2)],
                                 [float(s*t*DATA_STD**2),float((1-t)**2+t**2*DATA_STD**2)]],dtype=torch.float64)
        residual=torch.stack([inner[:,None]-s[:,None]*CENTERS,outer[:,None]-t[:,None]*CENTERS],-1)
        logits=-.5*torch.einsum('bkdi,ij,bkdj->bk',residual,torch.linalg.inv(covariance),residual)
        expected=(logits.softmax(-1)[...,None]*means).sum(1)
        torch.testing.assert_close(glass_denoiser(inner,s,outer,t),expected,atol=1e-9,rtol=1e-9)

    def test_glass_no_inner_observation(self):
        outer=torch.randn(5,2);t=torch.full((5,1),.4)
        torch.testing.assert_close(glass_denoiser(torch.randn(5,2),torch.zeros_like(t),outer,t),exact_denoiser(outer,t))

    def test_meta_surrogate_gradient(self):
        delta=torch.tensor([[.3]],requires_grad=True)
        w=torch.tensor([[1.],[4.]]);gw=torch.tensor([[.2],[.8]])
        loss=fine_tune_surrogate(delta,w,gw,.5);loss.backward()
        torch.testing.assert_close(delta.grad,2*(w*delta.detach()-.5*gw).mean().reshape(1,1))

    def test_importance_weights_recover_posterior_mean(self):
        x=torch.tensor([[.5,-.3]])
        _,mean,ess=weighted_diamond_samples(x,.5,50000)
        self.assertLess(float((mean-exact_denoiser(x,.5)).abs().max()),.2)
        self.assertGreater(float(ess),100)

    def test_insertion_preserves_order_and_clocks(self):
        x=torch.tensor([[1.,0.],[0.,1.]]);b=torch.tensor([0.,.2]);counts=torch.tensor([1,0,1])
        noise=torch.tensor([[-.2,.4],[.3,-.1]])
        y,bt=insert_tokens(x,b,counts,noise,.5)
        torch.testing.assert_close(y[1:3],x)
        torch.testing.assert_close(bt,torch.tensor([.5,0.,.2,.5]))
        torch.testing.assert_close(local_clock(.75,bt),torch.tensor([.5,.75,.6875,.5]))

    def test_gap_labels_and_global_budget(self):
        torch.testing.assert_close(gap_counts(torch.tensor([1,4]),torch.tensor([0,2,3,5]),6),torch.tensor([1.,2.,1.]))
        counts,_=bounded_counts(torch.tensor([100.,100.,100.]),4)
        self.assertLessEqual(int(counts.sum()),4)
        self.assertTrue((counts>=0).all())

    def test_count_divergence_minimizes_conditional_mean(self):
        prediction=torch.tensor(2.,requires_grad=True)
        count_divergence(torch.tensor([0.,1.,5.]),prediction).mean().backward()
        self.assertAlmostEqual(float(prediction.grad),0.,places=6)

    def test_insertion_interval_zero_and_terminal(self):
        net=ExpandingNet(4,3,8);x=torch.zeros(1,4,3);b=torch.zeros(1,4);m=b.bool();s=torch.tensor([[.3]])
        self.assertEqual(float(net.counts(x,b,m,s,s).sum()),0.)
        t=torch.ones_like(s)
        self.assertTrue(torch.isfinite(net.counts(x,b,m,s,t)).all())

    def test_chen_arithmetic_and_covariance(self):
        left=torch.tensor([.2,.04]);right=torch.tensor([-.1,-.02])
        torch.testing.assert_close(chen_two(left,right,.5,.5),torch.tensor([.1,-.14]))
        L=sample_coefficients(torch.tensor(.3),100000);R=sample_coefficients(torch.tensor(.7),100000)
        C=chen_two(L,R,.3,.7)
        torch.testing.assert_close(torch.cov(C.T),torch.diag(torch.tensor([1.,1/3])),atol=.015,rtol=0)

    def test_same_noise_constant_sde_and_wrong_noise_counterexample(self):
        direct=1+.5+.8*(.2-.1)
        split=(1+.25+.8*.2)+.25+.8*(-.1)
        self.assertAlmostEqual(direct,split);self.assertAlmostEqual(direct,1.58)
        self.assertNotAlmostEqual(direct,1+.5+.8*(-.3))

    def test_chen_tree_matches_increment_sum(self):
        c=torch.randn(6,16,2)
        torch.testing.assert_close(aggregate_tree(c)[:,0],c[:,:,0].sum(1))


if __name__=='__main__':unittest.main()
