"""Discrete generative models: course and project code.

  common/    shared across projects: filesystem layout, the ESM-2 registry
  project1/  flow matching vs diffusion with guidance, on ESM-2 protein latents

Later projects are added as sibling packages under `dgm`, so anything they
share moves into `dgm.common` rather than being imported across projects.
"""
__version__ = "0.1.0"
