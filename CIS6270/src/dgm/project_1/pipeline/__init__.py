"""The Project 1 experiment pipeline, split by stage.

  config      device, alphabet, default widths, the shared weight cache
  data        CSV -> standardized ESM-2 latents; reference encoding
  nets        velocity, noise and reward networks; the weight averager
  paths       interpolants, the DDPM schedule, endpoint estimators
  training    the two training loops
  objective   what guidance climbs: senses, setpoints, constraints
  guidance    the reward gradient
  sampling    the two guided samplers
  decoding    latents -> amino-acid sequences
  oracle      the independent judge
  prepare     arguments -> every pre-training choice
  arms        the objective and the guidance arms to sample
  runconfig   the configuration record saved with the samples
  results     results.pt and the FASTA files
  reporting   the tables a run prints about itself
  plots       optional figures, from the in-memory run
  cli         the command line

run_experiment.py at the project root orchestrates these and re-exports the
names other scripts import.
"""
