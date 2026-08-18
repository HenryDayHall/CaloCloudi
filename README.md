[![Run python tests](https://github.com/HenryDayHall/CaloCloudi/actions/workflows/ci.yml/badge.svg)](https://github.com/HenryDayHall/CaloCloudi/actions/workflows/ci.yml)
[![cov](https://HenryDayHall.github.io/CaloCloudi/badges/coverage.svg)](https://github.com/HenryDayHall/CaloCloudi/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# CaloCloudi; CaloClouds diffusion

This is currently a work in progress.

# installation

Gotta have setuptools==69.0.3

# to use


- setup a virtual env, gonna add a requirements.txt, not there yet
- Adjust the output path in the config file.
- Try `python3 scripts/train_teacher.py`
- Sample with `python3 scripts/generate_test_set.py <path to model file` e.g. `python3 scripts/generate_test_set.py /data/dust/user/dayhallh/data/CaloClouds_diffusion/logs/2026_08_05__19_20_26/checkpoints/2026-08-06_11-20-44_model.pt`
