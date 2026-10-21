# CL-SCA
covariance learning via successive convex approximations 

## Directory Structure

```
root/
├── AD_simul.py              functions for simulation
├── main_AD_simul.py         function to run the AD simulation 
├── src/                     source functions of the toolbox
├──────base.py               contains the base class for covariance learning algorithms
├──────methods/              CL-SCA/CP-FP/CL-Hyb algorithms and baseline CL-methods (CWO, MSBL, CLMP,CLMM)
├──────plotting/             scripts for plotting
├──────data/                 scripts for data generation 
└── saved_results/           folder to save figures,json files
```

## Citation

If you use this code, please cite the accompanying manuscript:

```bibtex
@inproceedings{ollila2026joint,
  title   = {Efficient {SCA} Algorithms for Sparse Bayesian Learning},
  author  = {Ollila, Esa and Esfandiari, Majdoddin  and Palomar, Daniel P. and Wang, Xinjue},
  year    = {2026},
  pages ={1--13},
  note={arXiv:TBD [eess.SP]}
}
```

```bibtex
@inproceedings{ollila2026joint,
  title   = {Joint Activity Detection and Channel Estimation for Massive Random Access Using {SBL} and {SCA}},
  author  = {Ollila, Esa and Esfandiari, Majdoddin  and Palomar, Daniel P.},
  booktitle={2026 IEEE International Workshop on Signal Processing and Artificial Intelligence for Wireless Communications ({SPAWC})},
  year    = {2026},
  month = {Sep 6--9,},
  address = {Athens, Greece},
  pages ={1--5},
  note={arXiv:2604.12620 [eess.SP]}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact:

Esa Ollila (firstname DOT lastname AT aalto dot fi)

