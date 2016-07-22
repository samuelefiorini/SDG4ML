#!/bin/bash
# Launch synth_run.py with multiple options

# number of samples
N_SAMPLES=(50 100 200 500 1000)

# exponents for the number of dimensions
N_EXP=(3 4 5)

# base for the number of dimensions
N_BASE=(1 2 5)

# relevant features (constant in the exponents of the number of dimensions)
N_RELEVANT=(25 50 100)

# other options
STRATEGY=multivariate_groups
RHO=0.25
OUT=pkl

# for n in 50 100 200 500 1000; do
for i in $(seq 0 4); do
  # Get the number of samples
  N=${N_SAMPLES[i]}

  # Get the number of dimensions
  for j in $(seq 0 2); do
    # Get the exponent for the number of dimensions
    exp=${N_EXP[j]}
    molt=`echo '10^'${exp} | bc`
    # Get the base for the number of dimensions
    for x in $(seq 0 2); do
      D=`echo ${N_BASE[x]}*$molt | bc`
      K=${N_RELEVANT[j]}
      python synth_run.py --strategy $STRATEGY -n $N -d $D -k $K -rho $RHO -o $OUT
      echo $N 'x' $D 'done!'
    done # bases
  done # exponents

done # samples
