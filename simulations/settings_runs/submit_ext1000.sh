#!/bin/bash
# submit_ext1000.sh -- 2026-09-21: replicates 501-1000 for the cells whose settings are unchanged (cfg 1,3,5,6,7,8), ORIGINAL settings
# (l1 1e-5 at sigma 1, 1e-3 at sigma 3, 120 epochs, patience 20, clip 0.05, K 5), tau0 / tau3 / S2 at rho 0 and 0.3.
# Run through cm_test_v2.R with CM_CLIP=0.05: for these cells it reproduces cm_test.R exactly (674 paired reps, max abs diff 1e-15)
# and also stores the score-form p-values as a diagnostic. Seeds follow the same scheme, so reps 501-1000 extend reps 1-500.
# Outputs go to NEW folders (out/test_ext_rho00, out/test_ext_rho03); the original 500-rep folders are not touched.
cd /nas/longleaf/home/shuaiy/project/corrmix_design; mkdir -p logs; O=$PWD/out; LOG=logs/submit_ext1000.log
for RHO in 0 0.3; do tag=$(echo $RHO | tr -d .); [ "$tag" = "0" ] && tag=00
  for TAU in tau0 tau3 S2; do for CFG in 1 3 5 6 7 8; do name=cmX_${TAU}_r${tag}_c$CFG
    case $CFG in 6|8) TL=8:00:00; MEM=6g;; *) TL=4:00:00; MEM=4g;; esac
    if squeue -u shuaiy -h -n $name | grep -q .; then echo "$(date) $name already queued" | tee -a $LOG; continue; fi
    out=$(sbatch --job-name=$name --nice=0 --time=$TL --mem=$MEM --array=51-100 --export=ALL,SCRIPT=cm_test_v2.R,PREFIX=test_$TAU,OUT_DIR=$O/test_ext_rho$tag,TAU_TYPE=$TAU,CM_RHO=$RHO,CM_CLIP=0.05,CFG=$CFG,RPT=10,MKL_NUM_THREADS=1 cm_multi.sh 2>&1)
    echo "$(date) $name: $out" | tee -a $LOG
  done; done
done
