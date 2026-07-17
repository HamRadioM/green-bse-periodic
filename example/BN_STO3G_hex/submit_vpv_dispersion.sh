DATAPATH="/Users/wenming/green-bse-periodic/green-bse-periodic-bse-solver/example/BN_STO3G_hex"
BSEPATH="/Users/wenming/green-bse-periodic/green-bse-periodic-bse-solver/green_bse"
BETA=1000

# ---- active space for the BSE ladder (band counts around the gap) ----
# empty -> full space.  N_ACT_OCC keeps the n highest occupied bands,
# N_ACT_VIR the n lowest virtual bands.  The mesh cache depends on this
# choice, so its name is tagged automatically.
N_ACT_OCC="4"     # e.g. 2
N_ACT_VIR="4"     # e.g. 2

ACT_ARGS=""
CACHE_TAG=""
[ -n "$N_ACT_OCC" ] && ACT_ARGS="$ACT_ARGS --n_act_occ $N_ACT_OCC" && CACHE_TAG="${CACHE_TAG}_o$N_ACT_OCC"
[ -n "$N_ACT_VIR" ] && ACT_ARGS="$ACT_ARGS --n_act_vir $N_ACT_VIR" && CACHE_TAG="${CACHE_TAG}_v$N_ACT_VIR"

# General VPV-diagonal exciton-dispersion solver (green_bse/vpv_dispersion.py):
# mesh sweep (cached) -> validation incl. off-diagonal weight of VPV(Omega=0)
# -> Wannier interpolation along G-M-K-G -> per-channel continuation ->
# results in vpv_dispersion.h5 (+ /errors group), figure, and an error report
# at the end of the log.
#
# --cont_method   : aaa | plaspole | nevanlinna  (default nevanlinna =
#     bosonic->fermionic auxiliary trick + Pick continuation; cleanest on
#     interpolated data, ~15 min for 24 pairs x 181 points)
# --channel_basis : ov | eigen  (default eigen = static-VPV eigenbasis per
#     mesh Q; diagonal approximation exact at Omega=0)
python $BSEPATH/vpv_dispersion.py \
    --input $DATAPATH/mean_field_input.h5 \
    --sim $DATAPATH/scGW_sim_beta_$BETA.h5 \
    --int_path $DATAPATH/df_int/ \
    --ir_file $DATAPATH/../H2_STO3G_grid/irgrid/1e5.h5 \
    --beta $BETA \
    --channel singlet \
    --cont_method nevanlinna \
    --channel_basis eigen \
    --observable trP \
    --eta 0.5 \
    --e_min 0.0 \
    --e_max 20.0 \
    --path "G:0,0,0 M:0.5,0,0 K:0.333333333,0.333333333,0 G:0,0,0" \
    --npts 60 \
    --dim 2 \
    $ACT_ARGS \
    --cache "vpv_disp_trp_cache$CACHE_TAG.npz" \
    --out vpv_dispersion_o4v4.h5 \
    --fig vpv_dispersion_o4v4
