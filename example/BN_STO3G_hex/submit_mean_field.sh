PYPATH="/Users/wenming/green-mbpt/python"

python $PYPATH/init_data_df.py \
    --atom BN_geom.dat \
    --a BN_lattice.dat \
    --nk 6 6 1 \
    --basis STO-3G \
    --output mean_field_input.h5 \
    --restricted True \
    --space_symm False 
