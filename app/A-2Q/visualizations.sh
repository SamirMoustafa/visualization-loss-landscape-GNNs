# GCN Cora
python node_level_1.py --lr_quant_bit_fea 0.1 --lr_quant_scale_fea 0.04 --a_loss 2.5 --lr_quant_scale_weight 0.02 --lr_quant_scale_xw 0.008 --drop_out 0.35 --weight_decay 0.02 --dataset_name Cora --model GCN --max_cycle 1
# GCN-CiteSeer
python node_level_1.py --lr_quant_bit_fea 0.1 --lr_quant_scale_fea 0.04 --a_loss 1.5 --lr_quant_scale_weight 0.008 --lr_quant_scale_xw 0.008 --drop_out 0.5 --weight_decay 0.015 --dataset_name CiteSeer --model GCN --max_cycle 1
# GCN-PubMed
python node_level_1.py --lr_quant_bit_fea 0.02 --lr_quant_scale_fea 0.005 --a_loss 0.1 --lr_quant_scale_weight 0.005 --dataset_name PubMed --model GCN --max_cycle 1

# GIN-Cora
python node_level_1.py --lr_quant_bit_fea 0.02 --lr_quant_scale_fea 0.05 --a_loss 2 --lr_quant_scale_weight 0.005  --lr_quant_scale_xw 0.005 --dataset_name Cora --model GIN --max_cycle 1
# GIN-CiteSeer
python node_level_1.py --lr_quant_bit_fea 0.02 --lr_quant_scale_fea 0.05 --a_loss 0.5 --lr_quant_scale_weight 0.005 --lr_quant_scale_xw 0.005 --dataset_name CiteSeer --model GIN --max_cycle 1
# GIN-PubMed
python node_level_1.py --lr_quant_bit_fea 0.02 --lr_quant_scale_fea 0.05 --a_loss 0.5 --lr_quant_scale_weight 0.005 --lr_quant_scale_xw 0.005 --dataset_name PubMed --model GIN --max_cycle 1

# GAT-Cora
python gat_nc_lsb.py --weight_decay 1e-3 --lr 0.01 --lr_quant_scale_fea 0.05 --lr_quant_scale_weight 0.005 --lr_quant_scale_gat_fea 0.05 --lr_quant_scale_gat 0.005 --lr_quant_bit_fea 0.1 --a_loss 0.3 --drop_out 0.6 --drop_attn 0.6 --dataset_name Cora --max_cycle 1
# GAT-CiteSeer
python gat_nc_lsb.py --weight_decay 1e-3 --lr 0.01 --lr_quant_scale_fea 0.05 --lr_quant_scale_weight 0.005 --lr_quant_scale_gat_fea 0.05 --lr_quant_scale_gat 0.005 --lr_quant_bit_fea 0.1 --a_loss 0.3 --drop_out 0.6 --drop_attn 0.6 --dataset_name CiteSeer --max_cycle 1
# GAT-PubMed
python gat_nc_lsb.py --weight_decay 1e-3 --lr 0.01 --lr_quant_scale_fea 0.05 --lr_quant_scale_weight 0.005 --lr_quant_scale_gat_fea 0.05 --lr_quant_scale_gat 0.005 --lr_quant_bit_fea 0.1 --a_loss 0.3 --drop_out 0.6 --drop_attn 0.6 --dataset_name PubMed --max_cycle 1