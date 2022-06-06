#python test.py --data_root /datasets/anonygan-dataset --name exp2base-multiid-viz --ch_input 6 --ckpt lightning_logs/default/exp2base/checkpoints/epoch=699-step=44799.ckpt

#python test.py --data_root /datasets/anonygan-dataset --name exp0base --ch_input 3 --ckpt lightning_logs/default/exp0base/checkpoints/epoch=699-step=44799.ckpt
#python test.py --data_root /datasets/anonygan-dataset --name exp0base_bis --ch_input 6 --ckpt lightning_logs/default/exp0_bis/checkpoints/epoch=699-step=44799.ckpt
#python test.py --data_root /datasets/anonygan-dataset --name exp2base_lr --ch_input 6 --ckpt lightning_logs/default/exp2base_new_lr/checkpoints/epoch=699-step=44799.ckpt

# python test.py --data_root /home/tonyw/ba/anonygan/data/aligned --name test --ch_input 6 --ckpt /home/tonyw/ba/anonygan/ckpts/anonygan.ckpt

python test.py --data_root /home/tonyw/ba/anonygan/data/img_align_celeba --name baseline_add_rf-multiid --ch_input 6 --ckpt ./ckpts/anonygan.ckpt