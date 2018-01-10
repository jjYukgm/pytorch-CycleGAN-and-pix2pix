python test.py --dataroot ./datasets/oxford_pet --name cd_chimera3gan2 --model chimera3_gan --phase train --no_dropout --how_many 4213\
    --which_epoch 20 --dataset_mode munaligned --isG3 --input_nc 1 && \
python test.py --dataroot ./datasets/oxford_pet --name cd_chimera3gan2 --model chimera3_gan --phase train --no_dropout --how_many 4213\
    --which_epoch 50 --dataset_mode munaligned --isG3 --input_nc 1  && \
python test.py --dataroot ./datasets/oxford_pet --name cd_chimera3gan2 --model chimera3_gan --phase train --no_dropout --how_many 4213\
    --which_epoch 100 --dataset_mode munaligned --isG3 --input_nc 1  && \
python test.py --dataroot ./datasets/oxford_pet --name cd_chimera3gan2 --model chimera3_gan --phase train --no_dropout --how_many 4213\
    --which_epoch 150 --dataset_mode munaligned --isG3 --input_nc 1  && \
python test.py --dataroot ./datasets/oxford_pet --name cd_chimera3gan2 --model chimera3_gan --phase train --no_dropout --how_many 4213\
    --which_epoch 200 --dataset_mode munaligned --isG3 --input_nc 1
