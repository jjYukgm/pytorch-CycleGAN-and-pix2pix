netname=cd_chimera4gan
python2 train.py --dataroot ./datasets/oxford_pet --name ${netname} --model chimera3_gan \
--dataset_mode munaligned --input_nc 1 --pool_size 50 --no_dropout \
|| {echo "Error terminal c2gan" && exit; }
mkdir ./checkpoints/${netname}2
cp ./checkpoints/$netname/latest_net_G_A.pth ./checkpoints/${netname}2/latest_net_G_A.pth
cp ./checkpoints/$netname/latest_net_G_B.pth ./checkpoints/${netname}2/latest_net_G_B.pth
python2 train.py --dataroot ./datasets/oxford_pet --name ${netname}2 --model chimera3_gan \
--dataset_mode munaligned --input_nc 1 --pool_size 50 --no_dropout --isG3 \
|| { echo "Error terminal c2gan" && exit; }
exit;

