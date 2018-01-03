netname=cd_chimera2.1gan
python2 train.py --dataroot ./datasets/oxford_pet --name ${netname} --model chimera2_gan \
--dataset_mode munaligned --input_nc 1 --pool_size 50 --no_dropout \
|| echo "Error terminal c2gan" && exit
mkdir ./checkpoints/cd_chimera2gan2
cp ./checkpoints/$netname/latest_net_G_mA.pth ./checkpoints/cd_chimera2gan2/latest_net_G_mA.pth
cp ./checkpoints/$netname/latest_net_G_mB.pth ./checkpoints/cd_chimera2gan2/latest_net_G_mB.pth
python2 train.py --dataroot ./datasets/oxford_pet --name cd_chimera2gan2 --model chimera2_gan \
--dataset_mode munaligned --input_nc 1 --pool_size 50 --no_dropout --isG3 \
|| echo "Error terminal c2gan" && exit
