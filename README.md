# DeepPerson_DeID

## set model
* Download the .pth models from TransReID-SSL.
* Replace the .txt file to .pth in tr_test/transreid_pytorch/model directory.

## Libraries
* Install the torch=1.8.0+cu111
    pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
* Then install prerequisites in tr_test and ViTPose
    pip install -r requirements.txt

## Train
* Enjoy the adversarial training process.
    gan.py --config_file configs/market/strong.yml --lr 0.0001 --lrG 0.0002 --lr_dis 0.0002 --reID_weight 30.0 --w-adv 1.0 --pose_start 0 --made --project reid_weight_up --name reid_1 MODEL.DEVICE_ID ('1') SOLVER.LAMBDA_L1 0.03 SOLVER.MAX_EPOCHS 100 INPUT.RE_PROB 0. OUTPUT_DIR log/transreid/reid_weight_up_6
