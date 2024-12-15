# training
python train.py  --segmentation_model 'Unet' --logs_file_path 'logs/cryonuseg_unet.txt' --model_save_path 'models/cryonuseg_unet.pth';
python train.py  --segmentation_model 'UnetPlusPlus' --logs_file_path 'logs/cryonuseg_unetplus.txt' --model_save_path 'models/cryonuseg_unetplus.pth';
python train.py  --segmentation_model 'MAnet' --logs_file_path 'logs/cryonuseg_manet.txt' --model_save_path 'models/cryonuseg_manet.pth';
python train.py  --segmentation_model 'Linknet' --logs_file_path 'logs/cryonuseg_linknet.txt' --model_save_path 'models/cryonuseg_linknet.pth';
python train.py  --segmentation_model 'FPN' --logs_file_path 'logs/cryonuseg_fpn.txt' --model_save_path 'models/cryonuseg_fpn.pth';
python train.py  --segmentation_model 'PSPNet' --logs_file_path 'logs/cryonuseg_pspnet.txt' --model_save_path 'models/cryonuseg_pspnet.pth';
python train.py  --segmentation_model 'PAN' --logs_file_path 'logs/cryonuseg_pan.txt' --model_save_path 'models/cryonuseg_pan.pth';
python train.py  --segmentation_model 'DeepLabV3Plus' --logs_file_path 'logs/cryonuseg_deeplab.txt' --model_save_path 'models/cryonuseg_deeplab.pth';
