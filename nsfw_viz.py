import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import cv2
import numpy as np
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import torchvision

#model = resnet50(pretrained=True)
model = torchvision.models.__dict__['resnet101'](pretrained=True)

num_ftrs = model.fc.in_features
classes=['drawings', 'neutral', 'nsfw']
model.fc = nn.Linear(num_ftrs, len(classes))

#support muti gpu
#model = nn.DataParallel(model, device_ids=args.device)
#model.cuda()

checkpoint = torch.load('model_1_1200.pth', map_location='cpu')

#need to rename the state_dict to what resnet expects
state_dict_renamed = {}

for k, v in checkpoint['model'].items():
    new_key = k.replace('module.', '')
    state_dict_renamed[new_key] = v

# load the model state
#model.load_state_dict(checkpoint['model'])
model.load_state_dict(state_dict_renamed)



#model.eval()
#model.cuda()



target_layers = [model.layer4[-1]]

#input_tensor = # Create an input tensor image for your model..


image_list = [
    'nsfw-neutral/1.jpg',
    'nsfw-neutral/2.jpg',
    'nsfw-neutral/3.jpg',
    'nsfw-neutral/4.jpg',
    'nsfw-neutral/5.jpg',
    'nsfw-neutral/6.jpg',
    'nsfw-neutral/7.jpg',
    'nsfw-neutral/8.jpg'
]


for img_file in image_list:
    print(img_file)

    rgb_img = cv2.imread(img_file, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    #cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)
    cam = GradCAM(model=model, target_layers=target_layers)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    #target_category = 281

    target_category = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # In this example grayscale_cam has only one image in the batch:
    #visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'{img_file}_cam.jpg', cam_image)
    #cv2.imwrite(f'{img_file}_gb.jpg', gb)
    #cv2.imwrite(f'{img_file}_cam_gb.jpg', cam_gb)

