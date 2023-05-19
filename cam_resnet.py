import cv2
import os
import math
import glob
import torch
import librosa
import argparse
import numpy as np
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
from torch import topk
from scipy import signal as ss
from torch.nn import functional as F
from framework.gaussian import get_Gaussian_Kernel
from package.FL.attackers import Attackers
from package.FL.resnet import ResNet18
from package.FL.resnext import ResNeXt29_2x64d
from package.FL.regnet import RegNetY_400MF
from pytorch_grad_cam.utils.image import show_cam_on_image
from functools import cmp_to_key

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument('--folder', type=str, help='It will process all the pictures in this folder')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


# save pixel infomation
class PIXEL:
    def __init__(self, value, i, j):
        self.value = value # mask
        self.i = i # coordinate(i, j)
        self.j = j

# define compare function
def cmp(a, b):
    return b.value - a.value

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam":HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}
    
    # get Gassian filter (sigma = 3)
    gaussian_kernel = get_Gaussian_Kernel(sigma = 3)
    # Class
    keys = ['five', 'stop', 'house', 'on', 'happy', 'marvin', 'wow', 'no', 'left', 'four', 'tree', 'go', 'cat', 'bed', 'two', 'right', 'down', 'seven', 'nine', 'up', 'sheila', 'bird', 'three', 'one', 'six', 'dog', 'eight', 'off', 'zero', 'yes']
    values = [i for i in range(30)]
    my_dict = {k : v for k, v in zip(keys, values)}
    # Read global model
    global_model = RegNetY_400MF()
    PATH = './model/resnet_15_0.2_V1_start.pth'
    global_model.eval().cuda()
    global_model.load_state_dict(torch.load(PATH))
    # Read examine model
    model = ResNet18()
    PATH = './student_model/student_resnet_15_0.2_V1_start.pth'
    model.eval().cuda()
    model.load_state_dict(torch.load(PATH))
    # Get trigger
    my_attackers = Attackers()
    trigger = my_attackers.poison_setting(15, "start", True)
    target_label = 7
    # target layer
    target_layers = [global_model.layer4]
    
    # record statistics
    ac = 0
    wa = 0    
    count = 0
    verify = 0
    poison = 0
    ac_global = 0
    for path in glob.glob('./TEST_DATA/*'):
        print("IMAGE: ", count)
        count += 1
        # get file path/name
        file_path, file_name = os.path.split(path)
        # Get Label
        s = ""
        for i in file_name:
            if i == '_':
                break
            s += i
        label = my_dict[s]

        # Read .wav
        signal, sr = librosa.load(path, sr = 44100)
        # Resample the size, so that it can do add operation
        # must do twice, otherwise it has error
        signal = ss.resample(signal, int(44100/signal.shape[0]*signal.shape[0]))
        signal = ss.resample(signal, int(44100/signal.shape[0]*signal.shape[0]))

        # add trigger
        signal = signal + trigger
        # Get mfccs
        mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=40, n_fft=1103, hop_length=int(sr/100))
        # Get mfccs tensor
        mfccs_tensor = torch.tensor(mfccs, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda() # 1 * 1 * 40 * 100
        # initialize probability
        probability = torch.tensor([0.0 for i in range(30)]).cuda()
        
        # Get Global Model Prediction
        output1 = global_model(mfccs_tensor).cuda()
        prob1 = F.softmax(output1).data.squeeze().cuda()
        probability += prob1
        class_idx1 = topk(prob1, 1)[1].int()
        res1 = int(class_idx1[0])
        # record acc
        if res1 == label: 
          ac_global += 1
        # record posion 
        if res1 == target_label:
          poison += 1

        # Get Examine Model Prediction
        output2 = model(mfccs_tensor).cuda()
        prob2 = F.softmax(output2).data.squeeze().cuda()
        probability += prob2
        class_idx2 = topk(prob2, 1)[1].int()
        res2 = int(class_idx2[0])

        # print prediction & label
        print("CORRECT LABEL: ", label)
        print("GLOBAL LABEL:", res1)
        print("VALIDATION LABEL:", res2)

        # Suspicious Data Recognition (SDR)
        # Examine == Global -> Get predict
        if res1 == res2:
            verify += 1
            if res1 == label:
                ac += 1
                print("AC", ac)
            else:
                wa += 1
                print("WA")
            continue

        # Suspicious Feature Identification (SFI)
        targets = None
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=global_model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda) as cam:
            
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=mfccs_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)
            
            grayscale_cam = grayscale_cam[0, :]
            heatmap = show_cam_on_image(signal, grayscale_cam, use_rgb=True)
            map = np.array(grayscale_cam)
            # save heatmap
            #cv2.imwrite("./" + file_name + ".jpg", heatmap)
            
        # Select important pixel   
        pixel_value = []
        for i in range(len(map)):
            for j in range(len(map[0])):
                pixel_value.append(PIXEL(map[i][j], i, j))
        # Sorting 
        pixel_value = sorted(pixel_value, key = cmp_to_key(cmp)) 

        # Feature Cancellation Mechanism (FCM)
        new_map = [[0.0 for i in range(100)] for j in range(40)]
        for i in range(40):
            for j in range(100):
                tmp = mfccs[i][j] * gaussian_kernel[1][1]
                if i - 1 >= 0 and j - 1 >= 0:
                    tmp += mfccs[i - 1][j - 1] * gaussian_kernel[0][0]
                if i - 1 >= 0 and j + 1 < 100:
                    tmp += mfccs[i - 1][j + 1] * gaussian_kernel[0][2]
                if i + 1 < 40 and j - 1 >= 0:
                    tmp += mfccs[i + 1][j - 1] * gaussian_kernel[2][0]
                if i + 1 < 40 and j + 1 < 100:
                    tmp += mfccs[i + 1][j + 1] * gaussian_kernel[2][2]
                if i - 1 >= 0:
                    tmp += mfccs[i - 1][j] * gaussian_kernel[0][1]
                if i + 1 < 40:
                    tmp += mfccs[i + 1][j] * gaussian_kernel[2][1]
                if j - 1 >= 0:
                    tmp += mfccs[i][j - 1] * gaussian_kernel[1][0]
                if j + 1 < 100:
                    tmp += mfccs[i][j + 1] * gaussian_kernel[1][2]
                new_map[i][j] = tmp 

        # erase influence with Tp = {0.55, 0.50, 0.45}
        for threshold in [0.55, 0.50, 0.45]:
          for i in range(4000):
              x = pixel_value[i]
              if x.value < threshold:
                  break
              mfccs[x.i][x.j] = new_map[x.i][x.j]
        
          # Predict again
          mfccs_tensor = torch.tensor(mfccs, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda() # 1 * 1 * 40 * 100
          output = global_model(mfccs_tensor).cuda()
          prob = F.softmax(output).data.squeeze().cuda()
          probability += prob
          class_idx = topk(prob, 1)[1].int()
          res = int(class_idx[0])

        # KD-based Inference (KDI) -> choose highest probability as final prediction
        new_pred = topk(probability, 1)[1].int()
        new_pred = int(new_pred)
        print("Prediction(with framework): ", new_pred)  
        if new_pred == label:
            ac += 1
        else:
            wa += 1
           
    # Print Accuracy
    print("ACCURACY(without framework): ", ac, " / ", ac + wa, " = ", ac / (ac + wa))
    print("ACCURACY(after framework): ", ac, " / ", ac + wa, " = ", ac / (ac + wa))
    print("GLOBAL POISON: ", poison, " / ", ac + wa, " = ", poison / (ac + wa))
    print("VERIFICATION:", verify, " / ", ac + wa, " = ", verify / (ac + wa))
    