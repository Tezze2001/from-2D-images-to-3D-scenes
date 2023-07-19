
import cv2 as cv
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

from math import cos, sin, pi
from direct.task import Task

from models_source.isnet.util import imagematting_isnet
from models_source.removebackgroundv2.util import imagematting_rembgv2
from models_source.indexmatting.index_matting import imagematting_indexmatting
from models_source.lama.predict import imageinpainting_lama

from panda3d_viewer.panda3d_viewer import Viewer
from panda3d_viewer.panda3d_viewer import ViewerConfig



models = {
    'isnet': os.path.abspath('./models_source/isnet/isnet-general-use.pth'),
    'lama': {
        'checkpoint': os.path.abspath('./models_source/lama/best.ckpt'),
        'model': os.path.abspath('./models_source/lama/config.yaml')
    },
    'removebackgroundv2': os.path.abspath('./models_source/removebackgroundv2/torchscript_resnet50_fp32.pth'),
    'indexmatting': os.path.abspath('./models_source/indexmatting/indexnet_matting.pth.tar'),
}

INPUT_DIR = './input/'
IMG_INPUT = 'GT04.png'
IMG_EXT = '.png'

paths = {
    'image': os.path.abspath(INPUT_DIR + IMG_INPUT),
    'image_resized': os.path.abspath(INPUT_DIR + IMG_INPUT.split('.')[0] + IMG_EXT),
    'lama_mask': os.path.abspath(INPUT_DIR + IMG_INPUT.split('.')[0] + '_mask001' + IMG_EXT),
    'lama_mask_dilated': os.path.abspath(INPUT_DIR + 'lama/' + IMG_INPUT.split('.')[0] + '_mask001' + IMG_EXT),
    'BG': os.path.abspath(INPUT_DIR + 'BG' + IMG_EXT),
    'rembgv2_seg': os.path.abspath(INPUT_DIR + IMG_INPUT.split('.')[0] + '_seg' + IMG_EXT),
    'trimap': os.path.abspath(INPUT_DIR + 'trimap' + IMG_EXT),
    'FG_seg': os.path.abspath(INPUT_DIR + 'FG_seg' + IMG_EXT),
    'FG': os.path.abspath(INPUT_DIR + 'FG' + IMG_EXT),
    'OUTPUT_VIDEO': os.path.abspath('./output.avi')
}

def resize_image(input_path, output_path):
    imm = Image.open(input_path)
    imm_resized = imm.resize((imm.size[0] - (imm.size[0] % 4), imm.size[1] - (imm.size[1] % 4)))
    imm_resized.save(output_path)
    return imm_resized.size

def generate_trimap(alpha):
    k_size = 3
    iterations = 10
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv.dilate(alpha, kernel, iterations=iterations, borderType=cv.BORDER_CONSTANT)
    eroded = cv.erode(alpha, kernel, iterations=iterations, borderType=cv.BORDER_CONSTANT)
    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    
    return trimap

def binarization_mask(input_path):
    imm = np.array(Image.open(input_path).convert('L'))
    imm[imm != 0] = 255
    return imm

def dilate_mask(input_path, output_path):
    imm = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated = cv.dilate(imm, kernel, iterations=10, borderType=cv.BORDER_CONSTANT)
    cv.imwrite(output_path, dilated)

def bin_dil_mask(input_path, output_path):
    Image.fromarray(binarization_mask(input_path).astype('uint8'), 'L').save('./input/lama/fg_seg_bin_ext.png')
    dilate_mask('./input/lama/fg_seg_bin_ext.png', output_path)

def apply_mask(img_path, fg_alpha_path, out_fg_path):
    img = Image.open(img_path)
    fg_alpha = Image.open(fg_alpha_path)
    img.putalpha(fg_alpha)
    img.save(out_fg_path)

def spinCameraTask(viewer, width, task):
    angleDegrees = task.time * 50
    angleRadians = angleDegrees * pi / 180.0
    x =  2* sin(angleRadians)/2
    y =  2* cos(angleRadians)/2
    viewer.reset_camera(pos=(x, y, (width/10)/80 * 100), look_at=(x, y, 0))
    return Task.cont

width, height = resize_image(paths['image'], paths['image_resized'])

print('--IS-NET--')
imagematting_isnet(paths['image_resized'], models['isnet'], paths['lama_mask'])

# binarizzation
Image.fromarray(binarization_mask(paths['lama_mask']).astype('uint8'), 'L').save(paths['lama_mask'])
                                                                                                                           
print('--LAMA--')
imageinpainting_lama(os.path.abspath(INPUT_DIR), models['lama']['model'], models['lama']['checkpoint'],paths['BG'])
print('--RBV2--')
imagematting_rembgv2(paths['image_resized'],paths['BG'], models['removebackgroundv2'], paths['rembgv2_seg'])
print('--TRIMAP--')
Image.fromarray(generate_trimap(np.array(Image.open(paths['rembgv2_seg']).convert('L'))).astype('uint8'), 'L').save(paths['trimap'])
print('--IndexMatting--')
imagematting_indexmatting(paths['image_resized'], paths['trimap'], models['indexmatting'], paths['FG_seg'])
apply_mask(paths['image_resized'], paths['FG_seg'], paths['FG'])

print('--Creating foreground--')

os.makedirs(INPUT_DIR + 'lama') if not os.path.exists(INPUT_DIR + 'lama') else None
os.system('cp ' + paths['image_resized'] + ' ' + INPUT_DIR + 'lama/' + paths['image_resized'].split('/')[-1])  
bin_dil_mask(paths['lama_mask'], paths['lama_mask_dilated'])

imageinpainting_lama(INPUT_DIR + 'lama', models['lama']['model'], models['lama']['checkpoint'],paths['BG'])

print('--Creating scene--')

config = ViewerConfig()
config.set_window_size(1920,1080)
config.enable_antialiasing(True, multisamples=4)
config.show_axes(False)
config.show_floor(False)
config.show_grid(False)
config.show_fps_meter(False)

with Viewer(window_type='offscreen', config=config) as viewer:
    viewer.append_group('root')
    print('Adding meshes')

    viewer.append_plane('root', 'FG', (width/20, height/20))
    viewer.set_material('root', 'FG', color_rgba=(1, 1, 1, 1), texture_path=paths['FG'])
    viewer.move_nodes('root', {'FG': ((0, 0, 50), (1, 0, 0, 0))})

    viewer.append_plane('root', 'BG', (width/10, height /10))
    viewer.set_material('root', 'BG', color_rgba=(1, 1, 1, 1), texture_path=paths['BG'])
    viewer.move_nodes('root', {'BG': ((0, 0, 0), (1, 0, 0, 0))})
    print('Control moving')
    viewer.add_task(spinCameraTask, "Moving", extra_args = [viewer, width], append_task=True)
    print('recording film')
    viewer.save_movie(paths['OUTPUT_VIDEO'], 'XVID', 30, 10)
