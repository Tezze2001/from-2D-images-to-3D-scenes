import torch
from PIL import Image
import os
from torchvision.transforms.functional import to_tensor, to_pil_image

def imagematting_rembgv2(img_input, bg_input, model, result_path):
    device = torch.device('cuda')
    #precision = torch.float16

    model = torch.jit.load(model)

    '''
    model.backbone_scale = 0.25
    model.refine_mode = 'sampling'
    model.refine_sample_pixels = 80_000
    '''

    model = model.to(device)

    list = [(img_input, bg_input)]
    for i in list:
        print(i)
        src = Image.open(i[0])
        bgr = Image.open(i[1])
        src_clone = src.copy()
        src = to_tensor(src).cuda().unsqueeze(0)
        bgr = to_tensor(bgr).cuda().unsqueeze(0)

        model.refine_mode = 'sampling'

        if src.size(2) <= 2048 and src.size(3) <= 2048:
            model.backbone_scale = 1/4
            model.refine_sample_pixels = 80_000
        else:
            model.backbone_scale = 1/8
            model.refine_sample_pixels = 320_000


        pha, fgr = model(src, bgr)[:2]
        com = pha * fgr
        #src_clone.putalpha(to_pil_image(pha[0].cuda()))
        #src_clone.save(os.path.join(result_path, i[0].split('/')[-1]))
        to_pil_image(pha[0].cuda()).save(result_path)