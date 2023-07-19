# from-2D-images-to-3D-scenes
## Enviroment setup
It is required python 3.6 and it is suggested to use python eviroment.

First of all clone the repository and enter in the folder just created. 
```shell
foo@bar:~$ git clone https://github.com/Tezze2001/from-2D-images-to-3D-scenes.git
foo@bar:~$ cd from-2D-images-to-3D-scenes
```

Create new python enviroment inside .venv folder.
```shell
foo@bar:~/from-2D-images-to-3D-scenes$ python3.6 -m venv .venv
```

Activate the virtual enviroment.
```shell
foo@bar:~/from-2D-images-to-3D-scenes$ source .venv/bin/activate
```

Update pip package.
```shell
(.venv) foo@bar:~/from-2D-images-to-3D-scenes$ pip install --upgrade pip
```

Install all the dependences.
```shell
(.venv) foo@bar:~/from-2D-images-to-3D-scenes$ pip install -r requirements.txt
```

Clone inside the main directory the wrapper of Panda3D
```shell
(.venv) foo@bar:~/from-2D-images-to-3D-scenes$ git clone https://github.com/Tezze2001/panda3d_viewer.git
```

Download models and checkpoints of the neural networks inside the following paths:
- [isnet-general-use.pth](https://drive.google.com/file/d/1nV57qKuy--d5u1yvkng9aXW1KS4sOpOi/view): put this file inside the folder models_source>isnet
- [best.ckpt](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg/models): put this file inside the folder models_source>lama
- [torchscript_resnet50_fp32.pth](https://drive.google.com/file/d/1-t9SO--H4WmP7wUl1tVNNeDkq47hjbv4/view?usp=share_link): put this file inside the folder models_source>removebackgroundv2
- [indexnet_matting.pth.tar](https://github.com/poppinace/indexnet_matting/tree/master/pretrained): put this file inside the folder models_source>indexmatting

Now you have everything to execute sorce code.

## Using
To execute the source code follow these steps:
- create the folder "input" inside the main directory
- copy the image in the folder created before
- update IMG_INPUT variable in main.py with name and extension of the file
    ```python
    # main.py
    IMG_INPUT = 'name.extension'
    ```
- update paths dictionary with the output path of the animation
    ```python
    # main.py
   
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
        'OUTPUT_VIDEO': os.path.abspath('./output.avi') # here
    }
    ```
- run the code
    ```shell
    (.venv) foo@bar:~$ python main.py
    ```

## Reference
Here you can find everything that I used:
- [IS-NET](https://github.com/xuebinqin/DIS.git)
- [LAMA](https://github.com/advimman/lama.git)
- [RemoveBackgroundV2](https://github.com/PeterL1n/BackgroundMattingV2.git)
- [IndexNet](https://github.com/poppinace/indexnet_matting.git)
- [Wrapper for Panda3D](https://github.com/Tezze2001/panda3d_viewer.git)