from rembg import remove, new_session
import PIL

if __name__ == "__main__":
    
    session = new_session()
    
    input_path = "./source/tests/input.jpg"
    
    input = PIL.Image.open(input_path)
    input_copy = input.copy()

    alpha_foreground = remove(input, session=session, only_mask=True, post_process_mask=True)
    alpha_background = PIL.ImageChops.invert(alpha_foreground)

    input.putalpha(alpha_foreground)
    input.save("./source/tests/foreground.png")
    input.putalpha(alpha_background)
    input.save("./source/tests/background.png")