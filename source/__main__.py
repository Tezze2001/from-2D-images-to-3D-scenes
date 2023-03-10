from rembg import remove, new_session
import PIL

def separate_bg_fg(input: PIL.Image):
    session = new_session()

    input_copy = input.copy()

    alpha_foreground = remove(input, session=session, only_mask=True, post_process_mask=True)
    alpha_background = PIL.ImageChops.invert(alpha_foreground)

    input.putalpha(alpha_foreground)
    input_copy.putalpha(alpha_background)
    return (input_copy, input)

if __name__ == "__main__":
    
    input_path = "./source/tests/input.jpg"
    input = PIL.Image.open(input_path)
    (bg, fg) = separate_bg_fg(input)
    (bg1, fg1) = separate_bg_fg(fg)

    bg.save('./source/tests/bg.png')
    bg1.save('./source/tests/2-fg.png')
    fg1.save('./source/tests/fg.png')