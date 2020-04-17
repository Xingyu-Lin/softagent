from PIL import Image, ImageDraw, ImageSequence
import io

for env in ["PassWater", "PourWater", "ClothFold", "ClothDrop", "ClothFlatten", "RopeFlatten"]:
    im = Image.open('data/planet_open_loop_predictions/{}.gif'.format(env))

    frames = []
    for idx, frame in enumerate(ImageSequence.Iterator(im)):
        frame = frame.convert('RGB')
        
        for pidx in range(8):
            d = ImageDraw.Draw(frame)
            d.text((10 + pidx * 130, 120), "Frame" + str(idx), fill=(0, 0, 0))
            del d

        for pidx in range(8):
            d = ImageDraw.Draw(frame)
            d.text((10 + pidx * 130, 250), "Frame" + str(idx), fill=(0, 0, 0))
            del d
        
        frames.append(frame)

    frames[0].save('data/planet_open_loop_predictions/{}_labeled.gif'.format(env), save_all=True, append_images=frames[1:])