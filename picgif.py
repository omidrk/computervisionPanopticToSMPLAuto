import imageio
import argparse
import os

def makegif(args):
    with imageio.get_writer(args.output_path+'/final2.gif', mode='I') as writer:
        # filenames = os.listdir(args.image_path)

        # filenames = sorted(filenames)
        # print(filenames)
        # for filename in filenames:
        #     image = imageio.imread(os.path.join(args.image_path,filename))
        #     for i in range(3):
        #         writer.append_data(image)
        for i in range(2990):
            if i% 200 == 0:
                print(f'{i} items out of 3000 processed.')
            image = imageio.imread(os.path.join(args.image_path,f'img_{i+1}.png'))
            # for i in range(3):
            writer.append_data(image)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--image_path', required=True, type=str, default='pics',
                        help='The path to the image folder')
    parser.add_argument('--output_path', required=True, type=str, default='gifs',
                        help='The path to the gif folder')
    args = parser.parse_args()
    makegif(args)
