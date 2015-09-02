from neural_art import combine_images
import numpy as np
import PIL.Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--info', dest='info_image', required=True,
	help='Image providing information')

parser.add_argument('--style', dest='style_image', required=True,
	help='Image providing style')

parser.add_argument('--models_dir', dest='models_dir', default='models',
	help='Directory used to save model files')

parser.add_argument('--output_dir', dest='output_dir', default='output',
	help='Directory used to save produced images')

parser.add_argument('--info_weight', dest='info_weight', type=float, default=2.5,
	help='Ratio of info/style weights in resulting image. Should be in range 1 to 10 for best results')

parser.add_argument('--iterations', dest='iterations', type=int, default=400,
	help='Number of gradient descent steps. 400 is a good value to start with')

args = parser.parse_args()
info_image = np.float32(PIL.Image.open(args.info_image))
style_image = np.float32(PIL.Image.open(args.style_image))

combine_images(info_image,
	style_image,
	args.models_dir,
	args.output_dir,
	args.info_weight,
	args.iterations)