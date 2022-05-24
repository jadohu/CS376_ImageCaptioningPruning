# CS376_ImageCaptioningPruning

Final project for CS376

Main code for image captioning model "Show, Attend, and Tell" implementation is based on [a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).

Before running this code, make sure that you set the appropriate path for each files.

# Dataset

You'd need to download the [Training (13GB)](http://images.cocodataset.org/zips/train2014.zip) and [Validation (6GB)](http://images.cocodataset.org/zips/val2014.zip) images. We will use  [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).

# Training

Before you begin, make sure to save the required data files for training, validation, and testing. To do this, run the contents of create_input_files.py after pointing it to the the Karpathy JSON file and the image folder containing the extracted train2014 and val2014 folders from your downloaded data.

To train your model from scratch, simply run this file –

python train.py

# Inference

See caption.py.

During inference, we cannot directly use the forward() method in the Decoder because it uses Teacher Forcing. Rather, we would actually need to feed the previously generated word to the LSTM at each timestep.

caption_image_beam_search() reads an image, encodes it, and applies the layers in the Decoder in the correct order, while using the previously generated word as the input to the LSTM at each timestep. It also incorporates Beam Search.

visualize_att() can be used to visualize the generated caption along with the weights at each timestep as seen in the examples.

To caption an image from the command line, point to the image, model checkpoint, word map (and optionally, the beam size) as follows –

python caption.py --img='path/to/image.jpeg' --model='path/to/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='path/to/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5

Alternatively, use the functions in the file as needed.

Also see eval.py, which implements this process for calculating the BLEU score on the validation set, with or without Beam Search.
