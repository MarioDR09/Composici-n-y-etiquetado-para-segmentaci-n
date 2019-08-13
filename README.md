# Composici-n-y-etiquetado-para-segmentaci-n
Este directorio cuenta con el archivo de composición y etiquetado; así como las respectivas redes para segmentación de imágenes.

# Create Synthetic Images and Masks
In this section, we will use "image_composition.py" to randomly pick foregrounds and automatically super-impose them on backgrounds. You will need a number of foreground cutouts with transparent backgrounds. For example, you might have a picture of an eagle with a completely transparent background. Due to the need for transparency, these images should be .png format (.jpg doesn't have transparency). I cut out my foregrounds with [GIMP](https://www.gimp.org/), which is free.

## Using the sample dataset
For this guide, all examples assume you'll be using the box_dataset_synthetic sample dataset. Find it here [../datasets/README.md](../datasets/README.md). Download it and extract the contents to "../datasets/box_dataset_synthetic".

## Custom dataset directory setup:
- Inside the "datasets" directory, create a new folder for your dataset (e.g. "wild_animal_dataset")
- Inside that dataset directory, create a folder called "input"
- Inside "input", create two folders called "foregrounds" and "backgrounds"
- Inside "foregrounds", create a folder for each super category (e.g. "bird", "lizard")
- Inside each foreground super category folder, create a folder for each category (e.g. "eagle", "owl")
- Inside each category folder, add all foreground photos you intend to use for the respective category (e.g. all of you eagle foreground cutouts)
- Inside "backgrounds", add all background photos you intend to use

Run "image_composition.py" to create your images and masks
```
python ./python/image_composition.py --input_dir ./datasets/box_dataset_synthetic/input --output_dir ./datasets/box_dataset_synthetic/output --count 10 --width 512 --height 512
```

# Create COCO Instances JSON
Now we're going to use the images, masks, and json to create COCO instances.

Optional: Run "coco_json_utils.py" with --help to see the documentation. This will explain the next command.
```
python ./python/coco_json_utils.py --help
```
Run the command with the correct parameters
```
python ./python/coco_json_utils.py -md ./datasets/box_dataset_synthetic/output/mask_definitions.json -di ./datasets/box_dataset_synthetic/output/dataset_info.json
```

You will now have a new json file called "coco_instances.json". This is contains all of your COCO json!
