#!/usr/bin/env python3

import json
import warnings
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance

class MaskJsonUtils():
    """ Creamos un achivo de definición JSON para las máscara.
    """

    def __init__(self, output_dir):
        """ Inicializamos las clases.
        Args:
            output_dir: el directorio donde la definición será salvada (la salida)
        """
        self.output_dir = output_dir
        self.masks = dict()
        self.super_categories = dict()

    def add_category(self, category, super_category):
        """ Añade una nueva categoría al set de la correspondiente supercategorías
        Args:
            categoría: e.g. 'águila'
            super_categoría: e.g. 'pájaro'
        Returns:
            True si es exitoso, False si la categoría estaba en el diccionario
        """
        if not self.super_categories.get(super_category):
            # Super category doesn't exist yet, create a new set
            self.super_categories[super_category] = {category}
        elif category in self.super_categories[super_category]:
            # Category is already accounted for
            return False
        else:
            # Add the category to the existing super category set
            self.super_categories[super_category].add(category)

        return True # Addition was successful

    def add_mask(self, image_path, mask_path, color_categories):
        """ Toma el path de una imagen, su correspondiente máscara, su categoría y color
        y lo añade al diccionario
        Args:
            image_path: el path relativo a la imagen, e.g. './images/00000001.png'
            mask_path: el path relativo a la máscara, e.g. './masks/00000001.png'
            color_categories: las categorías de cada color, para esta mpascara en particular,
                representada por un color rgb (diccionario).
                (the color category associations are not assumed to be consistent across images)
        Returns:
            True si es exitoso, False si la imagen ya estaba en el diccionario
        """
        if self.masks.get(image_path):
            return False # image/mask is already in the dictionary

        # Create the mask definition
        mask = {
            'mask': mask_path,
            'color_categories': color_categories
        }

        # Add the mask definition to the dictionary of masks
        self.masks[image_path] = mask

        # Regardless of color, we need to store each new category under its supercategory
        for _, item in color_categories.items():
            self.add_category(item['category'], item['super_category'])

        return True # Addition was successful

    def get_masks(self):
        """ Gets all masks that have been added
        """
        return self.masks

    def get_super_categories(self):
        """ Un diccionario de las listas de categorías en una super-categoría
        """
        serializable_super_cats = dict()
        for super_cat, categories_set in self.super_categories.items():
            # Convertimos a lista porque los sets no son serializables
            serializable_super_cats[super_cat] = list(categories_set)
        return serializable_super_cats

    def write_masks_to_json(self):
        # Serializamos máscaras y categorías
        serializable_masks = self.get_masks()
        serializable_super_cats = self.get_super_categories()
        masks_obj = {
            'masks': serializable_masks,
            'super_categories': serializable_super_cats
        }

        # JSON de salida
        output_file_path = Path(self.output_dir) / 'mascaras.json'
        with open(output_file_path, 'w+') as json_file:
            json_file.write(json.dumps(masks_obj))

class ImageComposition():
    """ Hacemos la composición de forma random, aplicando transformaciones la los foregrounss para crear
         una imagen sintética combinada.
    """

    def __init__(self):
        self.allowed_output_types = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types = ['.png', '.jpg', '.jpeg']
        self.zero_padding = 8 # 00000027.png, supports up to 100 million images
        self.max_foregrounds = 3
        self.mask_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        assert len(self.mask_colors) >= self.max_foregrounds, 'longitud de los colores de las mascaras >= max_foregrounds'

    def _validate_and_process_args(self, args):

        self.silent = args.silent

        # Valida la cuenta
        assert args.count > 0, 'cuenta debe ser mayor que 0'
        self.count = args.count

        # Validamos alto y ancho
        assert args.width >= 64, 'ancho tiene que ser mayor que 64'
        self.width = args.width
        assert args.height >= 64, 'altura tiene que ser mayor que 64'
        self.height = args.height

        # Validamos y pricesamos el formato de salida
        if args.output_type is None:
            self.output_type = '.jpg' # default
        else:
            if args.output_type[0] != '.':
                self.output_type = f'.{args.output_type}'
            assert self.output_type in self.allowed_output_types, f'output_type is not supported: {self.output_type}'

        # Validamos directorios de entrada y salida
        self._validate_and_process_output_directory()
        self._validate_and_process_input_directory()

    def _validate_and_process_output_directory(self):
        self.output_dir = Path(args.output_dir)
        self.images_output_dir = self.output_dir / 'imágenes'
        self.masks_output_dir = self.output_dir / 'máscaras'

        # Creamos directorios
        self.output_dir.mkdir(exist_ok=True)
        self.images_output_dir.mkdir(exist_ok=True)
        self.masks_output_dir.mkdir(exist_ok=True)

        if not self.silent:
            # REvisamos si hay algo en el directorio
            for _ in self.images_output_dir.iterdir():
                # Si hay algo vemos si el usuario desea quitarlo
                should_continue = input('output_dir no esta vacío, los archivos será sobrescritos.\nContinuar (y/n)? ').lower()
                if should_continue != 'y' and should_continue != 'yes':
                    quit()
                break

    def _validate_and_process_input_directory(self):
        self.input_dir = Path(args.input_dir)
        assert self.input_dir.exists(), f'input_dir no existe: {args.input_dir}'

        for x in self.input_dir.iterdir():
            if x.name == 'foregrounds':
                self.foregrounds_dir = x
            elif x.name == 'backgrounds':
                self.backgrounds_dir = x

        assert self.foregrounds_dir is not None, 'foregrounds no encontrado en input_dir'
        assert self.backgrounds_dir is not None, 'backgrounds no encontrado en input_dir'

        self._validate_and_process_foregrounds()
        self._validate_and_process_backgrounds()

    def _validate_and_process_foregrounds(self):
        # Validamos foregrounds y los pasamos a un diccionario.
        # Esperamos estructura:
        # + foregrounds_dir
        #     + super_category_dir
        #         + category_dir
        #             + foreground_imagen.png

        self.foregrounds_dict = dict()

        for super_category_dir in self.foregrounds_dir.iterdir():
            if not super_category_dir.is_dir():
                warnings.warn(f'archivo encontrado en directorio de foregrounds (esperaba directorio super-category), ignorar: {super_category_dir}')
                continue


            for category_dir in super_category_dir.iterdir():
                if not category_dir.is_dir():
                    warnings.warn(f'archivo encontrado en directorio de super-category (esperaba directorio de categorías), ignorar: {category_dir}')
                    continue


                for image_file in category_dir.iterdir():
                    if not image_file.is_file():
                        warnings.warn(f'un directorio adrento de super-categpría, ignorando: {str(image_file)}')
                        continue
                    if image_file.suffix != '.png':
                        warnings.warn(f'foreground debe ser .png, saltamos: {str(image_file)}')
                        continue


                    super_category = super_category_dir.name
                    category = category_dir.name

                    if super_category not in self.foregrounds_dict:
                        self.foregrounds_dict[super_category] = dict()

                    if category not in self.foregrounds_dict[super_category]:
                        self.foregrounds_dict[super_category][category] = []

                    self.foregrounds_dict[super_category][category].append(image_file)

        assert len(self.foregrounds_dict) > 0, 'no se encontraron foregrouns validos'

    def _validate_and_process_backgrounds(self):
        self.backgrounds = []
        for image_file in self.backgrounds_dir.iterdir():
            if not image_file.is_file():
                warnings.warn(f'directorio en supercategorías, ignorando: {image_file}')
                continue

            if image_file.suffix not in self.allowed_background_types:
                warnings.warn(f'los fondos dben corresponder{str(self.allowed_background_types)}, ignorando: {image_file}')
                continue

            # Valid file, add to backgrounds list
            self.backgrounds.append(image_file)

        assert len(self.backgrounds) > 0, 'no hay backgrounds'

    def _generate_images(self):
        # Generates a number of images and creates segmentation masks, then
        # saves a mask_definitions.json file that describes the dataset.

        print(f'Generando {self.count} imágenes con máscaras...')

        mju = MaskJsonUtils(self.output_dir)

        # Creamos todas las imágenes/máscaras (with tqdm to have a progress bar)
        for i in tqdm(range(self.count)):
            # escogemos un fondo
            background_path = random.choice(self.backgrounds)

            num_foregrounds = random.randint(1, self.max_foregrounds)
            foregrounds = []
            for fg_i in range(num_foregrounds):
                # Randomly choose a foreground
                super_category = random.choice(list(self.foregrounds_dict.keys()))
                category = random.choice(list(self.foregrounds_dict[super_category].keys()))
                foreground_path = random.choice(self.foregrounds_dict[super_category][category])

                # Get the color
                mask_rgb_color = self.mask_colors[fg_i]

                foregrounds.append({
                    'super_category':super_category,
                    'category':category,
                    'foreground_path':foreground_path,
                    'mask_rgb_color':mask_rgb_color
                })

            # Composición
            composite, mask = self._compose_images(foregrounds, background_path)

            save_filename = f'{i:0{self.zero_padding}}' # e.g. 00000023.jpg


            composite_filename = f'{save_filename}{self.output_type}' # e.g. 00000023.jpg
            composite_path = self.output_dir / 'images' / composite_filename # e.g. mi_output_dir/images/00000023.jpg
            composite = composite.convert('RGB') # remove alpha
            composite.save(composite_path)


            mask_filename = f'{save_filename}.png' #simpre png
            mask_path = self.output_dir / 'masks' / mask_filename # e.g. my_output_dir/masks/00000023.png
            mask.save(mask_path)

            color_categories = dict()
            for fg in foregrounds:

                mju.add_category(fg['category'], fg['super_category'])
                color_categories[str(fg['mask_rgb_color'])] = \
                    {
                        'category':fg['category'],
                        'super_category':fg['super_category']
                    }


            mju.add_mask(
                composite_path.relative_to(self.output_dir).as_posix(),
                mask_path.relative_to(self.output_dir).as_posix(),
                color_categories
            )


        mju.write_masks_to_json()

    def _compose_images(self, foregrounds, background_path):

        # Abrimos fondo y convertimos a RGBA
        background = Image.open(background_path)
        background = background.convert('RGBA')

        # REcortamos fondo (self.width x self.height)
        bg_width, bg_height = background.size
        max_crop_x_pos = bg_width - self.width
        max_crop_y_pos = bg_height - self.height
        assert max_crop_x_pos >= 0, f'achira deseada, {self.width}, mayor que ancho de fondo, {bg_width}, for {str(background_path)}'
        assert max_crop_y_pos >= 0, f'altura deseada, {self.height}, mayor que alto de fondo, {bg_height}, for {str(background_path)}'
        crop_x_pos = random.randint(0, max_crop_x_pos)
        crop_y_pos = random.randint(0, max_crop_y_pos)
        composite = background.crop((crop_x_pos, crop_y_pos, crop_x_pos + self.width, crop_y_pos + self.height))
        composite_mask = Image.new('RGB', composite.size, 0)

        for fg in foregrounds:
            fg_path = fg['foreground_path']

            # transformaciones
            fg_image = self._transform_foreground(fg, fg_path)

            # Escogemos posición al azar
            max_x_position = composite.size[0] - fg_image.size[0]
            max_y_position = composite.size[1] - fg_image.size[1]
            assert max_x_position >= 0 and max_y_position >= 0, \
            f'foreground {fg_path} es demasiado grande ({fg_image.size[0]}x{fg_image.size[1]}) para la salida requerida ({self.width}x{self.height}), revisar parámetros de entrada'
            paste_position = (random.randint(0, max_x_position), random.randint(0, max_y_position))

            # Create a new foreground image as large as the composite and paste it on top
            new_fg_image = Image.new('RGBA', composite.size, color = (0, 0, 0, 0))
            new_fg_image.paste(fg_image, paste_position)

            # Alpha
            alpha_mask = fg_image.getchannel(3)
            new_alpha_mask = Image.new('L', composite.size, color = 0)
            new_alpha_mask.paste(alpha_mask, paste_position)
            composite = Image.composite(new_fg_image, composite, new_alpha_mask)

            alpha_threshold = 200
            mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
            uint8_mask = np.uint8(mask_arr) # This is composed of 1s and 0s

            mask_rgb_color = fg['mask_rgb_color']
            red_channel = uint8_mask * mask_rgb_color[0]
            green_channel = uint8_mask * mask_rgb_color[1]
            blue_channel = uint8_mask * mask_rgb_color[2]
            rgb_mask_arr = np.dstack((red_channel, green_channel, blue_channel))
            isolated_mask = Image.fromarray(rgb_mask_arr, 'RGB')
            isolated_alpha = Image.fromarray(uint8_mask * 255, 'L')

            composite_mask = Image.composite(isolated_mask, composite_mask, isolated_alpha)

        return composite, composite_mask

    def _transform_foreground(self, fg, fg_path):

        fg_image = Image.open(fg_path)
        fg_alpha = np.array(fg_image.getchannel(3))
        assert np.any(fg_alpha == 0), f'foreground needs to have some transparency: {str(fg_path)}'

        # ** Aplicamos transformaciones **
        # Rotamosforeground
        angle_degrees = random.randint(0, 359)
        fg_image = fg_image.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)

        # Escalamos el foreground
        scale = random.random() * .5 + .5 # Pick something between .5 and 1
        new_size = (int(fg_image.size[0] * scale), int(fg_image.size[1] * scale))
        fg_image = fg_image.resize(new_size, resample=Image.BICUBIC)

        # Ajustamos brillos
        brightness_factor = random.random() * .4 + .7 # Pick something between .7 and 1.1
        enhancer = ImageEnhance.Brightness(fg_image)
        fg_image = enhancer.enhance(brightness_factor)

        # Añadimos transformaciones aquí...

        return fg_image

    def _create_info(self):

        if self.silent:
            return

        should_continue = input('¿Craer el info json? (y/n) ').lower()
        if should_continue != 'y' and should_continue != 'yes':
            print('No hay problema. Siempre puedes crear el json manualmente.')
            quit()

        print('Nota: puedes crar json manualemente para actualizar.')
        info = dict()
        info['descripion'] = input('Description: ')
        info['url'] = input('URL: ')
        info['version'] = input('Version: ')
        info['contributor'] = input('Contributor: ')
        now = datetime.now()
        info['year'] = now.year
        info['date_created'] = f'{now.month:0{2}}/{now.day:0{2}}/{now.year}'

        image_license = dict()
        image_license['id'] = 0

        should_add_license = input('¿Añadimos licencia? (y/n) ').lower()
        if should_add_license != 'y' and should_add_license != 'yes':
            image_license['url'] = ''
            image_license['name'] = 'None'
        else:
            image_license['name'] = input('License name: ')
            image_license['url'] = input('License URL: ')

        dataset_info = dict()
        dataset_info['info'] = info
        dataset_info['license'] = image_license

        # Write the JSON output file
        output_file_path = Path(self.output_dir) / 'dataset_info.json'
        with open(output_file_path, 'w+') as json_file:
            json_file.write(json.dumps(dataset_info))

        print('Creado exitosamente {output_file_path}')


    # Start here
    def main(self, args):
        self._validate_and_process_args(args)
        self._generate_images()
        self._create_info()
        print('Composición de imágenes terminada.')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Composition")
    parser.add_argument("--input_dir", type=str, dest="input_dir", required=True, help="Directorio de entrada. \
                        Contiene fondos ya sean pngs or jpgs, y 'foregrounds' que contienen \
                        supercategorias (e.g. 'animal', 'recibo'), con sus respectivas categorias \
                         (e.g. 'caballo', 'oso'). Cada cosa tiene una imaen vectorizada en un fondo transaparente \
                        (e.g. recibo en un fondo transparente).")
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=True, help="El directorio donde imégenes, máscaras, \
                        and json serán colocados")
    parser.add_argument("--count", type=int, dest="count", required=True, help="number of composed images to create")
    parser.add_argument("--width", type=int, dest="width", required=True, help="ancho en pixels")
    parser.add_argument("--height", type=int, dest="height", required=True, help="alto en pixels")
    parser.add_argument("--output_type", type=str, dest="output_type", help="png or jpg (default)")
    parser.add_argument("--silent", action='store_true', help="modo silencioso, \
                        sobreescribe automáticamente")

    args = parser.parse_args()

    image_comp = ImageComposition()
    image_comp.main(args)
