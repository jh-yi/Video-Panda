import torch
from PIL import Image
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_image_transform(config):
    if isinstance(config, dict) and config.get('patch_stride_teacher'):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(config['image_size'], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(config['image_size']),
                transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)  # assume image
            ]
        )

    else:
        config = config.vision_config
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)  # assume image
            ]
        )
    return transform


def load_and_transform_image(image_path, transform):
    image = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path
    image_outputs = transform(image)
    return image_outputs

class LanguageBindImageProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindImageTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_image_transform(config)
        self.image_processor = load_and_transform_image
        self.tokenizer = tokenizer
        self.image_mean = OPENAI_DATASET_MEAN
        self.crop_size = {'height': 224, 'width': 224}

        if isinstance(config, dict) and config.get('patch_stride_teacher'):
            self.image_size = config['image_size']
            self.crop_size = {'height': config['crop_size'], 'width': config['crop_size']}
            self.patch_stride = config['patch_stride']
            self.conv_stride = config['conv_stride']
            self.image_size_teacher = config['image_size_teacher']
            self.patch_stride_teacher = config['patch_stride_teacher']

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            image_features = [self.image_processor(image, self.transform) for image in images]
            image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
