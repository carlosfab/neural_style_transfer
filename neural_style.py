# importar as bibliotecas necessárias
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import Model
from PIL import Image
import tensorflow as tf
import numpy as np


class NeuralStyle(Model):
    """
    Classe com arquitetura do Estilo Neural.

    Referências:
        1. ROSEBROCK, Adrian. Deep Learning for Computer Vision
        Pactitioner Bundle. 2019.
    """
    def __init__(self, style_layers, content_layers):
        super().__init__()

        # construir a rede com o conjunto de layers
        self.vgg = self.vgg_layers(style_layers + content_layers)

        # armazenar layers e setar para serem nao-treinaveis
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        (style_outputs, content_outputs) = (
            outputs[:self.num_style_layers],
            outputs[self.num_style_layers:]
        )

        # calcular gram matrix entre os diferentes outputs de estilo
        style_outputs= [self.gramMatrix(style_output) for style_output in style_outputs]

        # preparar os dicionários
        content_dict = {
            content_name: value for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}

    @staticmethod
    def vgg_layers(layer_names):
        # carregar o modelo a partir do disco e setar para nao-treinavel
        vgg = VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False

        # criar o modelo com outputs dos layers específicos
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = Model([vgg.input], outputs)

        return model

    @ staticmethod
    def gramMatrix(input_tensor):
        result = tf.linalg.einsum("bijc,bijd->bcd",
                                  input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return result / locations

    @staticmethod
    def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]

        # style loss
        style_loss = [
            tf.reduce_mean((
            style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ]
        style_loss = tf.add_n(style_loss)
        style_loss *= style_weight

        # content loss
        content_loss = [
            tf.reduce_mean((
            content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ]
        content_loss = tf.add_n(content_loss)
        content_loss *= content_weight

        # somar style and content loss
        loss = style_loss + content_loss

        return loss

    @staticmethod
    def clip_pixels(image):
        return tf.clip_by_value(
            image,
            clip_value_min=0.0,
            clip_value_max=1.0
        )

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)

        if np.ndim(tensor) > 3:
            tensor = tensor[0]

        return Image.fromarray(tensor)