# -*- encoding: utf-8 -*-

# NEURAL STYLE TRANSFER
# @carlos_melo.py
#
# Script para transferir estilos entre duas imagens diferentes
#
# As principais referências teóricas e códigos usados neste
# script estão listadas abaixo.
# Em especial, a classe NeuralStyle e o pipeline de
# carregamento das imagens, foram extraídas ou adaptados do livro
# Deep Learning for Computer Vision, do autor Adrian Rosebrock
#
#
# REFERÊNCIAS:
# 1. ROSEBROCK, A. Deep Learning for Computer Vision
#   Pactitioner Bundle. Pyimagesearch. 2019.
# 2. GATYS, L.; ECKER, A.; BETHGE, M. A Neural Algorithm of Artistic
#   Style. Journal of Vision, v. 16, n. 12, p. 326. 2016.
# 3. GATYS, L. A.; ECKER, A. S.; BETHGE, M. Image Style Transfer
#   Using Convolutional Neural Networks. 2016.



# importar as bibliotecas necessarias
import argparse
import os
from neural_style import NeuralStyle
import tensorflow as tf
from pyfiglet import Figlet
from PyInquirer import prompt
import progressbar
import style_transfer_config as config
import time


def welcome_banner():
    """""Banner da tela inicial """

    f = Figlet(font='drpepper')
    print(f.renderText('Sigmoidal'))
    g = Figlet(font='digital')
    print(g.renderText('neural style transfer'))
    print("Carlos Melo\n\n\n")


def get_args():
    """Extrai os argumentos informados pelo usuário."""

    # cria um ArgmentParser e passa os argumentos para a variável args
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='Caminho da imagem com o contexto.')
    ap.add_argument('-s', '--save-intermediate', default=-1, type=int, help='Salvar as imagens intermediárias?')

    # retorna um diconário com os argumentos
    return vars(ap.parse_args())


def get_style():
    """Apresenta uma lista de estilos disponíveis o usuário escolher."""

    questions = [{
        'type': 'list',
        'name': 'style',
        'message': 'Qual estilo você quer aplicar?',
        'choices': [
            'Van Gogh',
            'Kanagawa',
            'Monet',
            'Caspar David Friedrich',
            'Picasso'
        ],
        'filter': lambda val: val.lower().replace(' ', '')
    }]

    user_style = prompt(questions)

    # retorna o estilo escolhido pelo usuário
    return user_style


def load_image(image_path):
    """Carrega e redimensiona a imagem para o tamanho máximo."""

    max_dim = 512

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]

    return image


@tf.function
def train_one_step(image, style_targets, content_targets):
    style_weight = config.style_weight / len(config.style_layers)
    content_weight = config.content_weight / len(config.content_layers)

    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = extractor.style_content_loss(outputs, style_targets,
                                          content_targets, style_weight, content_weight)
        loss += config.tv_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(extractor.clip_pixels(image))


# iniciar script e extrair os argumentos informados na execução
os.system('cls' if os.name == 'nt' else 'clear')
welcome_banner()
args = get_args()
user_style = get_style()

# armazenar os argumentos informados
content_image_path = args["image"]
style_image_path = os.path.sep.join(["styles", "{}.jpg".format(user_style["style"])])
final_output = "output"
final_image = os.path.sep.join([final_output, "final_{}.jpg".format(user_style["style"])])
interm_outputs = "intermediate_outputs"

# criar os diretorio intermediate_outputs e output, caso não existam
if not os.path.isdir(interm_outputs):
    os.mkdir(interm_outputs)

if not os.path.isdir(final_output):
    os.mkdir(final_output)

# inicializar o Adam optimizer
opt = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.99,
                         epsilon=1e-1)

# carregar as imagens de contexto e estilo the content and style images
print("\n\n[INFO] carregando as imagens de contexto e estilo...\n")
time.sleep(2)
content_image = load_image(content_image_path)
style_image = load_image(style_image_path)

# extrair os layers de acordo com o arquivo de configuração
content_layers = config.content_layers
style_layers = config.style_layers

# inicializar a arquitetura para extrair as features das imagens de contexto e estilo
extractor = NeuralStyle(style_layers, content_layers)
style_targets = extractor(style_image)["style"]
content_targets = extractor(content_image)["content"]

# iniciar a fase de treino
os.system('cls' if os.name == 'nt' else 'clear')
print("[INFO] treinando o modelo de transferência de estilo neural...\n\n")
image = tf.Variable(content_image)
step = 0

# instanciar uma Progress Bar
bar = progressbar.ProgressBar(max_value=config.epochs * config.steps_per_epoch)

# iterar ao longo das épocas
for epoch in range(config.epochs):
    # iterar ao longo dos steps
    for i in range(config.steps_per_epoch):
        # treinar um step e atualizar o contador
        train_one_step(image, style_targets, content_targets)
        step += 1

        # atualiza a progress bar
        bar.update(step)

    # salvar a imagem intermediária
    if args["save_intermediate"] > 0:
        int_name = "_".join([str(epoch), str(i)])
        int_name = "{}.png".format(int_name)
        int_name = os.path.join(interm_outputs, int_name)
        extractor.tensor_to_image(image).save(int_name)

print("\n\n[+] treinamento concluído!")

# salvar a imagem final
extractor.tensor_to_image(image).save(final_image)
