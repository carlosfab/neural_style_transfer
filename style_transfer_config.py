# importar as bibliotecas necessárias
import os

content_layers = ["block5_conv2"]

style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

# pesos do modelo
style_weight = 1.0
content_weight = 1e4
tv_weight = 20.0

# epochs e steps do treinamento
epochs = 100
steps_per_epoch = 100