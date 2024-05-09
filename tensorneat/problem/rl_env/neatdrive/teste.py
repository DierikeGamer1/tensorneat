import plaidml
import plaidml.keras

# Configure o dispositivo PlaidML


# Importe Keras e configure-o para usar o PlaidML
plaidml.keras.install_backend()

# Agora você pode usar o PlaidML para realizar cálculos
import keras.backend as K

# Define as dimensões das matrizes
N = 3
M = 3

# Cria duas matrizes simples usando list comprehension
matrix_a = K.variable([[i + j for j in range(M)] for i in range(N)])
matrix_b = K.variable([[i * j for j in range(M)] for i in range(N)])

# Realiza a multiplicação de matrizes usando o PlaidML
result_matrix = K.dot(matrix_a, matrix_b)

# Avalia a matriz resultante
with K.get_session() as sess:
    result = sess.run(result_matrix)

# Imprime a matriz resultante
print(result)
