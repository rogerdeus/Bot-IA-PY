import numpy as np

# Inicialização
print("1- Criptografia")
print("2- Descriptografia")
print("0- Sair")

seletor = input("VAMOS JOGAR? ")

if seletor == "1":  # Adicionado aspas para comparar string
    alpha = input("Digite algo: ").upper()  # Entrada sofre conversão para maiúsculo
    matriz_alpha = []
    numeros = []

    matriz_bheta = np.array([[-8, 12], [45, -78], [1, 0]])  # Matriz 3x2

    for i in range(len(alpha)):
        char = alpha[i]
        if 'A' <= char <= 'Z':
            num = ord(char) - ord('A')  # Mapeia A=0, B=1, ..., Z=25
            numeros.append(num)
            # print(f"'{char}' {num}")  # Comentado para depuração opcional
        elif char == '#':
            num = 26  # Mapeia "#" como 26
            numeros.append(num)
            # print(f"'{char}' {num}")  # Comentado para depuração opcional
        else:
            print(f"'{char}' não é uma letra ou # válido (ignorado).")

    # Organiza os números em blocos de 3, completando com 26 se necessário
    for i in range(0, len(numeros), 3):
        bloco = numeros[i:i + 3]  # Pega até 3 números
        while len(bloco) < 3:  # Completa com 26 se tiver menos de 3
            bloco.append(26)
        matriz_alpha.append(bloco)

    # Converte para array NumPy
    matriz_alpha_np = np.array(matriz_alpha)

    # Multiplicação usando np.dot() (compatível com 3 colunas e 3 linhas)
    result = np.dot(matriz_alpha_np, matriz_bheta)
    print("Matriz de criptografia:\n", result)