from librosa import sequence, load, feature
import numpy as np
from scipy.spatial.distance import euclidean


#audio1_path: uma gravação do som alvo.
#audio2_path: outra gravação do som alvo.
#audio3_path: uma gravação de um som diferente.

audio1_path = r"caminho do audio"
audio2_path = r"caminho do audio"
audio3_path = r"caminho do audio"

#Carrega os áudios.
try:
    y1, sr1 = load(audio1_path)
    y2, sr2 = load(audio2_path)
    y3, sr3 = load(audio3_path)
except FileNotFoundError:
    print("Caminho incorreto!")

# Extrai os MFCCs (Mel-Frequency Cepstral Coefficients)
# Usar os mesmos parâmetros para todas as extrações.
mfcc1 = feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
mfcc2 = feature.mfcc(y=y2, sr=sr2, n_mfcc=13)
mfcc3 = feature.mfcc(y=y3, sr=sr3, n_mfcc=13)

#Comparação 1: Dois sons que deveriam ser iguais.
# Calcula a matriz de custo e o caminho ótimo usando DTW
D_12, wp_12 = sequence.dtw(X=mfcc1, Y=mfcc2, metric='euclidean')
# A distância DTW normalizada é o último valor na matriz de custo acumulado
# Dividimos pelo número de passos no caminho para normalizar pela duração (custo médio por passo de alinhamento)
dtw_distance_12 = D_12[-1, -1] / len(wp_12)

#Comparação 2: Audio 1 diferente do audio 3
D_13, wp_13 = sequence.dtw(X=mfcc1, Y=mfcc3, metric='euclidean')
dtw_distance_13 = D_13[-1, -1] / len(wp_13)

#Comparação 3: Audio 2 diferente do audio 3
D_14, wp_14 = sequence.dtw(X=mfcc2, Y=mfcc3, metric='euclidean')
dtw_distance_14 = D_14[-1, -1] / len(wp_14)

print(f"Distância DTW Normalizada (som alvo 1 vs som alvo 2): {dtw_distance_12:.4f}")

print(f"Distância DTW Normalizada (som alvo 1 vs som diferente): {dtw_distance_13:.4f}")

print(f"Distância DTW Normalizada (som alvo 2 vs som diferente): {dtw_distance_14:.4f}")

# Definição o Threshold.
THRESHOLD = 40.8 # Valor a ser ajustado para casa caso(verificar após testes de valores com audios parecidos e diferentes)

if dtw_distance_12 < THRESHOLD:
    print("\nConclusão 1: Os dois primeiros sons são considerados IGUAIS.")
else:
    print("\nConclusão 1: Os dois primeiros sons são considerados DIFERENTES.")

if dtw_distance_13 < THRESHOLD:
    print("Conclusão 2: O primeiro e o terceiro som são considerados IGUAIS.")
else:
    print("Conclusão 2: O primeiro e o terceiro som são considerados DIFERENTES.")

if dtw_distance_14 < THRESHOLD:
    print("Conclusão 2: O segundo e o terceiro som são considerados IGUAIS.")
else:
    print("Conclusão 2: O segundo e o terceiro som são considerados DIFERENTES.")