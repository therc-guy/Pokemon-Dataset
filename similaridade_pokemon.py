
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)
    df = df[['Name', 'Type 1', 'Type 2']].rename(columns={
        'Type 1': 'Type',
        'Type 2': 'Other Type of Pokemon'
    })
    df['combined'] = df[['Name', 'Type', 'Other Type of Pokemon']].fillna('').agg(' '.join, axis=1)
    return df

def calcular_similaridade(pokemon1, pokemon2, df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined'])

    try:
        idx1 = df[df['Name'].str.lower() == pokemon1.lower()].index[0]
        idx2 = df[df['Name'].str.lower() == pokemon2.lower()].index[0]
    except IndexError:
        return "Erro: Um dos Pokémon não foi encontrado no dataset."

    sim_score = cosine_similarity(tfidf_matrix[idx1], tfidf_matrix[idx2])[0][0]
    return f"A similaridade entre '{pokemon1}' e '{pokemon2}' é: {sim_score:.4f}"

if __name__ == "__main__":
    caminho_csv = "Pokemon.csv"  # Altere se necessário
    df_pokemon = carregar_dados(caminho_csv)

    print("Digite o nome de dois Pokémon para calcular a similaridade entre eles.")
    p1 = input("Primeiro Pokémon: ")
    p2 = input("Segundo Pokémon: ")

    resultado = calcular_similaridade(p1, p2, df_pokemon)
    print(resultado)
