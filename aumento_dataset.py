import nltk
import random
import pandas as pd
from nltk.corpus import wordnet
import re

# Download dos recursos necessários do NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word):
    """Obtém sinônimos para uma palavra"""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Exclui o próprio termo e termos com underscore
            if lemma.name() != word and '_' not in lemma.name():
                synonyms.append(lemma.name())
    return list(set(synonyms))

def word_replacement(text, replacement_rate=0.2):
    """Substitui palavras por sinônimos baseado em uma taxa"""
    words = text.split()
    num_to_replace = max(1, int(len(words) * replacement_rate))
    
    # Identifica candidatos (palavras que têm sinônimos)
    candidates = []
    for i, word in enumerate(words):
        if len(word) > 3:  # Ignora palavras muito curtas
            synonyms = get_synonyms(word)
            if synonyms:
                candidates.append((i, word, synonyms))
    
    # Se não houver candidatos suficientes, reduz a quantidade a substituir
    num_to_replace = min(num_to_replace, len(candidates))
    
    if num_to_replace == 0:
        return text
    
    # Seleciona aleatoriamente palavras para substituir
    to_replace = random.sample(candidates, num_to_replace)
    
    # Realiza as substituições
    for i, word, synonyms in to_replace:
        words[i] = random.choice(synonyms)
    
    return ' '.join(words)

def random_insertion(text, n=1):
    """Insere n palavras aleatórias no texto"""
    words = text.split()
    if len(words) < 2:
        return text
    
    # Encontra palavras que têm sinônimos
    candidates = []
    for word in words:
        if len(word) > 3:
            synonyms = get_synonyms(word)
            if synonyms:
                candidates.append((word, synonyms))
    
    if not candidates:
        return text
    
    # Insere palavras aleatórias
    for _ in range(n):
        word, synonyms = random.choice(candidates)
        random_synonym = random.choice(synonyms)
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, random_synonym)
    
    return ' '.join(words)

def random_swap(text, n=1):
    """Troca a posição de n pares de palavras"""
    words = text.split()
    if len(words) < 2:
        return text
    
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    
    return ' '.join(words)

def random_deletion(text, p=0.1):
    """Exclui palavras com probabilidade p"""
    words = text.split()
    if len(words) < 3:
        return text
    
    # Mantém pelo menos 80% das palavras
    kept_words = []
    for word in words:
        if random.random() > p or len(kept_words) < int(len(words) * 0.8):
            kept_words.append(word)
    
    # Se todas as palavras forem excluídas, mantém uma aleatória
    if len(kept_words) == 0:
        kept_words = [random.choice(words)]
    
    return ' '.join(kept_words)

def augment_text(text, num_augmented=1):
    """Gera versões aumentadas de um texto"""
    augmented_texts = []
    
    for _ in range(num_augmented):
        # Escolhe uma técnica aleatoriamente
        technique = random.choice([
            lambda x: word_replacement(x, replacement_rate=0.2),
            lambda x: random_insertion(x, n=max(1, len(x.split()) // 20)),
            lambda x: random_swap(x, n=max(1, len(x.split()) // 20)),
            lambda x: random_deletion(x, p=0.1)
        ])
        
        # Aplica a técnica
        augmented_text = technique(text)
        
        # Corrige espaçamento excessivo
        augmented_text = re.sub(r'\s+', ' ', augmented_text).strip()
        
        if augmented_text and augmented_text != text:
            augmented_texts.append(augmented_text)
    
    # Se não conseguirmos gerar textos diferentes, faz uma última tentativa com substituição de palavras
    while len(augmented_texts) < num_augmented:
        backup_text = word_replacement(text, replacement_rate=0.3)
        if backup_text != text:
            augmented_texts.append(backup_text)
        else:
            # Se ainda falhar, apenas adiciona o texto original com pequena modificação
            augmented_texts.append(text + ".")
    
    return augmented_texts

def augment_dataset(df, text_column='data', label_column='labels', augmentation_factor=2):
    """
    Aumenta o conjunto de dados usando técnicas simples de aumento
    
    augmentation_factor: quantas vezes o dataset original será multiplicado
    """
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        text = row[text_column]
        label = row[label_column]
        
        # Adiciona o texto original
        texts.append(text)
        labels.append(label)
        
        # Gera novas amostras
        augmented_samples = augment_text(text, num_augmented=augmentation_factor-1)
        
        for augmented_text in augmented_samples:
            texts.append(augmented_text)
            labels.append(label)
    
    augmented_df = pd.DataFrame({
        text_column: texts,
        label_column: labels
    })
    
    return augmented_df

df_original = pd.read_csv('bbc_data.csv')
df_aumentado = augment_dataset(df_original, augmentation_factor=3)
df_aumentado.to_csv('bbc_data_augmented.csv', index=False)