#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🃏 Playing Card Classification with PyTorch

Este script implementa um classificador de cartas de baralho usando PyTorch e EfficientNet.
O modelo é treinado para reconhecer 53 classes diferentes de cartas.

Autor: Your Name
Data: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# =============================================================================
# 📦 IMPORTS E CONFIGURAÇÕES
# =============================================================================

print("🚀 Iniciando Classificador de Cartas de Baralho...")

# Configuração do dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"📱 Usando dispositivo: {device}")

# =============================================================================
# 🗂️ DATASET E PREPARAÇÃO DOS DADOS
# =============================================================================

class PlayingCardDataSet(Dataset):
    """
    Dataset personalizado para classificação de cartas de baralho
    
    Esta classe encapsula o ImageFolder do torchvision para facilitar
    o uso com transformações personalizadas e acesso às classes.
    """
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

def load_and_explore_dataset():
    """
    Carrega e explora o dataset de cartas
    """
    print("\n📊 Carregando e explorando o dataset...")
    
    # Carregamento do dataset
    dataset = PlayingCardDataSet(
        data_dir='/kaggle/input/cards-image-datasetclassification/train'
    )
    print(f"📈 Total de imagens no dataset: {len(dataset)}")
    
    # Exemplo de uma imagem
    image, label = dataset[500]
    print(f"🎯 Label da imagem: {label}")
    print(f"🖼️ Formato da imagem: {image.size}")
    
    # Mapeamento de classes
    data_dir = '/kaggle/input/cards-image-datasetclassification/train'
    target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
    print(f"🎴 Total de classes: {len(target_to_class)}")
    
    return dataset, target_to_class

def create_data_loaders():
    """
    Cria os DataLoaders para treinamento, validação e teste
    """
    print("\n🔄 Criando DataLoaders...")
    
    # Transformações para as imagens
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Carregamento dos datasets com transformações
    train_folder = '/kaggle/input/cards-image-datasetclassification/train/'
    valid_folder = '/kaggle/input/cards-image-datasetclassification/valid/'
    test_folder = '/kaggle/input/cards-image-datasetclassification/test/'
    
    train_dataset = PlayingCardDataSet(train_folder, transform=transform)
    val_dataset = PlayingCardDataSet(valid_folder, transform=transform)
    test_dataset = PlayingCardDataSet(test_folder, transform=transform)
    
    # Criação dos DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"📚 Train samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    print(f"🧪 Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, train_dataset

# =============================================================================
# 🧠 ARQUITETURA DO MODELO
# =============================================================================

class SimpleCardClassifier(nn.Module):
    """
    Classificador de cartas baseado em EfficientNet
    
    Esta classe implementa um modelo de classificação que usa EfficientNet-B0
    como modelo base com transfer learning, adicionando uma camada de
    classificação personalizada para 53 classes de cartas.
    """
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        # Modelo base pré-treinado
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Camada de classificação personalizada
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

def create_and_test_model():
    """
    Cria e testa o modelo
    """
    print("\n🧠 Criando e testando o modelo...")
    
    # Instanciação do modelo
    model = SimpleCardClassifier(num_classes=53)
    print("✅ Modelo criado com sucesso!")
    
    # Teste do modelo
    model.to(device)
    
    # Teste com dados de exemplo
    for images, labels in train_loader:
        example_out = model(images.to(device))
        print(f"📊 Output shape: {example_out.shape}")
        print(f"🎯 Labels shape: {labels.shape}")
        break
    
    return model

# =============================================================================
# 🎯 CONFIGURAÇÃO DO TREINAMENTO
# =============================================================================

def setup_training(model):
    """
    Configura os parâmetros de treinamento
    """
    print("\n⚙️ Configurando parâmetros de treinamento...")
    
    # Função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"📉 Loss Function: CrossEntropyLoss")
    print(f"🚀 Optimizer: Adam (lr=0.001)")
    
    return criterion, optimizer

# =============================================================================
# 🚀 LOOP DE TREINAMENTO
# =============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    """
    Treina o modelo com validação em cada época
    """
    print(f"\n🚀 Iniciando treinamento por {num_epochs} épocas...")
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        # Fase de treinamento
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Época {epoch+1}/{num_epochs} - Treinamento'):
            # Move inputs e labels para o dispositivo
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Fase de validação
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Época {epoch+1}/{num_epochs} - Validação'):
                # Move inputs e labels para o dispositivo
                images, labels = images.to(device), labels.to(device)
             
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"📊 Época {epoch+1}/{num_epochs} - Loss de Treino: {train_loss:.4f}, Loss de Validação: {val_loss:.4f}")
    
    print("\n🎉 Treinamento concluído!")
    
    # Salvamento do modelo
    torch.save(model.state_dict(), 'card_classifier_model.pth')
    print("💾 Modelo salvo como 'card_classifier_model.pth'")
    
    return train_losses, val_losses

# =============================================================================
# 📊 VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================

def plot_training_results(train_losses, val_losses):
    """
    Plota as curvas de loss do treinamento
    """
    print("\n📈 Plotando resultados do treinamento...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Loss de Treino', linewidth=2)
    plt.plot(val_losses, label='Loss de Validação', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Evolução da Loss durante o Treinamento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Análise final
    print(f"📊 Loss final de treino: {train_losses[-1]:.4f}")
    print(f"📊 Loss final de validação: {val_losses[-1]:.4f}")
    print(f"📈 Melhoria na validação: {val_losses[0] - val_losses[-1]:.4f}")

# =============================================================================
# 🔮 SISTEMA DE PREDIÇÃO
# =============================================================================

def preprocess_image(image_path, transform):
    """
    Carrega e pré-processa uma imagem para predição
    """
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    """
    Faz predição usando o modelo treinado
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

def visualize_predictions(original_image, probabilities, class_names, top_k=5):
    """
    Visualiza a imagem original e as top-k predições
    """
    fig, axarr = plt.subplots(1, 2, figsize=(15, 7))
    
    # Exibe a imagem original
    axarr[0].imshow(original_image)
    axarr[0].set_title('Imagem de Entrada', fontsize=14, fontweight='bold')
    axarr[0].axis("off")
    
    # Exibe as predições (top-k)
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_probs = probabilities[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    bars = axarr[1].barh(range(top_k), top_probs, color='skyblue', alpha=0.8)
    axarr[1].set_yticks(range(top_k))
    axarr[1].set_yticklabels(top_classes, fontsize=10)
    axarr[1].set_xlabel('Probabilidade', fontsize=12)
    axarr[1].set_title(f'Top-{top_k} Predições', fontsize=14, fontweight='bold')
    axarr[1].set_xlim(0, 1)
    
    # Adiciona valores nas barras
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        axarr[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{prob:.3f}', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Retorna a predição mais provável
    best_class = top_classes[0]
    best_prob = top_probs[0]
    print(f"\n🎯 Predição: {best_class} (Confiança: {best_prob:.3f})")
    
    return best_class, best_prob

def test_prediction(model, dataset):
    """
    Testa o sistema de predição com uma imagem de exemplo
    """
    print("\n🔮 Testando sistema de predição...")
    
    # Exemplo de uso
    test_image = "/kaggle/input/cards-image-datasetclassification/test/five of diamonds/2.jpg"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    original_image, image_tensor = preprocess_image(test_image, transform)
    probabilities = predict(model, image_tensor, device)
    
    # Usando as classes do dataset
    class_names = dataset.classes 
    best_class, best_prob = visualize_predictions(original_image, probabilities, class_names)
    
    return best_class, best_prob

# =============================================================================
# 📈 AVALIAÇÃO DO MODELO
# =============================================================================

def evaluate_model(model, test_loader, device, class_names):
    """
    Avalia o modelo no conjunto de teste
    """
    print("\n📊 Avaliando modelo no conjunto de teste...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Avaliando modelo'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Cálculo da acurácia
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\n📊 Resultados da Avaliação:")
    print(f"🎯 Acurácia Geral: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Relatório de classificação
    print("\n📋 Relatório de Classificação:")
    print(classification_report(all_labels, all_predictions, 
                                target_names=class_names[:10], zero_division=0))
    
    return all_predictions, all_labels, accuracy

def plot_confusion_matrix(true_labels, predictions, class_names):
    """
    Plota a matriz de confusão
    """
    print("\n📈 Plotando matriz de confusão...")
    
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm[:10, :10], annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names[:10], yticklabels=class_names[:10])
    plt.title('Matriz de Confusão (Primeiras 10 Classes)', fontsize=14, fontweight='bold')
    plt.xlabel('Predições', fontsize=12)
    plt.ylabel('Valores Reais', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 🎉 FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """
    Função principal que executa todo o pipeline
    """
    print("🎉 Iniciando pipeline completo de classificação de cartas!")
    
    try:
        # 1. Carregamento e exploração dos dados
        dataset, target_to_class = load_and_explore_dataset()
        
        # 2. Criação dos DataLoaders
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders()
        
        # 3. Criação e teste do modelo
        model = create_and_test_model()
        
        # 4. Configuração do treinamento
        criterion, optimizer = setup_training(model)
        
        # 5. Treinamento do modelo
        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)
        
        # 6. Visualização dos resultados
        plot_training_results(train_losses, val_losses)
        
        # 7. Teste do sistema de predição
        best_class, best_prob = test_prediction(model, train_dataset)
        
        # 8. Avaliação completa do modelo
        predictions, true_labels, accuracy = evaluate_model(model, test_loader, device, train_dataset.classes)
        
        # 9. Matriz de confusão
        plot_confusion_matrix(true_labels, predictions, train_dataset.classes)
        
        print("\n🎉 Pipeline concluído com sucesso!")
        print(f"📊 Acurácia final: {accuracy*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
